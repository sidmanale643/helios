from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import torch

from helios.config import HeliosConfig
from helios.runtime.qwen3.config import Qwen3Config, qwen3_4b_config
from helios.runtime.qwen3.model import Qwen3Model


@dataclass(frozen=True)
class DeviceInfo:
    type: str
    name: str
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    uses_unified_memory: bool


@dataclass(frozen=True)
class WeightEstimate:
    parameter_count: int
    weight_bytes: int
    required_bytes: int
    dtype: str


@dataclass(frozen=True)
class CacheCapacity:
    available_after_model_load_bytes: int
    headroom_bytes: int
    cache_budget_bytes: int
    bytes_per_token: int
    max_tokens: int
    dtype: str


@dataclass(frozen=True)
class MemoryReport:
    device: DeviceInfo
    weights: WeightEstimate
    fits: bool
    reason: str
    cache: CacheCapacity | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryChecker:

    def __init__(self, config: HeliosConfig) -> None:
        self.config = config

    def available(self, device: DeviceInfo) -> int | None:
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            return free
        _, available = self._system_memory()
        return available

    def device(self) -> DeviceInfo:
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(index)
            return DeviceInfo("cuda", torch.cuda.get_device_name(index), total, free, False)

        if torch.backends.mps.is_available():
            total, available = self._system_memory()
            return DeviceInfo("mps", "Apple Metal", total, available, True)

        total, available = self._system_memory()
        return DeviceInfo("cpu", platform.processor() or platform.machine() or "CPU", total, available, False)

    def dtype(self, device: DeviceInfo) -> torch.dtype:
        return torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    def weights_for(self, device: DeviceInfo | None = None) -> MemoryReport:
        device = device or self.device()
        dtype = self.dtype(device)
        with torch.device("meta"):
            model = Qwen3Model(qwen3_4b_config(dtype))
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
        del model

        weight_bytes = parameter_count * torch.empty((), dtype=dtype).element_size()
        required_bytes = int(weight_bytes * (1 + self.config.weight_headroom_ratio))
        available = device.available_memory_bytes
        weights = WeightEstimate(parameter_count, weight_bytes, required_bytes, str(dtype))
        if available is None:
            return MemoryReport(device, weights, False, "Available memory could not be measured; refusing to load safely.")
        fits = required_bytes <= available
        reason = (
            "Estimated model weights fit in currently available memory."
            if fits
            else "Estimated model weights exceed currently available memory."
        )
        return MemoryReport(device, weights, fits, reason)

    def cache_for(self, model_config: Qwen3Config, device: DeviceInfo) -> CacheCapacity:
        available = self.available(device)
        if available is None:
            raise RuntimeError("Unable to measure memory after model load; refusing to size the KV cache safely.")

        layers = model_config.n_layers
        kv_heads = model_config.n_kv_heads
        head_dim = model_config.head_dim
        dtype = self.dtype(device)
        bytes_per_token = (
            layers * 2 * kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()
        )
        cache_budget_bytes = int(available / (1 + self.config.kv_cache_headroom_ratio))
        headroom_bytes = available - cache_budget_bytes
        max_tokens = cache_budget_bytes // bytes_per_token
        max_tokens = min(max_tokens, model_config.context_length)
        if max_tokens < 1:
            raise RuntimeError("No memory remains for even one KV-cache token after model load and headroom.")

        return CacheCapacity(
            available_after_model_load_bytes=available,
            headroom_bytes=headroom_bytes,
            cache_budget_bytes=cache_budget_bytes,
            bytes_per_token=bytes_per_token,
            max_tokens=max_tokens,
            dtype=str(dtype),
        )

    @staticmethod
    def _system_memory() -> tuple[int | None, int | None]:
        if platform.system() == "Darwin":
            return MemoryChecker._macos_memory()
        return MemoryChecker._linux_memory()

    @staticmethod
    def _macos_memory() -> tuple[int | None, int | None]:
        try:
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
            pages = subprocess.check_output(["vm_stat"], text=True)
            page_size = int(next(line for line in pages.splitlines() if "page size of" in line).split("page size of ")[1].split(" bytes")[0])
            counters = {}
            for line in pages.splitlines()[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    counters[key] = int(value.strip().rstrip("."))
            reclaimable = sum(
                counters.get(key, 0)
                for key in ("Pages free", "Pages inactive", "Pages speculative")
            ) * page_size
            return total, reclaimable
        except (OSError, StopIteration, ValueError):
            return None, None

    @staticmethod
    def _linux_memory() -> tuple[int | None, int | None]:
        try:
            fields = {}
            with open("/proc/meminfo", encoding="utf-8") as memory_info:
                for line in memory_info:
                    key, value = line.split(":", 1)
                    fields[key] = int(value.split()[0]) * 1024
            return fields.get("MemTotal"), fields.get("MemAvailable")
        except (OSError, ValueError):
            return None, None
