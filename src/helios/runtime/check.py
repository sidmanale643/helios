import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from helios.config import HeliosConfig
from helios.runtime.qwen3.config import Qwen3Config, qwen3_4b_config
from helios.runtime.qwen3.model import Qwen3Model


@dataclass(frozen=True)
class CacheCapacity:
    free_bytes: int
    device_occupied_bytes: int
    activation_headroom_bytes: int
    bytes_per_token: int
    max_tokens: int
    kv_budget_bytes: int
    model_occupied_bytes: int
    warmup_peak_bytes: int | None = None


@dataclass(frozen=True)
class MemoryReport:
    gpu: str
    total_bytes: int
    max_gpu_utilization: float
    max_gpu_bytes: int
    free_before_load_bytes: int
    weight_bytes: int
    required_bytes: int
    fits: bool
    cache: CacheCapacity | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryChecker:
    dtype = torch.float16

    def __init__(self, config: HeliosConfig) -> None:
        self.config = config

    def weights(self) -> MemoryReport:
        index = self._gpu()
        free, total = torch.cuda.mem_get_info(index)
        max_gpu_bytes = math.floor(total * self.config.max_gpu_utilization)
        with torch.device("meta"):
            model = Qwen3Model(qwen3_4b_config(self.dtype))
            weight_bytes = sum(parameter.numel() for parameter in model.parameters()) * 2

        required_bytes = int(weight_bytes * (1 + self.config.weight_headroom_ratio))
        available_bytes = max(0, max_gpu_bytes - (total - free))
        return MemoryReport(
            gpu=torch.cuda.get_device_name(index),
            total_bytes=total,
            max_gpu_utilization=self.config.max_gpu_utilization,
            max_gpu_bytes=max_gpu_bytes,
            free_before_load_bytes=free,
            weight_bytes=weight_bytes,
            required_bytes=required_bytes,
            fits=required_bytes <= available_bytes,
        )

    def cache(
        self,
        model: Qwen3Config,
        *,
        warmup_peak_bytes: int | None = None,
        warmup_kv_bytes: int = 0,
    ) -> CacheCapacity:
        device = self._gpu()
        free, total = torch.cuda.mem_get_info(device)
        bytes_per_token = (
            model.n_layers * 2 * model.n_kv_heads * model.head_dim * 2
        )
        model_occupied_bytes = torch.cuda.memory_reserved(device)
        max_gpu_bytes = math.floor(total * self.config.max_gpu_utilization)
        if warmup_peak_bytes is None:
            activation_headroom_bytes = free - int(
                free / (1 + self.config.kv_cache_headroom_ratio)
            )
        else:
            if warmup_peak_bytes < 0 or warmup_kv_bytes < 0:
                raise ValueError("Warmup peak memory must be non-negative.")
            activation_bytes = max(0, warmup_peak_bytes - warmup_kv_bytes)
            activation_headroom_bytes = math.ceil(
                activation_bytes * (1 + self.config.kv_cache_headroom_ratio)
            )
        device_occupied_bytes = total - free
        cache_bytes = (
            max_gpu_bytes - device_occupied_bytes - activation_headroom_bytes
        )
        max_tokens = min(cache_bytes // bytes_per_token, model.context_length)
        if max_tokens < 1:
            raise RuntimeError(
                "Not enough GPU memory remains for the KV cache and activations."
            )
        return CacheCapacity(
            free_bytes=free,
            device_occupied_bytes=device_occupied_bytes,
            activation_headroom_bytes=activation_headroom_bytes,
            bytes_per_token=bytes_per_token,
            max_tokens=max_tokens,
            kv_budget_bytes=cache_bytes,
            model_occupied_bytes=model_occupied_bytes,
            warmup_peak_bytes=warmup_peak_bytes,
        )

    @staticmethod
    def _gpu() -> int:
        if not torch.cuda.is_available():
            raise RuntimeError("Helios requires a CUDA-capable NVIDIA GPU.")
        return torch.cuda.current_device()
