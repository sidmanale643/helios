import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class HeliosConfig:
    model_id: str
    hf_token: str | None
    model_revision: str | None = None
    torch_compile: bool = False
    max_gpu_utilization: float = 0.90
    weight_headroom_ratio: float = 0.20
    kv_cache_headroom_ratio: float = 0.20
    prefix_cache_ttl_seconds: float = 300.0
    max_batch_size: int = 8
    max_queue_size: int = 32
    batch_wait_ms: float = 2.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_gpu_utilization)
            or not 0 < self.max_gpu_utilization <= 1
        ):
            raise ValueError(
                "max_gpu_utilization must be greater than 0 and at most 1."
            )
        for name, value in (
            ("weight_headroom_ratio", self.weight_headroom_ratio),
            ("kv_cache_headroom_ratio", self.kv_cache_headroom_ratio),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be at least 0 and less than 1.")
        if (
            not math.isfinite(self.prefix_cache_ttl_seconds)
            or self.prefix_cache_ttl_seconds <= 0
        ):
            raise ValueError(
                "prefix_cache_ttl_seconds must be finite and greater than 0."
            )
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1.")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1.")
        if not math.isfinite(self.batch_wait_ms) or self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be finite and non-negative.")


def get_config() -> HeliosConfig:
    load_dotenv()
    return HeliosConfig(
        model_id=os.getenv("HELIOS_MODEL_ID", "Qwen/Qwen3-4B"),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY"),
        model_revision=os.getenv("HELIOS_MODEL_REVISION") or None,
        torch_compile=os.getenv("HELIOS_TORCH_COMPILE", "0").lower()
        not in {"0", "false", "no"},
        max_gpu_utilization=float(os.getenv("HELIOS_MAX_GPU_UTILIZATION", "0.90")),
        weight_headroom_ratio=float(os.getenv("HELIOS_WEIGHT_HEADROOM_RATIO", "0.20")),
        kv_cache_headroom_ratio=float(
            os.getenv("HELIOS_KV_CACHE_HEADROOM_RATIO", "0.20")
        ),
        prefix_cache_ttl_seconds=float(
            os.getenv("HELIOS_PREFIX_CACHE_TTL_SECONDS", "300")
        ),
        max_batch_size=int(os.getenv("HELIOS_MAX_BATCH_SIZE", "8")),
        max_queue_size=int(os.getenv("HELIOS_MAX_QUEUE_SIZE", "32")),
        batch_wait_ms=float(os.getenv("HELIOS_BATCH_WAIT_MS", "2")),
    )
