import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class HeliosConfig:
    model_id: str
    hf_token: str | None
    model_revision: str | None = None
    max_gpu_utilization: float = 0.90
    weight_headroom_ratio: float = 0.20
    kv_cache_headroom_ratio: float = 0.20
    prefix_cache_ttl_seconds: float = 300.0
    prefill_chunk_size: int = 256
    max_batch_size: int = 8
    max_queue_size: int = 32
    batch_wait_ms: float = 2.0
    speculative_tokens: int = 0
    torch_compile: bool = False
    compile_fullgraph: bool = False
    compile_diagnostics: bool = True

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
        if (
            not isinstance(self.prefill_chunk_size, int)
            or isinstance(self.prefill_chunk_size, bool)
            or self.prefill_chunk_size < 1
        ):
            raise ValueError("prefill_chunk_size must be a positive integer.")
        if (
            not isinstance(self.speculative_tokens, int)
            or isinstance(self.speculative_tokens, bool)
            or not 0 <= self.speculative_tokens <= 16
        ):
            raise ValueError("speculative_tokens must be an integer from 0 to 16.")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1.")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1.")
        if not math.isfinite(self.batch_wait_ms) or self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be finite and non-negative.")
        if self.compile_fullgraph and not self.torch_compile:
            raise ValueError("compile_fullgraph requires torch_compile.")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise ValueError(f"{name} must be true, false, 1, or 0.")


def get_config() -> HeliosConfig:
    load_dotenv()
    return HeliosConfig(
        model_id=os.getenv("HELIOS_MODEL_ID", "Qwen/Qwen3-4B"),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY"),
        model_revision=os.getenv("HELIOS_MODEL_REVISION") or None,
        max_gpu_utilization=float(os.getenv("HELIOS_MAX_GPU_UTILIZATION", "0.90")),
        weight_headroom_ratio=float(os.getenv("HELIOS_WEIGHT_HEADROOM_RATIO", "0.20")),
        kv_cache_headroom_ratio=float(
            os.getenv("HELIOS_KV_CACHE_HEADROOM_RATIO", "0.20")
        ),
        prefix_cache_ttl_seconds=float(
            os.getenv("HELIOS_PREFIX_CACHE_TTL_SECONDS", "300")
        ),
        prefill_chunk_size=int(os.getenv("HELIOS_PREFILL_CHUNK_SIZE", "256")),
        max_batch_size=int(os.getenv("HELIOS_MAX_BATCH_SIZE", "8")),
        max_queue_size=int(os.getenv("HELIOS_MAX_QUEUE_SIZE", "32")),
        batch_wait_ms=float(os.getenv("HELIOS_BATCH_WAIT_MS", "2")),
        speculative_tokens=int(os.getenv("HELIOS_SPECULATIVE_TOKENS", "0")),
        torch_compile=_env_bool("HELIOS_TORCH_COMPILE", False),
        compile_fullgraph=_env_bool("HELIOS_COMPILE_FULLGRAPH", False),
        compile_diagnostics=_env_bool("HELIOS_COMPILE_DIAGNOSTICS", True),
    )
