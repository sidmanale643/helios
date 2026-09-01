import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class HeliosConfig:
    model_id: str
    hf_token: str | None
    model_revision: str | None = None
    weight_headroom_ratio: float = 0.20
    kv_cache_headroom_ratio: float = 0.20
    prefix_cache_ttl_seconds: float = 300.0

    def __post_init__(self) -> None:
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


def get_config() -> HeliosConfig:
    load_dotenv()
    return HeliosConfig(
        model_id=os.getenv("HELIOS_MODEL_ID", "Qwen/Qwen3-4B"),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY"),
        model_revision=os.getenv("HELIOS_MODEL_REVISION") or None,
        weight_headroom_ratio=float(os.getenv("HELIOS_WEIGHT_HEADROOM_RATIO", "0.20")),
        kv_cache_headroom_ratio=float(
            os.getenv("HELIOS_KV_CACHE_HEADROOM_RATIO", "0.20")
        ),
        prefix_cache_ttl_seconds=float(
            os.getenv("HELIOS_PREFIX_CACHE_TTL_SECONDS", "300")
        ),
    )
