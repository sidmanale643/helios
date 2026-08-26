import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class HeliosConfig:
    model_id: str
    hf_token: str | None
    model_revision: str | None = None
    scheduler_endpoint: str = "tcp://127.0.0.1:5555"
    scheduler_timeout_ms: int = 600_000
    weight_headroom_ratio: float = 0.20
    kv_cache_headroom_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.scheduler_timeout_ms < 1:
            raise ValueError("scheduler_timeout_ms must be at least 1.")
        for name, value in (
            ("weight_headroom_ratio", self.weight_headroom_ratio),
            ("kv_cache_headroom_ratio", self.kv_cache_headroom_ratio),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be at least 0 and less than 1.")


def get_config() -> HeliosConfig:
    load_dotenv()
    return HeliosConfig(
        model_id=os.getenv("HELIOS_MODEL_ID", "Qwen/Qwen3-4B"),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY"),
        model_revision=os.getenv("HELIOS_MODEL_REVISION") or None,
        scheduler_endpoint=os.getenv(
            "HELIOS_SCHEDULER_ENDPOINT", "tcp://127.0.0.1:5555"
        ),
        scheduler_timeout_ms=int(os.getenv("HELIOS_SCHEDULER_TIMEOUT_MS", "600000")),
        weight_headroom_ratio=float(os.getenv("HELIOS_WEIGHT_HEADROOM_RATIO", "0.20")),
        kv_cache_headroom_ratio=float(
            os.getenv("HELIOS_KV_CACHE_HEADROOM_RATIO", "0.20")
        ),
    )
