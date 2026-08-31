from dataclasses import dataclass, replace
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from helios.config import HeliosConfig
from helios.runtime.check import CacheCapacity, MemoryChecker, MemoryReport
from helios.runtime.qwen3.config import QWEN3_4B_MODEL_ID, qwen3_4b_config
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.qwen3.weights import Qwen3Weights


@dataclass
class LoadedQwen3:
    model: Qwen3Model
    cache: CacheCapacity
    report: MemoryReport
    model_revision: str


class Qwen3Loader:
    def load(self, config: HeliosConfig) -> LoadedQwen3:
        if config.model_id != QWEN3_4B_MODEL_ID:
            raise ValueError(
                f"Helios's native runtime supports {QWEN3_4B_MODEL_ID}; "
                f"received {config.model_id}."
            )
        checker = MemoryChecker(config)
        report = checker.weights()
        if not report.fits:
            raise RuntimeError(
                f"Refusing to load {config.model_id}: weights and headroom need "
                f"{report.required_bytes:,} bytes; {report.free_before_load_bytes:,} "
                "bytes are free."
            )
        snapshot = Path(
            snapshot_download(
                repo_id=config.model_id,
                revision=config.model_revision,
                token=config.hf_token,
                allow_patterns=["model*.safetensors", "model.safetensors.index.json"],
            )
        )
        native_config = qwen3_4b_config(checker.dtype)
        with torch.device("cuda"):
            model = Qwen3Model(native_config)
        weights = Qwen3Weights(snapshot)
        weights.load_into(model)
        model.eval()
        cache = checker.cache(native_config)
        report = replace(report, cache=cache)
        return LoadedQwen3(model, cache, report, snapshot.name)
