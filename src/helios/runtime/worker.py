from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from helios.config import HeliosConfig
from helios.runtime.qwen3.tokenizer import Qwen3Tokenizer


@dataclass(frozen=True)
class Tokenizer:

    tokenizer: Qwen3Tokenizer
    model_id: str
    model_revision: str

    @classmethod
    def load(cls, config: HeliosConfig) -> "Tokenizer":
        snapshot = Path(
            snapshot_download(
                repo_id=config.model_id,
                revision=config.model_revision,
                token=config.hf_token,
                allow_patterns=["tokenizer.json"],
            )
        )
        return cls(
            tokenizer=Qwen3Tokenizer(snapshot / "tokenizer.json"),
            model_id=config.model_id,
            model_revision=snapshot.name,
        )

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
