from threading import Lock

from helios.config import HeliosConfig
from helios.runtime.generate import GenerationResult, Generator
from helios.runtime.load import Loader
from helios.runtime.types import Sampling


class Engine:
    def __init__(self, config: HeliosConfig, loader: Loader | None = None) -> None:
        loaded = (loader or Loader()).load(config)
        self.model_id = config.model_id
        self.model_revision = loaded.model_revision
        self.report = loaded.report
        self.generator = Generator(
            loaded.model,
            loaded.cache,
            prefix_cache_ttl_seconds=config.prefix_cache_ttl_seconds,
        )
        self._generation_lock = Lock()

    def prefix_cache_snapshot(self) -> dict[str, object]:
        with self._generation_lock:
            cache = self.generator.prefix_cache
            blocks = cache.blocks()
            return {
                "block_size": cache.block_size,
                "occupied_blocks": len(blocks),
                "cached_tokens": cache.token_count,
                "memory_bytes": cache.memory_bytes,
                "max_blocks": None,
                "max_memory_bytes": None,
                "blocks": [block.as_dict() for block in blocks],
            }

    def run(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
    ) -> GenerationResult:
        vocabulary_size = self.generator.decoder.model.config.vocab_size
        if eos_token_id >= vocabulary_size or any(
            token_id >= vocabulary_size for token_id in input_ids
        ):
            raise ValueError(
                f"Token IDs must be smaller than the model vocabulary size ({vocabulary_size:,})."
            )
        with self._generation_lock:
            return self.generator.run(input_ids, eos_token_id, sampling)
