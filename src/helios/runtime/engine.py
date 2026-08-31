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
        self.generator = Generator(loaded.model, loaded.cache)
        self._generation_lock = Lock()

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
