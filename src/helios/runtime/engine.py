from helios.config import HeliosConfig
from helios.runtime.generate import Generator
from helios.runtime.load import Loader
from helios.runtime.protocol import GenerateCommand, GenerateResult


class Engine:
    def __init__(self, config: HeliosConfig, loader: Loader | None = None) -> None:
        loaded = (loader or Loader()).load(config)
        self.model_id = config.model_id
        self.model_revision = loaded.model_revision
        self.report = loaded.report
        self.generator = Generator(loaded.model, loaded.cache)

    def run(self, command: GenerateCommand) -> GenerateResult:
        if (
            command.model_id != self.model_id
            or command.model_revision != self.model_revision
        ):
            raise ModelMismatchError(
                "Tokenizer and model snapshots do not match: "
                f"tokenizer={command.model_id}@{command.model_revision}, "
                f"model={self.model_id}@{self.model_revision}."
            )
        vocabulary_size = self.generator.decoder.model.config.vocab_size
        if command.eos_token_id >= vocabulary_size or any(
            token_id >= vocabulary_size for token_id in command.input_ids
        ):
            raise ValueError(
                f"Token IDs must be smaller than the model vocabulary size ({vocabulary_size:,})."
            )
        return self.generator.run(command)


class ModelMismatchError(ValueError):
    pass
