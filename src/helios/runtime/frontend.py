from dataclasses import dataclass

from helios.runtime.engine import Engine
from helios.runtime.generate import GenerationResult
from helios.runtime.types import GenerateRequest, Sampling
from helios.runtime.worker import Tokenizer


@dataclass(frozen=True)
class ChatGeneration:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class TextGenerator:

    def __init__(self, tokenizer: Tokenizer, engine: Engine) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        if (
            tokenizer.model_id != engine.model_id
            or tokenizer.model_revision != engine.model_revision
        ):
            raise RuntimeError(
                "Tokenizer and model snapshots do not match: "
                f"tokenizer={tokenizer.model_id}@{tokenizer.model_revision}, "
                f"model={engine.model_id}@{engine.model_revision}."
            )

    def run(self, request: GenerateRequest) -> str:
        input_ids = self.tokenizer.tokenize(request.text)
        result = self._generate(input_ids, request.sampling)
        return self.tokenizer.detokenize(result.output_ids)

    def run_chat(
        self,
        messages: list[tuple[str, str]],
        sampling: Sampling,
    ) -> ChatGeneration:
        input_ids = self.tokenizer.tokenize_chat(messages)
        result = self._generate(input_ids, sampling)
        return ChatGeneration(
            text=self.tokenizer.detokenize(result.output_ids),
            finish_reason=result.finish_reason,
            prompt_tokens=len(input_ids),
            completion_tokens=len(result.output_ids),
        )

    @property
    def model_id(self) -> str:
        return self.tokenizer.model_id

    def _generate(self, input_ids: list[int], sampling: Sampling) -> GenerationResult:
        return self.engine.run(
            input_ids,
            self.tokenizer.eos_token_id,
            sampling,
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model": self.engine.model_id,
            "model_revision": self.engine.model_revision,
            "memory": self.engine.report.as_dict(),
        }
