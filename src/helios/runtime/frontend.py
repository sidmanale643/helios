from helios.runtime.client import SchedulerClient
from helios.runtime.protocol import HealthResult
from helios.runtime.types import GenerateRequest
from helios.runtime.worker import Tokenizer


class TextGenerator:

    def __init__(self, tokenizer: Tokenizer, client: SchedulerClient) -> None:
        self.tokenizer = tokenizer
        self.client = client

    def run(self, request: GenerateRequest) -> str:
        input_ids = self.tokenizer.tokenize(request.text)
        result = self.client.generate(
            model_id=self.tokenizer.model_id,
            model_revision=self.tokenizer.model_revision,
            input_ids=input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            sampling=request.sampling,
        )
        return self.tokenizer.detokenize(result.output_ids)

    def health(self) -> HealthResult:
        result = self.client.health()
        if (
            result.model_id != self.tokenizer.model_id
            or result.model_revision != self.tokenizer.model_revision
        ):
            raise RuntimeError(
                "Tokenizer and scheduler model snapshots do not match: "
                f"tokenizer={self.tokenizer.model_id}@{self.tokenizer.model_revision}, "
                f"model={result.model_id}@{result.model_revision}."
            )
        return result
