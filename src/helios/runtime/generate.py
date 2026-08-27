from helios.runtime.check import CacheCapacity
from helios.runtime.protocol import GenerateCommand, GenerateResult, GenerationTiming
from helios.runtime.qwen3.decode import Decoder
from helios.runtime.qwen3.model import Qwen3Model


class Generator:
    def __init__(self, model: Qwen3Model, cache: CacheCapacity) -> None:
        self.decoder = Decoder(model)
        self.cache = cache

    def run(self, command: GenerateCommand) -> GenerateResult:
        output_ids, finish_reason, timing = self.decoder.generate(
            command.input_ids,
            command.eos_token_id,
            command.sampling,
            max_total_tokens=self.cache.max_tokens,
        )
        return GenerateResult(
            request_id=command.request_id,
            output_ids=output_ids,
            finish_reason=finish_reason,
            timing=GenerationTiming(**timing),
        )
