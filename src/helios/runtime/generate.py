from dataclasses import dataclass

from helios.runtime.check import CacheCapacity
from helios.runtime.prefix_cache import PrefixCache
from helios.runtime.qwen3.decode import Decoder
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling

PREFIX_CACHE_BLOCK_SIZE = 16


@dataclass(frozen=True)
class GenerationResult:
    output_ids: list[int]
    finish_reason: str
    prefill_seconds: float
    inter_token_seconds: list[float]


class Generator:
    def __init__(self, model: Qwen3Model, cache: CacheCapacity) -> None:
        self.decoder = Decoder(model)
        self.cache = cache
        self.prefix_cache = PrefixCache(block_size=PREFIX_CACHE_BLOCK_SIZE)

    def run(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
    ) -> GenerationResult:
        prefix_hit = self.prefix_cache.longest_prefix(input_ids)
        output_ids, finish_reason, timing, request_cache = self.decoder.generate(
            input_ids,
            eos_token_id,
            sampling,
            max_total_tokens=self.cache.max_tokens,
            prefix_hit=prefix_hit,
        )
        self.prefix_cache.store_completed_blocks(input_ids, request_cache)
        return GenerationResult(
            output_ids=output_ids,
            finish_reason=finish_reason,
            prefill_seconds=timing["prefill_seconds"],
            inter_token_seconds=timing["inter_token_seconds"],
        )
