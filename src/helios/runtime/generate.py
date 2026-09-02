import time
from dataclasses import dataclass

from helios.runtime.check import CacheCapacity
from helios.runtime.prefix_cache import (
    PrefixCache,
    PromptBlockView,
    describe_prompt_blocks,
)
from helios.runtime.qwen3.decode import Decoder
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling

PREFIX_CACHE_BLOCK_SIZE = 16


@dataclass(frozen=True)
class PrefixTrace:
    block_size: int
    prompt_blocks: tuple[PromptBlockView, ...]
    hit_tokens: int
    restored_tokens: int
    stored_blocks: int

    def as_dict(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "prompt_blocks": [block.as_dict() for block in self.prompt_blocks],
            "hit_tokens": self.hit_tokens,
            "restored_tokens": self.restored_tokens,
            "stored_blocks": self.stored_blocks,
        }


@dataclass(frozen=True)
class GenerationResult:
    output_ids: list[int]
    finish_reason: str
    prefill_seconds: float
    inter_token_seconds: list[float]
    restore_seconds: float
    prefix_lookup_seconds: float
    store_seconds: float
    queue_seconds: float
    prefix: PrefixTrace


class Generator:
    def __init__(
        self,
        model: Qwen3Model,
        cache: CacheCapacity,
        torch_compile: bool = False,
        prefix_cache_ttl_seconds: float = 300.0,
    ) -> None:
        self.decoder = Decoder(model, torch_compile=torch_compile)
        self.cache = cache
        self.prefix_cache = PrefixCache(
            block_size=PREFIX_CACHE_BLOCK_SIZE,
            ttl_seconds=prefix_cache_ttl_seconds,
        )

    def run(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
    ) -> GenerationResult:
        lookup_started = time.perf_counter()
        prefix_hit = self.prefix_cache.longest_prefix(input_ids)
        prefix_lookup_seconds = time.perf_counter() - lookup_started
        decoded = self.decoder.generate(
            input_ids,
            eos_token_id,
            sampling,
            max_total_tokens=self.cache.max_tokens,
            prefix_hit=prefix_hit,
        )
        store_started = time.perf_counter()
        stored_blocks = self.prefix_cache.store_completed_blocks(
            input_ids, decoded.cache
        )
        store_seconds = time.perf_counter() - store_started
        return GenerationResult(
            output_ids=decoded.output_ids,
            finish_reason=decoded.finish_reason,
            prefill_seconds=decoded.prefill_seconds,
            inter_token_seconds=decoded.inter_token_seconds,
            restore_seconds=decoded.restore_seconds,
            prefix_lookup_seconds=prefix_lookup_seconds,
            store_seconds=store_seconds,
            queue_seconds=0.0,
            prefix=PrefixTrace(
                block_size=self.prefix_cache.block_size,
                prompt_blocks=describe_prompt_blocks(
                    input_ids, self.prefix_cache.block_size, prefix_hit
                ),
                hit_tokens=0 if prefix_hit is None else prefix_hit.length,
                restored_tokens=decoded.restored_tokens,
                stored_blocks=stored_blocks,
            ),
        )
