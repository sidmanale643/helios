import logging
import time
from dataclasses import dataclass

import torch

from helios.runtime.check import CacheCapacity
from helios.runtime.prefix_cache import (
    PrefixCache,
    PromptBlockView,
    describe_prompt_blocks,
)
from helios.runtime.qwen3.decode import BatchDecodeResult, Decoder
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling

PREFIX_CACHE_BLOCK_SIZE = 16
logger = logging.getLogger("uvicorn.error")


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
            max_memory_bytes=cache.kv_budget_bytes,
            ttl_seconds=prefix_cache_ttl_seconds,
        )

    def update_cache_capacity(self, cache: CacheCapacity) -> None:
        self.cache = cache
        self.prefix_cache.max_memory_bytes = cache.kv_budget_bytes
        if self.prefix_cache.reserve(0):
            torch.cuda.empty_cache()

    def run(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        *,
        request_id: str = "internal",
    ) -> GenerationResult:
        request_cache_bytes = (
            len(input_ids) + sampling.max_new_tokens
        ) * self.cache.bytes_per_token
        if request_cache_bytes > self.cache.kv_budget_bytes:
            requested_tokens = len(input_ids) + sampling.max_new_tokens
            raise ValueError(
                f"Request needs {requested_tokens:,} KV-cache tokens, but the profiled "
                f"limit is {self.cache.max_tokens:,}."
            )
        if self.prefix_cache.reserve(request_cache_bytes):
            torch.cuda.empty_cache()
        lookup_started = time.perf_counter()
        prefix_hit = self.prefix_cache.longest_prefix(input_ids)
        prefix_lookup_seconds = time.perf_counter() - lookup_started
        hit_tokens = 0 if prefix_hit is None else prefix_hit.length
        logger.info(
            "prefix_cache request_id=%s status=%s cached_tokens=%d prompt_tokens=%d lookup_ms=%.1f",
            request_id,
            "hit" if hit_tokens else "miss",
            hit_tokens,
            len(input_ids),
            prefix_lookup_seconds * 1_000,
        )
        decoded = self.decoder.generate(
            input_ids,
            eos_token_id,
            sampling,
            max_total_tokens=self.cache.max_tokens,
            prefix_hit=prefix_hit,
            request_id=request_id,
        )
        prompt_blocks = describe_prompt_blocks(
            input_ids, self.prefix_cache.block_size, prefix_hit
        )
        prefix_hit = None
        store_started = time.perf_counter()
        stored_blocks = self.prefix_cache.store_completed_blocks(
            input_ids,
            decoded.cache,
            reserved_memory_bytes=request_cache_bytes,
        )
        store_seconds = time.perf_counter() - store_started
        logger.info(
            "prefix_cache_store request_id=%s stored_blocks=%d occupied_blocks=%d store_ms=%.1f",
            request_id,
            stored_blocks,
            len(self.prefix_cache.blocks()),
            store_seconds * 1_000,
        )
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
                prompt_blocks=prompt_blocks,
                hit_tokens=hit_tokens,
                restored_tokens=decoded.restored_tokens,
                stored_blocks=stored_blocks,
            ),
        )

    def run_batch(
        self,
        input_ids: list[list[int]],
        eos_token_id: int,
        sampling: Sampling,
    ) -> BatchDecodeResult:
        longest_prompt = max((len(tokens) for tokens in input_ids), default=0)
        cache_tokens = len(input_ids) * (longest_prompt + sampling.max_new_tokens)
        request_cache_bytes = cache_tokens * self.cache.bytes_per_token
        if request_cache_bytes > self.cache.kv_budget_bytes:
            raise ValueError(
                f"Batch needs {cache_tokens:,} KV-cache tokens, but the profiled "
                f"limit is {self.cache.kv_budget_bytes // self.cache.bytes_per_token:,}."
            )
        if self.prefix_cache.reserve(request_cache_bytes):
            torch.cuda.empty_cache()
        return self.decoder.generate_batch(
            input_ids,
            eos_token_id,
            sampling,
            max_total_tokens=self.cache.kv_budget_bytes // self.cache.bytes_per_token,
        )
