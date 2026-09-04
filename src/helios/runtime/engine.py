import logging
import time
from dataclasses import replace
from threading import Lock

from helios.config import HeliosConfig
from helios.runtime.check import MemoryChecker
from helios.runtime.generate import GenerationResult, Generator
from helios.runtime.load import Loader
from helios.runtime.qwen3.decode import BatchDecodeResult
from helios.runtime.types import Sampling

logger = logging.getLogger("uvicorn.error")


class Engine:
    def __init__(self, config: HeliosConfig, loader: Loader | None = None) -> None:
        loaded = (loader or Loader()).load(config)
        self.model_id = config.model_id
        self.model_revision = loaded.model_revision
        self.torch_compile = config.torch_compile
        self._memory_checker = MemoryChecker(config)
        self.report = loaded.report
        self.generator = Generator(
            loaded.model,
            loaded.cache,
            torch_compile=config.torch_compile,
            prefix_cache_ttl_seconds=config.prefix_cache_ttl_seconds,
        )
        self._generation_lock = Lock()

    def update_cache_capacity(
        self,
        *,
        warmup_peak_bytes: int,
        warmup_kv_bytes: int,
    ) -> None:
        with self._generation_lock:
            cache = self._memory_checker.cache(
                self.generator.decoder.model.config,
                warmup_peak_bytes=warmup_peak_bytes,
                warmup_kv_bytes=warmup_kv_bytes,
            )
            self.generator.update_cache_capacity(cache)
            self.report = replace(self.report, cache=cache)

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
                "max_memory_bytes": cache.max_memory_bytes,
                "blocks": [block.as_dict() for block in blocks],
            }

    def run(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        request_id: str | None = None,
    ) -> GenerationResult:
        request_id = request_id or "internal"
        vocabulary_size = self.generator.decoder.model.config.vocab_size
        if eos_token_id >= vocabulary_size or any(
            token_id >= vocabulary_size for token_id in input_ids
        ):
            raise ValueError(
                f"Token IDs must be smaller than the model vocabulary size ({vocabulary_size:,})."
            )
        if self._generation_lock.locked():
            logger.info("request_queued request_id=%s", request_id)
        wait_started = time.perf_counter()
        with self._generation_lock:
            queue_seconds = time.perf_counter() - wait_started
            logger.info(
                "request_running request_id=%s prompt_tokens=%d max_new_tokens=%d queue_ms=%.1f",
                request_id,
                len(input_ids),
                sampling.max_new_tokens,
                queue_seconds * 1_000,
            )
            result = self.generator.run(
                input_ids,
                eos_token_id,
                sampling,
                request_id=request_id,
            )

            result = replace(result, queue_seconds=queue_seconds)
            generation_seconds = result.prefill_seconds + sum(
                result.inter_token_seconds
            )
            tokens_per_second = (
                len(result.output_ids) / generation_seconds
                if generation_seconds > 0
                else 0.0
            )
            logger.info(
                "request_completed request_id=%s finish_reason=%s output_tokens=%d "
                "cache_hit=%s cached_tokens=%d model_ttft_ms=%.1f generation_tok_s=%.2f "
                "total_ms=%.1f",
                request_id,
                result.finish_reason,
                len(result.output_ids),
                result.prefix.restored_tokens > 0,
                result.prefix.restored_tokens,
                (
                    result.prefix_lookup_seconds
                    + result.restore_seconds
                    + result.prefill_seconds
                )
                * 1_000,
                tokens_per_second,
                (time.perf_counter() - wait_started) * 1_000,
            )
            return result

    def run_batch(
        self,
        input_ids: list[list[int]],
        eos_token_id: int,
        sampling: Sampling,
    ) -> BatchDecodeResult:
        vocabulary_size = self.generator.decoder.model.config.vocab_size
        if eos_token_id >= vocabulary_size or any(
            token_id >= vocabulary_size for prompt in input_ids for token_id in prompt
        ):
            raise ValueError(
                f"Token IDs must be smaller than the model vocabulary size ({vocabulary_size:,})."
            )
        with self._generation_lock:
            return self.generator.run_batch(input_ids, eos_token_id, sampling)
