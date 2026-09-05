import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace
from threading import Lock
from typing import cast

from helios.config import HeliosConfig
from helios.runtime.check import MemoryChecker
from helios.runtime.generate import GenerationResult, Generator
from helios.runtime.load import Loader
from helios.runtime.qwen3.decode import BatchDecodeResult
from helios.runtime.scheduler import Job, Scheduler
from helios.runtime.types import Sampling

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class _Request:
    input_ids: list[int]
    eos_token_id: int
    sampling: Sampling
    request_id: str


@dataclass(frozen=True)
class _Batch:
    input_ids: list[list[int]]
    eos_token_id: int
    samplings: list[Sampling]
    request_id: str


_Payload = _Request | _Batch
_Result = GenerationResult | BatchDecodeResult


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
        self._scheduler: Scheduler[_Payload, _Result] = Scheduler(
            self._execute,
            self._can_add,
            max_batch_size=config.max_batch_size,
            max_queue_size=config.max_queue_size,
            batch_wait_seconds=config.batch_wait_ms / 1_000,
        )

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

    def scheduler_snapshot(self) -> dict[str, object]:
        return self._scheduler.snapshot()

    def close(self) -> None:
        self._scheduler.close()

    def run(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        request_id: str | None = None,
    ) -> GenerationResult:
        return self.enqueue(input_ids, eos_token_id, sampling, request_id).result()

    def enqueue(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        request_id: str | None = None,
    ) -> Future[GenerationResult]:
        request_id = request_id or "internal"
        self._validate_request(input_ids, eos_token_id, sampling)
        logger.info("request_waiting request_id=%s", request_id)
        payload = _Request(input_ids, eos_token_id, sampling, request_id)
        return cast(
            Future[GenerationResult],
            self._scheduler.enqueue(Job(payload=payload, request_ids=(request_id,))),
        )

    def _run_one(self, request: _Request, queue_seconds: float) -> GenerationResult:
        with self._generation_lock:
            started = time.perf_counter()
            logger.info(
                "request_running request_id=%s prompt_tokens=%d max_new_tokens=%d queue_ms=%.1f",
                request.request_id,
                len(request.input_ids),
                request.sampling.max_new_tokens,
                queue_seconds * 1_000,
            )
            result = self.generator.run(
                request.input_ids,
                request.eos_token_id,
                request.sampling,
                request_id=request.request_id,
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
                request.request_id,
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
                (queue_seconds + time.perf_counter() - started) * 1_000,
            )
            return result

    def run_batch(
        self,
        input_ids: list[list[int]],
        eos_token_id: int,
        samplings: list[Sampling],
        request_id: str = "internal-batch",
    ) -> BatchDecodeResult:
        if len(input_ids) != len(samplings):
            raise ValueError("Every prompt must have sampling settings.")
        for prompt, sampling in zip(input_ids, samplings, strict=True):
            self._validate_request(prompt, eos_token_id, sampling)
        logger.info(
            "batch_waiting batch_id=%s batch_size=%d", request_id, len(input_ids)
        )
        payload = _Batch(input_ids, eos_token_id, samplings, request_id)
        result = self._scheduler.submit(
            Job(payload=payload, request_ids=(request_id,), batchable=False)
        )
        if not isinstance(result, BatchDecodeResult):
            raise TypeError(
                "The scheduler returned a single-request result for a batch."
            )
        return result

    def _run_explicit_batch(
        self, batch: _Batch, queue_seconds: float
    ) -> BatchDecodeResult:
        with self._generation_lock:
            logger.info(
                "batch_running batch_id=%s batch_size=%d prompt_tokens=%d queue_ms=%.1f",
                batch.request_id,
                len(batch.input_ids),
                sum(len(prompt) for prompt in batch.input_ids),
                queue_seconds * 1_000,
            )
            return self.generator.run_batch(
                batch.input_ids, batch.eos_token_id, batch.samplings
            )

    def _execute(self, jobs: tuple[Job[_Payload, _Result], ...]) -> tuple[_Result, ...]:
        queue_seconds = [time.perf_counter() - job.enqueued_at for job in jobs]
        first = jobs[0].payload
        if isinstance(first, _Batch):
            logger.info("batch_active batch_id=%s", first.request_id)
            return (self._run_explicit_batch(first, queue_seconds[0]),)
        requests = [job.payload for job in jobs]
        if not all(isinstance(request, _Request) for request in requests):
            raise RuntimeError("A scheduled batch cannot mix request types.")
        typed_requests = [
            request for request in requests if isinstance(request, _Request)
        ]
        for request, seconds in zip(typed_requests, queue_seconds, strict=True):
            logger.info(
                "request_active request_id=%s active_batch_size=%d queue_ms=%.1f",
                request.request_id,
                len(typed_requests),
                seconds * 1_000,
            )
        if len(typed_requests) == 1:
            return (self._run_one(typed_requests[0], queue_seconds[0]),)
        with self._generation_lock:
            logger.info(
                "batch_running batch_size=%d request_ids=%s queue_ms=%s",
                len(typed_requests),
                ",".join(request.request_id for request in typed_requests),
                ",".join(f"{seconds * 1_000:.1f}" for seconds in queue_seconds),
            )
            results = self.generator.run_scheduled_batch(
                [request.input_ids for request in typed_requests],
                typed_requests[0].eos_token_id,
                [request.sampling for request in typed_requests],
            )
        return tuple(
            self._finish_scheduled_request(result, seconds, request.request_id)
            for result, seconds, request in zip(
                results, queue_seconds, typed_requests, strict=True
            )
        )

    def _finish_scheduled_request(
        self, result: GenerationResult, queue_seconds: float, request_id: str
    ) -> GenerationResult:
        result = replace(result, queue_seconds=queue_seconds)
        generation_seconds = result.prefill_seconds + sum(result.inter_token_seconds)
        tokens_per_second = (
            len(result.output_ids) / generation_seconds
            if generation_seconds > 0
            else 0.0
        )
        logger.info(
            "request_completed request_id=%s finish_reason=%s output_tokens=%d "
            "cache_hit=false cached_tokens=0 model_ttft_ms=%.1f "
            "generation_tok_s=%.2f total_ms=%.1f",
            request_id,
            result.finish_reason,
            len(result.output_ids),
            result.prefill_seconds * 1_000,
            tokens_per_second,
            (queue_seconds + generation_seconds) * 1_000,
        )
        return result

    def _validate_request(
        self, input_ids: list[int], eos_token_id: int, sampling: Sampling
    ) -> None:
        vocabulary_size = self.generator.decoder.model.config.vocab_size
        if not input_ids:
            raise ValueError("A request prompt must contain at least one token.")
        token_ids = [eos_token_id, *input_ids]
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < vocabulary_size
            for token_id in token_ids
        ):
            raise ValueError(
                f"Token IDs must be between 0 and {vocabulary_size - 1:,}."
            )
        capacity = len(input_ids) + sampling.max_new_tokens
        context_length = self.generator.decoder.model.config.context_length
        if capacity > context_length:
            raise ValueError(
                f"Request needs {capacity:,} cache positions, but the model supports "
                f"{context_length:,}."
            )
        max_tokens = (
            self.generator.cache.kv_budget_bytes // self.generator.cache.bytes_per_token
        )
        if capacity > max_tokens:
            raise ValueError(
                f"Request needs {capacity:,} KV-cache tokens, but the profiled limit "
                f"is {max_tokens:,}."
            )

    def _can_add(
        self,
        active: tuple[Job[_Payload, _Result], ...],
        candidate: Job[_Payload, _Result],
    ) -> bool:
        payloads = [job.payload for job in active]
        if not isinstance(candidate.payload, _Request) or not all(
            isinstance(payload, _Request) for payload in payloads
        ):
            return False
        requests = [
            payload for payload in payloads if isinstance(payload, _Request)
        ] + [candidate.payload]
        first = requests[0]
        if any(
            request.eos_token_id != first.eos_token_id
            or request.sampling.temperature != first.sampling.temperature
            or request.sampling.top_p != first.sampling.top_p
            for request in requests[1:]
        ):
            return False
        longest_prompt = max(len(request.input_ids) for request in requests)
        max_new_tokens = max(request.sampling.max_new_tokens for request in requests)
        capacity = longest_prompt + max_new_tokens
        max_total_tokens = (
            self.generator.cache.kv_budget_bytes // self.generator.cache.bytes_per_token
        )
        return (
            capacity <= self.generator.decoder.model.config.context_length
            and len(requests) * capacity <= max_total_tokens
        )
