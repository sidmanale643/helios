import asyncio
import time
from dataclasses import dataclass, replace

import torch

from helios.runtime.engine import Engine
from helios.runtime.generate import GenerationResult
from helios.runtime.types import Sampling
from helios.runtime.warmup import (
    WARMUP_OUTPUT_TOKENS,
    WARMUP_PROMPT,
)
from helios.runtime.worker import Tokenizer


@dataclass(frozen=True)
class ChatGeneration:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    tokenize_seconds: float = 0.0
    queue_seconds: float = 0.0
    prefix_lookup_seconds: float = 0.0
    restore_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    store_seconds: float = 0.0
    first_token_seconds: float | None = None
    elapsed_seconds: float | None = None
    decode_compute_seconds: float = 0.0
    inter_token_seconds: tuple[float, ...] = ()

    @property
    def time_to_first_token_seconds(self) -> float:
        if self.first_token_seconds is not None:
            return self.first_token_seconds
        return (
            self.tokenize_seconds
            + self.queue_seconds
            + self.prefix_lookup_seconds
            + self.restore_seconds
            + self.prefill_seconds
        )

    @property
    def total_seconds(self) -> float:
        if self.elapsed_seconds is not None:
            return self.elapsed_seconds
        return (
            self.time_to_first_token_seconds + self.decode_seconds + self.store_seconds
        )

    @property
    def generation_tokens_per_second(self) -> float | None:
        generation_seconds = self.prefill_seconds + self.decode_compute_seconds
        if generation_seconds == 0:
            return None
        return self.completion_tokens / generation_seconds

    @property
    def prefill_tokens_per_second(self) -> float | None:
        if self.prefill_seconds == 0:
            return None
        return (self.prompt_tokens - self.cached_tokens) / self.prefill_seconds

    @property
    def decode_tokens_per_second(self) -> float | None:
        if self.decode_seconds == 0:
            return None
        return max(0, self.completion_tokens - 1) / self.decode_seconds

    @property
    def decode_compute_tokens_per_second(self) -> float | None:
        if self.decode_compute_seconds == 0:
            return None
        return max(0, self.completion_tokens - 1) / self.decode_compute_seconds

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens


class TextGenerator:
    def __init__(self, tokenizer: Tokenizer, engine: Engine) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        self._warmed = False
        self._decode_warmup_batch_sizes: tuple[int, ...] = ()
        if (
            tokenizer.model_id != engine.model_id
            or tokenizer.model_revision != engine.model_revision
        ):
            raise RuntimeError(
                "Tokenizer and model snapshots do not match: "
                f"tokenizer={tokenizer.model_id}@{tokenizer.model_revision}, "
                f"model={engine.model_id}@{engine.model_revision}."
            )

    def run_chat(
        self,
        messages: list[tuple[str, str]],
        sampling: Sampling,
        request_id: str | None = None,
    ) -> ChatGeneration:
        started = time.perf_counter()
        input_ids, tokenize_seconds = self._tokenize_chat(messages)
        result = self._generate(input_ids, sampling, request_id=request_id)
        chat = self._chat_generation(input_ids, tokenize_seconds, result)
        return replace(
            chat,
            elapsed_seconds=time.perf_counter() - started,
            first_token_seconds=(
                result.first_token_at - started
                if result.first_token_at is not None
                else chat.first_token_seconds
            ),
        )

    async def run_chat_async(
        self,
        messages: list[tuple[str, str]],
        sampling: Sampling,
        request_id: str | None = None,
    ) -> ChatGeneration:
        started = time.perf_counter()
        input_ids, tokenize_seconds = await asyncio.to_thread(
            self._tokenize_chat, messages
        )
        future = self.engine.enqueue(
            input_ids,
            self.tokenizer.eos_token_id,
            sampling,
            request_id=request_id,
        )
        result = await asyncio.wrap_future(future)
        chat = await asyncio.to_thread(
            self._chat_generation, input_ids, tokenize_seconds, result
        )
        return replace(
            chat,
            elapsed_seconds=time.perf_counter() - started,
            first_token_seconds=(
                result.first_token_at - started
                if result.first_token_at is not None
                else chat.first_token_seconds
            ),
        )

    def _tokenize_chat(
        self, messages: list[tuple[str, str]]
    ) -> tuple[list[int], float]:
        tokenize_started = time.perf_counter()
        input_ids = self.tokenizer.tokenize_chat(messages)
        tokenize_seconds = time.perf_counter() - tokenize_started
        return input_ids, tokenize_seconds

    def _chat_generation(
        self,
        input_ids: list[int],
        tokenize_seconds: float,
        result: GenerationResult,
    ) -> ChatGeneration:
        return ChatGeneration(
            text=self.tokenizer.detokenize(result.output_ids),
            finish_reason=result.finish_reason,
            prompt_tokens=len(input_ids),
            completion_tokens=len(result.output_ids),
            cached_tokens=result.prefix.restored_tokens,
            tokenize_seconds=tokenize_seconds,
            queue_seconds=result.queue_seconds,
            prefix_lookup_seconds=result.prefix_lookup_seconds,
            restore_seconds=result.restore_seconds,
            prefill_seconds=result.prefill_seconds,
            decode_seconds=sum(
                result.token_intervals
                if result.token_intervals is not None
                else result.inter_token_seconds
            ),
            decode_compute_seconds=sum(result.inter_token_seconds),
            inter_token_seconds=result.token_intervals or (),
            first_token_seconds=(
                tokenize_seconds + result.first_token_seconds
                if result.first_token_seconds is not None
                else None
            ),
            elapsed_seconds=(
                tokenize_seconds + result.elapsed_seconds
                if result.elapsed_seconds is not None
                else None
            ),
            store_seconds=result.store_seconds,
        )

    def warm_up(self) -> None:
        if self._warmed:
            return

        prompt = WARMUP_PROMPT
        input_ids = self.tokenizer.tokenize_chat([("user", prompt)])
        while len(input_ids) <= self.engine.generator.prefix_cache.block_size:
            prompt += "\n\n" + WARMUP_PROMPT
            input_ids = self.tokenizer.tokenize_chat([("user", prompt)])
        extended_ids = self.tokenizer.tokenize_chat(
            [
                ("user", prompt),
                (
                    "assistant",
                    "Start with transaction integrity and cache invalidation.",
                ),
                ("user", "Explain the rollout steps and how to measure their success."),
            ]
        )

        def run(token_ids: list[int], request_id: str) -> GenerationResult:
            result = self.engine.run_warmup(
                token_ids,
                self.tokenizer.eos_token_id,
                Sampling(
                    temperature=0,
                    top_p=1,
                    max_new_tokens=WARMUP_OUTPUT_TOKENS,
                ),
                request_id,
            )
            if not 0 <= len(result.output_ids) <= WARMUP_OUTPUT_TOKENS:
                raise RuntimeError(
                    "Warmup must generate between 0 and "
                    f"{WARMUP_OUTPUT_TOKENS} tokens; generated "
                    f"{len(result.output_ids)}."
                )
            return result

        cache = self.engine.generator.prefix_cache
        device = self.engine.generator.decoder.device
        cache.clear()
        try:
            run(input_ids, "startup-warmup-cold")
            result = run(extended_ids, "startup-warmup-prefix")
            if not 0 < result.prefix.restored_tokens < len(extended_ids) - 1:
                raise RuntimeError(
                    "Warmup must restore a prefix and prefill multiple new tokens."
                )
            del result
            cache.clear()
            decode_activation_bytes, decode_batch_sizes = self.engine.warm_decode(
                input_ids
            )
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            run(input_ids, "startup-profile-cold")
            torch.cuda.synchronize(device)
            warmup_peak_bytes = torch.cuda.max_memory_reserved(device) - baseline
        finally:
            cache.clear()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        self.engine.update_cache_capacity(
            warmup_peak_bytes=max(warmup_peak_bytes, decode_activation_bytes),
            warmup_kv_bytes=0,
        )
        self._decode_warmup_batch_sizes = decode_batch_sizes
        self._warmed = True

    @property
    def model_id(self) -> str:
        return self.tokenizer.model_id

    def _generate(
        self,
        input_ids: list[int],
        sampling: Sampling,
        *,
        request_id: str | None = None,
    ) -> GenerationResult:
        return self.engine.run(
            input_ids,
            self.tokenizer.eos_token_id,
            sampling,
            request_id=request_id,
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model": self.engine.model_id,
            "model_revision": self.engine.model_revision,
            "warmup_batch_sizes": list(self._decode_warmup_batch_sizes),
            "memory": self.engine.report.as_dict(),
            "scheduler": self.engine.scheduler_snapshot(),
        }

    def close(self) -> None:
        self.engine.close()
