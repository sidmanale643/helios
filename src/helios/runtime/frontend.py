import time
from dataclasses import dataclass

import torch

from helios.runtime.engine import Engine
from helios.runtime.generate import GenerationResult
from helios.runtime.types import GenerateBatch, GenerateRequest, Sampling
from helios.runtime.warmup import (
    COMPILE_WARMUP_OUTPUT_TOKENS,
    COMPILE_WARMUP_PROMPT,
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


class TextGenerator:

    def __init__(self, tokenizer: Tokenizer, engine: Engine) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        self._warmed = False
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

    def run_batch(self, batch: GenerateBatch) -> list[str]:
        sampling = batch.requests[0].sampling
        if any(request.sampling != sampling for request in batch.requests[1:]):
            raise ValueError(
                "Every request in a batch must use the same sampling settings."
            )
        input_ids = [
            self.tokenizer.tokenize(request.text) for request in batch.requests
        ]
        result = self.engine.run_batch(
            input_ids, self.tokenizer.eos_token_id, sampling
        )
        return [self.tokenizer.detokenize(tokens) for tokens in result.output_ids]

    def run_chat(
        self,
        messages: list[tuple[str, str]],
        sampling: Sampling,
        request_id: str | None = None,
    ) -> ChatGeneration:
        tokenize_started = time.perf_counter()
        input_ids = self.tokenizer.tokenize_chat(messages)
        tokenize_seconds = time.perf_counter() - tokenize_started
        result = self._generate(input_ids, sampling, request_id=request_id)
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
            decode_seconds=sum(result.inter_token_seconds),
            store_seconds=result.store_seconds,
        )

    def warm_up(self) -> None:
        if self._warmed:
            return

        input_ids = self.tokenizer.tokenize_chat(
            [("user", COMPILE_WARMUP_PROMPT)]
        )
        extended_ids = self.tokenizer.tokenize_chat(
            [
                ("user", COMPILE_WARMUP_PROMPT),
                ("assistant", "Start with transaction integrity and cache invalidation."),
                ("user", "Explain the rollout steps and how to measure their success."),
            ]
        )

        def run(token_ids: list[int], request_id: str) -> GenerationResult:
            result = self._generate(
                token_ids,
                Sampling(
                    temperature=0,
                    top_p=1,
                    max_new_tokens=COMPILE_WARMUP_OUTPUT_TOKENS,
                ),
                request_id=request_id,
            )
            if not 2 <= len(result.output_ids) <= COMPILE_WARMUP_OUTPUT_TOKENS:
                raise RuntimeError(
                    "Compile warmup must generate between 2 and "
                    f"{COMPILE_WARMUP_OUTPUT_TOKENS} tokens; generated "
                    f"{len(result.output_ids)}."
                )
            return result

        cache = self.engine.generator.prefix_cache
        device = self.engine.generator.decoder.device
        cache.clear()
        try:
            run(input_ids, "startup-compile-cold")
            result = run(extended_ids, "startup-compile-prefix")
            if not 0 < result.prefix.restored_tokens < len(extended_ids) - 1:
                raise RuntimeError(
                    "Compile warmup must restore a prefix and prefill multiple new tokens."
                )
            del result
            cache.clear()
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
        warmup_kv_bytes = (
            len(input_ids) + COMPILE_WARMUP_OUTPUT_TOKENS
        ) * self.engine.generator.cache.bytes_per_token
        self.engine.update_cache_capacity(
            warmup_peak_bytes=warmup_peak_bytes,
            warmup_kv_bytes=warmup_kv_bytes,
        )
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
            "torch_compile": {
                "enabled": self.engine.torch_compile,
                "warmed": self._warmed and self.engine.torch_compile,
            },
            "memory": self.engine.report.as_dict(),
        }
