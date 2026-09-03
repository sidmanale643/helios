import logging
import time
from dataclasses import dataclass

import torch

from helios.runtime.prefix_cache import PrefixCacheHit
from helios.runtime.qwen3.cache import KVCache
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling

logger = logging.getLogger("uvicorn.error")
PROGRESS_INTERVAL_TOKENS = 32


@dataclass
class DecodeResult:
    output_ids: list[int]
    finish_reason: str
    prefill_seconds: float
    inter_token_seconds: list[float]
    restore_seconds: float
    restored_tokens: int
    cache: KVCache


class Decoder:
    def __init__(self, model: Qwen3Model, *, torch_compile: bool = False) -> None:
        self.model = model
        self._forward = (
            torch.compile(
                model,
                dynamic=True,
                fullgraph=True,
                mode="default",
            )
            if torch_compile
            else model
        )

    def generate(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        *,
        max_total_tokens: int,
        prefix_hit: PrefixCacheHit | None = None,
        request_id: str = "internal",
    ) -> DecodeResult:
        capacity = len(input_ids) + sampling.max_new_tokens
        if capacity > max_total_tokens:
            raise ValueError(
                f"Request needs {capacity:,} KV-cache tokens, but the profiled "
                f"limit is {max_total_tokens:,}."
            )
        # Cache capacity includes the prompt and the maximum possible continuation.
        cache = KVCache(self.model.config, capacity, device=self.device)
        cached_blocks = prefix_hit.blocks if prefix_hit is not None else ()
        if prefix_hit is not None:
            if any(
                len(block.tokens) != block.snapshot.length
                for block in cached_blocks
            ):
                raise ValueError("Prefix-cache token and KV block lengths do not match.")
            cached_tokens = tuple(
                token for block in cached_blocks for token in block.tokens
            )
            if len(cached_tokens) > len(input_ids):
                raise ValueError("Prefix-cache hit is longer than the request prompt.")
            if tuple(input_ids[: len(cached_tokens)]) != cached_tokens:
                raise ValueError("Prefix-cache hit does not match the request tokens.")
            if len(cached_tokens) == len(input_ids):
                cached_blocks = cached_blocks[:-1]

        restore_started = time.perf_counter()
        cache.restore_blocks(tuple(block.snapshot for block in cached_blocks))
        restore_seconds = time.perf_counter() - restore_started
        restored_tokens = cache.length
        token_tensor = torch.tensor(
            input_ids[cache.length :], device=self.device
        ).unsqueeze(0)
        generated: list[int] = []
        inter_token_seconds: list[float] = []
        finish_reason = "length"
        prefill_seconds = 0.0
        self.model.eval()
        with torch.inference_mode():
            self._synchronize()
            started = time.perf_counter()
            forward_logits = self._forward(token_tensor, cache=cache)
            self._validate_forward_shapes(token_tensor, forward_logits)
            logits = forward_logits[:, -1, :]
            for index in range(sampling.max_new_tokens):
                next_token = self._sample(logits, sampling)
                self._synchronize()
                elapsed = time.perf_counter() - started
                if index == 0:
                    prefill_seconds = elapsed
                token_id = next_token.item()
                if token_id == eos_token_id:
                    finish_reason = "eos"
                    break
                if index > 0:
                    inter_token_seconds.append(elapsed)
                generated.append(token_id)
                generated_tokens = len(generated)
                if (
                    generated_tokens == 1
                    or generated_tokens % PROGRESS_INTERVAL_TOKENS == 0
                ):
                    logger.info(
                        "generation_progress request_id=%s output_tokens=%d "
                        "max_new_tokens=%d last_token_ms=%.1f",
                        request_id,
                        generated_tokens,
                        sampling.max_new_tokens,
                        elapsed * 1_000,
                    )
                if index + 1 < sampling.max_new_tokens:
                    self._synchronize()
                    started = time.perf_counter()
                    forward_logits = self._forward(next_token, cache=cache)
                    self._validate_forward_shapes(next_token, forward_logits)
                    logits = forward_logits[:, -1, :]
        return DecodeResult(
            output_ids=generated,
            finish_reason=finish_reason,
            prefill_seconds=prefill_seconds,
            inter_token_seconds=inter_token_seconds,
            restore_seconds=restore_seconds,
            restored_tokens=restored_tokens,
            cache=cache,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _validate_forward_shapes(
        self, input_ids: torch.Tensor, logits: torch.Tensor
    ) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise RuntimeError(
                "Decoder input must have shape [1, tokens] with at least one token."
            )
        if input_ids.dtype != torch.long:
            raise RuntimeError("Decoder input token IDs must use torch.long.")
        expected_logits = (1, 1, self.model.config.vocab_size)
        if logits.shape != expected_logits:
            raise RuntimeError(
                "Decoder logits must have shape "
                f"{expected_logits}; received {tuple(logits.shape)}."
            )

    @staticmethod
    def _sample(logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        if sampling.temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probabilities = torch.softmax(logits / sampling.temperature, dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, descending=True
        )
        remove = torch.cumsum(sorted_probabilities, dim=-1) > sampling.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probabilities[remove] = 0
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probabilities, num_samples=1)
        return sorted_indices.gather(-1, sampled)
