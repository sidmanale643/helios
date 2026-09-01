import time
from dataclasses import dataclass

import torch

from helios.runtime.prefix_cache import PrefixCacheHit
from helios.runtime.qwen3.cache import KVCache
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling


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
    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    def generate(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        *,
        max_total_tokens: int,
        prefix_hit: PrefixCacheHit | None = None,
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
            logits = self.model(token_tensor, cache=cache)[:, -1, :]
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
                if index + 1 < sampling.max_new_tokens:
                    self._synchronize()
                    started = time.perf_counter()
                    logits = self.model(next_token, cache=cache)[:, -1, :]
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
