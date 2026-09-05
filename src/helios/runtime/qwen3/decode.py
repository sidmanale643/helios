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


@dataclass
class BatchDecodeResult:
    output_ids: list[list[int]]
    finish_reasons: list[str]
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0


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
                len(block.tokens) != block.snapshot.length for block in cached_blocks
            ):
                raise ValueError(
                    "Prefix-cache token and KV block lengths do not match."
                )
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

    def generate_batch(
        self,
        input_ids: list[list[int]],
        eos_token_id: int,
        samplings: list[Sampling],
        *,
        max_total_tokens: int,
    ) -> BatchDecodeResult:
        if not input_ids or any(not tokens for tokens in input_ids):
            raise ValueError("A batch must contain at least one non-empty prompt.")
        if len(samplings) != len(input_ids):
            raise ValueError("Every prompt must have sampling settings.")
        sampling = samplings[0]
        if any(
            item.temperature != sampling.temperature or item.top_p != sampling.top_p
            for item in samplings[1:]
        ):
            raise ValueError(
                "Every request in a batch must use the same temperature and top_p."
            )

        batch_size = len(input_ids)
        prompt_lengths = torch.tensor(
            [len(tokens) for tokens in input_ids], device=self.device
        )
        longest_prompt = int(prompt_lengths.max().item())
        max_new_tokens = max(item.max_new_tokens for item in samplings)
        capacity = longest_prompt + max_new_tokens
        if capacity > self.model.config.context_length:
            raise ValueError(
                f"A padded batch needs {capacity:,} cache positions, but the model "
                f"supports {self.model.config.context_length:,}."
            )
        if batch_size * capacity > max_total_tokens:
            raise ValueError(
                f"Batch needs {batch_size * capacity:,} KV-cache tokens, but the "
                f"profiled limit is {max_total_tokens:,}."
            )

        started = time.perf_counter()
        logger.info(
            "batch_prefill_started batch_size=%d padded_prompt_tokens=%d "
            "max_new_tokens=%d kv_cache_tokens=%d",
            batch_size,
            longest_prompt,
            max_new_tokens,
            batch_size * capacity,
        )
        token_tensor = torch.full(
            (batch_size, longest_prompt),
            eos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        key_mask = torch.zeros(
            (batch_size, longest_prompt), dtype=torch.bool, device=self.device
        )
        position_ids = torch.zeros_like(token_tensor)
        for row, tokens in enumerate(input_ids):
            start = longest_prompt - len(tokens)
            token_tensor[row, start:] = torch.tensor(tokens, device=self.device)
            key_mask[row, start:] = True
            position_ids[row, start:] = torch.arange(len(tokens), device=self.device)

        cache = KVCache(
            self.model.config, capacity, device=self.device, batch_size=batch_size
        )
        generated = [[] for _ in input_ids]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        eos_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        token_limits = torch.tensor(
            [item.max_new_tokens for item in samplings], device=self.device
        )

        self.model.eval()
        with torch.inference_mode():
            self._synchronize()
            prefill_started = time.perf_counter()
            logits = self._forward(
                token_tensor,
                cache=cache,
                attention_mask=key_mask,
                position_ids=position_ids,
            )[:, -1, :]
            decode_started = 0.0
            for index in range(max_new_tokens):
                next_token = self._sample(logits, sampling)
                if index == 0:
                    self._synchronize()
                    prefill_seconds = time.perf_counter() - prefill_started
                    decode_started = time.perf_counter()
                token_ids = next_token.squeeze(1).tolist()
                if index == 0:
                    logger.info(
                        "batch_prefill_completed elapsed_ms=%.1f",
                        (time.perf_counter() - started) * 1_000,
                    )
                active = ~finished
                active_rows = active.tolist()
                for row, token_id in enumerate(token_ids):
                    if active_rows[row] and token_id != eos_token_id:
                        generated[row].append(token_id)
                eos_finished |= active & next_token.squeeze(1).eq(eos_token_id)
                generated_counts = torch.tensor(
                    [len(tokens) for tokens in generated], device=self.device
                )
                finished = eos_finished | generated_counts.ge(token_limits)
                if index == 0 or (index + 1) % PROGRESS_INTERVAL_TOKENS == 0:
                    logger.info(
                        "batch_generation_progress step=%d max_new_tokens=%d "
                        "output_tokens=%d elapsed_seconds=%.1f",
                        index + 1,
                        max_new_tokens,
                        sum(len(tokens) for tokens in generated),
                        time.perf_counter() - started,
                    )
                if finished.all():
                    break

                step_mask = ~finished
                key_mask = torch.cat((key_mask, step_mask[:, None]), dim=1)
                step_positions = prompt_lengths + generated_counts - 1
                logits = self._forward(
                    next_token,
                    cache=cache,
                    attention_mask=key_mask,
                    position_ids=step_positions[:, None].clamp_min(0),
                )[:, -1, :]

            self._synchronize()
            decode_seconds = time.perf_counter() - decode_started if index > 0 else 0.0

        return BatchDecodeResult(
            output_ids=generated,
            finish_reasons=[
                "eos" if stopped_on_eos else "length"
                for stopped_on_eos in eos_finished.tolist()
            ],
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
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
