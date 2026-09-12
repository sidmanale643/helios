import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from helios.runtime.prefix_cache import PrefixCacheHit
from helios.runtime.qwen3.cache import BatchedKVCache, DenseDecodeBatch, KVCache
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.qwen3.paged_cache import (
    KVPagePool,
    PagedBatchCache,
    PagedKVCache,
)
from helios.runtime.types import Sampling

logger = logging.getLogger("uvicorn.error")
PROGRESS_INTERVAL_TOKENS = 32


@dataclass
class DecodeResult:
    first_token_at: float
    token_intervals: tuple[float, ...]
    output_ids: list[int]
    finish_reason: str
    prefill_seconds: float
    inter_token_seconds: list[float]
    restore_seconds: float
    restored_tokens: int
    cache: KVCache | PagedKVCache


@dataclass
class PrefillResult:
    cache: KVCache | PagedKVCache
    logits: torch.Tensor
    prefill_seconds: float
    restore_seconds: float
    restored_tokens: int


@dataclass
class PrefillState:
    cache: KVCache | PagedKVCache
    next_token_offset: int
    restore_seconds: float
    restored_tokens: int


@dataclass
class DecodedTokens:
    first_token_at: float
    token_intervals: tuple[float, ...]
    output_ids: list[int]
    finish_reason: str
    inter_token_seconds: list[float]


class Decoder:
    def __init__(self, model: Qwen3Model) -> None:
        self.model = model
        self.page_pool: KVPagePool | None = None
        self._paged_decode_batch: PagedBatchCache | None = None

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
        prefill = self.prefill(
            input_ids,
            sampling,
            max_total_tokens=max_total_tokens,
            prefix_hit=prefix_hit,
        )
        try:
            decoded = self.decode(
                prefill, eos_token_id, sampling, request_id=request_id
            )
        except Exception:
            self.release_cache(prefill.cache)
            raise
        return DecodeResult(
            first_token_at=decoded.first_token_at,
            token_intervals=decoded.token_intervals,
            output_ids=decoded.output_ids,
            finish_reason=decoded.finish_reason,
            prefill_seconds=prefill.prefill_seconds,
            inter_token_seconds=decoded.inter_token_seconds,
            restore_seconds=prefill.restore_seconds,
            restored_tokens=prefill.restored_tokens,
            cache=prefill.cache,
        )

    def prefill(
        self,
        input_ids: list[int],
        sampling: Sampling,
        *,
        max_total_tokens: int,
        prefix_hit: PrefixCacheHit | None = None,
    ) -> PrefillResult:
        state = self.begin_prefill(
            input_ids,
            sampling,
            max_total_tokens=max_total_tokens,
            prefix_hit=prefix_hit,
        )
        try:
            logits, prefill_seconds = self.prefill_chunk(
                state.cache, input_ids[state.next_token_offset :]
            )
        except Exception:
            self.release_cache(state.cache)
            raise
        return PrefillResult(
            cache=state.cache,
            logits=logits,
            prefill_seconds=prefill_seconds,
            restore_seconds=state.restore_seconds,
            restored_tokens=state.restored_tokens,
        )

    def begin_prefill(
        self,
        input_ids: list[int],
        sampling: Sampling,
        *,
        max_total_tokens: int,
        prefix_hit: PrefixCacheHit | None = None,
    ) -> PrefillState:
        capacity = len(input_ids) + sampling.max_new_tokens
        if capacity > max_total_tokens:
            raise ValueError(
                f"Request needs {capacity:,} KV-cache tokens, but the profiled "
                f"limit is {max_total_tokens:,}."
            )
        cache = (
            PagedKVCache(self.page_pool, capacity)
            if self.page_pool is not None
            else KVCache(self.model.config, capacity, device=self.device)
        )
        try:
            cached_blocks = prefix_hit.blocks if prefix_hit is not None else ()
            if prefix_hit is not None:
                if any(
                    len(block.tokens) != block.snapshot.length
                    for block in cached_blocks
                ):
                    raise ValueError(
                        "Prefix-cache token and KV block lengths do not match."
                    )
                cached_tokens = tuple(
                    token for block in cached_blocks for token in block.tokens
                )
                if len(cached_tokens) > len(input_ids):
                    raise ValueError(
                        "Prefix-cache hit is longer than the request prompt."
                    )
                if tuple(input_ids[: len(cached_tokens)]) != cached_tokens:
                    raise ValueError(
                        "Prefix-cache hit does not match the request tokens."
                    )
                if len(cached_tokens) == len(input_ids):
                    cached_blocks = cached_blocks[:-1]

            restore_started = time.perf_counter()
            cache.restore_blocks(tuple(block.snapshot for block in cached_blocks))
            restore_seconds = time.perf_counter() - restore_started
            return PrefillState(
                cache=cache,
                next_token_offset=cache.length,
                restore_seconds=restore_seconds,
                restored_tokens=cache.length,
            )
        except Exception:
            self.release_cache(cache)
            raise

    def prefill_chunk(
        self,
        cache: KVCache | PagedKVCache,
        input_ids_slice: Sequence[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if isinstance(input_ids_slice, torch.Tensor):
            token_tensor = input_ids_slice.to(device=self.device, dtype=torch.long)
            if token_tensor.ndim == 1:
                token_tensor = token_tensor.unsqueeze(0)
        else:
            token_tensor = torch.tensor(
                input_ids_slice, dtype=torch.long, device=self.device
            ).unsqueeze(0)
        if (
            token_tensor.ndim != 2
            or token_tensor.shape[0] != 1
            or token_tensor.shape[1] < 1
        ):
            raise ValueError(
                "Prefill chunks must contain at least one token with shape [1, tokens]."
            )
        self.model.eval()
        with torch.inference_mode():
            self._synchronize()
            started = time.perf_counter()
            forward_logits = self._forward_cache(token_tensor, cache)
            self._validate_forward_shapes(token_tensor, forward_logits)
            self._synchronize()
            elapsed = time.perf_counter() - started
        return forward_logits[:, -1, :], elapsed

    def decode_caches(
        self, caches: list[KVCache | PagedKVCache], token_ids: list[int]
    ) -> torch.Tensor:
        if not caches or len(caches) != len(token_ids):
            raise ValueError("Every request cache needs exactly one pending token.")
        self.model.eval()
        with torch.inference_mode():
            if isinstance(caches[0], PagedKVCache):
                if any(
                    not isinstance(cache, PagedKVCache)
                    or cache.pool is not caches[0].pool
                    for cache in caches
                ):
                    raise ValueError("Paged decode caches must share one pool.")
                if len({id(cache) for cache in caches}) != len(caches):
                    raise ValueError("Paged batches require distinct request caches.")
                batch = self._paged_decode_batch
                if batch is None or batch.pool is not caches[0].pool:
                    batch = self._paged_decode_batch = PagedBatchCache(caches)
                batch.caches = tuple(caches)
                batch.batch_size = len(caches)
                try:
                    tokens = batch.metadata(
                        "token_ids", token_ids, torch.long
                    ).unsqueeze(1)
                    batch.prepare(1)
                    slots = tuple(range(len(caches)))
                    return self.model(
                        tokens,
                        cache=batch,
                        position_ids=batch.slot_lengths(slots).unsqueeze(1),
                        cache_slots=slots,
                    )[:, -1, :]
                finally:
                    batch.caches = ()
            tokens = torch.tensor(
                token_ids, dtype=torch.long, device=self.device
            ).unsqueeze(1)
            return self._decode_dense(tokens, caches)[:, -1, :]

    def _decode_dense(
        self, tokens: torch.Tensor, caches: list[KVCache]
    ) -> torch.Tensor:
        BatchedKVCache(caches)
        lengths = [cache.length for cache in caches]
        if any(
            length >= cache.capacity
            for length, cache in zip(lengths, caches, strict=True)
        ):
            raise ValueError("Decode would exceed request KV capacity.")
        key_length = max(lengths) + 1
        positions = torch.tensor(lengths, device=self.device).unsqueeze(1)
        batch = caches[0]._decode_batch
        if batch is None or not batch.matches(caches):
            batch = DenseDecodeBatch(caches)
        layers = tuple(
            (keys[:, :, :key_length], values[:, :, :key_length])
            for keys, values in batch.layers
        )
        mask = None
        if len(set(lengths)) > 1:
            mask = (torch.arange(key_length, device=self.device)[None, :] <= positions)[
                :, None, None, :
            ]
        logits = self.model.decode_forward(tokens, layers, positions, mask)
        for cache in caches:
            cache.advance(1)
        return logits

    def decode(
        self,
        prefill: PrefillResult,
        eos_token_id: int,
        sampling: Sampling,
        *,
        request_id: str = "internal",
    ) -> DecodedTokens:
        first_token_at = None
        last_token_at = None
        token_intervals = []
        generated: list[int] = []
        inter_token_seconds: list[float] = []
        finish_reason = "length"
        logits = prefill.logits
        cache = prefill.cache
        self.model.eval()
        with torch.inference_mode():
            for index in range(sampling.max_new_tokens):
                next_token = self._sample(logits, sampling)
                token_id = next_token.item()
                sampled_at = time.perf_counter()
                if first_token_at is None:
                    first_token_at = sampled_at
                if token_id == eos_token_id:
                    finish_reason = "eos"
                    break
                if last_token_at is not None:
                    token_intervals.append(sampled_at - last_token_at)
                last_token_at = sampled_at
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
                        (
                            prefill.prefill_seconds
                            if index == 0
                            else inter_token_seconds[-1]
                        )
                        * 1_000,
                    )
                if index + 1 < sampling.max_new_tokens:
                    self._synchronize()
                    started = time.perf_counter()
                    forward_logits = self.decode_caches([cache], [token_id]).unsqueeze(
                        1
                    )
                    self._validate_forward_shapes(next_token, forward_logits)
                    self._synchronize()
                    inter_token_seconds.append(time.perf_counter() - started)
                    logits = forward_logits[:, -1, :]
        return DecodedTokens(
            first_token_at=first_token_at,
            token_intervals=tuple(token_intervals),
            output_ids=generated,
            finish_reason=finish_reason,
            inter_token_seconds=inter_token_seconds,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _forward_cache(
        self, tokens: torch.Tensor, cache: KVCache | PagedKVCache
    ) -> torch.Tensor:
        if isinstance(cache, PagedKVCache):
            batch = PagedBatchCache([cache])
            batch.prepare(tokens.shape[1])
            return self.model(tokens, cache=batch)
        return self.model(tokens, cache=cache)

    @staticmethod
    def release_cache(cache: KVCache | PagedKVCache) -> None:
        if isinstance(cache, PagedKVCache):
            cache.close()
        else:
            cache._layers.clear()
            cache._decode_batch = None

    def dense_reservation_bytes(
        self, caches: list[KVCache], *, extra_capacity: int = 0
    ) -> int:
        config = self.model.config
        bytes_per_token = (
            2 * config.n_layers * config.n_kv_heads * config.head_dim
            * torch.empty((), dtype=config.dtype).element_size()
        )
        owners = {}
        for cache in caches:
            batch = cache._decode_batch
            owner = batch if batch is not None else cache
            owners[id(owner)] = batch.token_capacity if batch is not None else cache.capacity
        allocated = sum(owners.values()) + extra_capacity
        capacities = [cache.capacity for cache in caches]
        if extra_capacity:
            capacities.append(extra_capacity)
        if not capacities:
            return 0
        batch = caches[0]._decode_batch if caches else None
        reusable = not extra_capacity and batch is not None and batch.matches(caches)
        fresh_single = len(capacities) == 1 and batch is None
        rebuild = 0 if reusable or fresh_single else len(capacities) * max(capacities)
        return (allocated + rebuild) * bytes_per_token

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
        if sampling.top_p == 1:
            return torch.multinomial(probabilities, num_samples=1)
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
