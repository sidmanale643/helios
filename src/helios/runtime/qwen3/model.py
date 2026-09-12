from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.attention.bias import causal_lower_right

from helios.runtime.qwen3.cache import BatchedKVCache, DecodeKVCache, KVCache
from helios.runtime.qwen3.config import Qwen3Config
from helios.runtime.qwen3.layers import RMSNorm, TransformerBlock, rope_parameters
from helios.runtime.qwen3.paged_cache import PackedBatchCache, PagedBatchCache


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=config.dtype
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype
        )
        self.output.weight = self.token_embedding.weight
        cos, sin = rope_parameters(config)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | BatchedKVCache | PagedBatchCache | None = None,
        position_ids: torch.Tensor | None = None,
        cache_slots: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(cache, PackedBatchCache):
            return self._forward_packed(input_ids, cache)
        if cache_slots is None:
            return self._forward_uniform_cache(input_ids, cache, position_ids)

        tokens = input_ids.shape[1]
        if cache is None:
            raise ValueError("Cache slots require a KV cache.")
        if isinstance(cache, KVCache):
            raise TypeError("Single-request KV caches do not accept cache slots.")
        cache_slots = cache.slot_ids(cache_slots)
        slot_starts = tuple(cache.slot_length(slot) for slot in cache_slots)
        if max(start + tokens for start in slot_starts) > self.config.context_length:
            raise ValueError(
                f"Qwen3-4B supports at most {self.config.context_length:,} tokens per request."
            )
        starts = cache.slot_lengths(cache_slots)
        ends = starts + tokens
        x = self.token_embedding(input_ids)
        key_length = max(start + tokens for start in slot_starts)
        mask = None
        uniform_start = len(set(slot_starts)) == 1
        is_causal = False
        paged = isinstance(cache, PagedBatchCache)
        if not paged and tokens > 1 and uniform_start:
            if slot_starts[0] == 0:
                is_causal = True
            else:
                mask = causal_lower_right(tokens, key_length)
        elif not paged and not uniform_start:
            key_positions = torch.arange(key_length, device=x.device)
            query_positions = starts[:, None] + torch.arange(tokens, device=x.device)
            mask = key_positions[None, None, :] <= query_positions[:, :, None]
            mask &= key_positions[None, None, :] < ends[:, None, None]
            mask = mask[:, None, :, :]
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                mask,
                self.cos,
                self.sin,
                is_causal=is_causal,
                start_pos=slot_starts[0],
                cache=cache,
                layer_index=index,
                position_ids=position_ids,
                cache_slots=cache_slots,
            )
        if cache is not None:
            cache.advance(tokens, slots=cache_slots)
        x = x[:, -1:, :]
        return self.output(self.final_norm(x).to(self.config.dtype))

    def _forward_packed(
        self, input_ids: torch.Tensor, cache: PackedBatchCache
    ) -> torch.Tensor:
        if input_ids.shape != (1, sum(cache.counts)):
            raise ValueError("Packed input does not match query lengths.")
        x = self.token_embedding(input_ids)
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                None,
                self.cos,
                self.sin,
                cache=cache,
                layer_index=index,
                position_ids=cache.positions,
            )
        cache.advance_packed()
        x = x.index_select(1, cache.last_indices)
        return self.output(self.final_norm(x).to(self.config.dtype))[0]

    def decode_forward(
        self,
        input_ids: torch.Tensor,
        layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        position_ids: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cache = DecodeKVCache(layers, position_ids)
        x = self.token_embedding(input_ids)
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                mask,
                self.cos,
                self.sin,
                cache=cache,
                layer_index=index,
                position_ids=position_ids,
            )
        return self.output(self.final_norm(x).to(self.config.dtype))

    def _forward_uniform_cache(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | PagedBatchCache | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        start_pos = cache.length if cache is not None else 0
        end_pos = start_pos + input_ids.shape[-1]
        if end_pos > self.config.context_length:
            raise ValueError(
                f"Qwen3-4B supports at most {self.config.context_length:,} tokens per request."
            )
        x = self.token_embedding(input_ids)
        tokens = x.shape[1]
        mask = None
        is_causal = False
        if tokens > 1 and not isinstance(cache, PagedBatchCache):
            if start_pos == 0:
                is_causal = True
            else:
                mask = causal_lower_right(tokens, end_pos)
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                mask,
                self.cos,
                self.sin,
                is_causal=is_causal,
                start_pos=start_pos,
                cache=cache,
                layer_index=index,
                position_ids=position_ids,
            )
        if cache is not None:
            cache.advance(tokens)
        x = x[:, -1:, :]
        return self.output(self.final_norm(x).to(self.config.dtype))
