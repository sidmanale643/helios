from dataclasses import dataclass

import torch

from helios.runtime.qwen3.config import Qwen3Config


@dataclass
class LayerKV:
    keys: torch.Tensor
    values: torch.Tensor


class KVCache:

    def __init__(
        self,
        config: Qwen3Config,
        capacity: int,
        *,
        device: torch.device,
        batch_size: int = 1,
    ) -> None:
        if not 1 <= capacity <= config.context_length:
            raise ValueError(
                f"KV-cache capacity must be between 1 and {config.context_length:,} tokens."
            )
        self.capacity = capacity
        self.length = 0
        self._layers = [
            LayerKV(
                keys=torch.empty(
                    batch_size, config.n_kv_heads, capacity, config.head_dim,
                    device=device, dtype=config.dtype,
                ),
                values=torch.empty(
                    batch_size, config.n_kv_heads, capacity, config.head_dim,
                    device=device, dtype=config.dtype,
                ),
            )
            for _ in range(config.n_layers)
        ]

    def append(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape != values.shape:
            raise ValueError("Key and value tensors must have identical shapes.")
        if keys.shape[:2] != self._layers[layer].keys.shape[:2] or keys.shape[-1] != self._layers[layer].keys.shape[-1]:
            raise ValueError("KV-cache tensor shape does not match this Qwen3 model.")
        end = self.length + keys.shape[2]
        if end > self.capacity:
            raise ValueError(
                f"KV cache overflow: requested {end:,} tokens; capacity is {self.capacity:,}."
            )
        entry = self._layers[layer]
        entry.keys[:, :, self.length:end].copy_(keys)
        entry.values[:, :, self.length:end].copy_(values)
        return entry.keys[:, :, :end], entry.values[:, :, :end]

    def advance(self, tokens: int) -> None:
        if tokens < 1 or self.length + tokens > self.capacity:
            raise ValueError("Cannot advance the KV cache beyond its capacity.")
        self.length += tokens
