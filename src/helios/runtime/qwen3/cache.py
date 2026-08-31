from collections.abc import Sequence
from dataclasses import dataclass
import torch

from helios.runtime.qwen3.config import Qwen3Config

@dataclass
class LayerKV:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class LayerKVSnapshot:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class KVCacheSnapshot:
    length: int
    layers: tuple[LayerKVSnapshot, ...]


@dataclass(frozen=True)
class KVBlockSnapshot:
    length: int
    layers: tuple[LayerKVSnapshot, ...]


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

    def snapshot(self, length: int | None = None) -> KVCacheSnapshot:
        length = self.length if length is None else length
        if (
            not isinstance(length, int)
            or isinstance(length, bool)
            or not 0 <= length <= self.length
        ):
            raise ValueError(
                f"Snapshot length must be between 0 and {self.length:,} tokens."
            )

        return KVCacheSnapshot(
            length=length,
            layers=tuple(
                LayerKVSnapshot(
                    keys=layer.keys[:, :, :length].detach().clone(),
                    values=layer.values[:, :, :length].detach().clone(),
                )
                for layer in self._layers
            ),
        )

    def snapshot_block(self, start: int, end: int) -> KVBlockSnapshot:
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= self.length
        ):
            raise ValueError(
                f"Block bounds must satisfy 0 <= start < end <= {self.length:,}."
            )

        return KVBlockSnapshot(
            length=end - start,
            layers=tuple(
                LayerKVSnapshot(
                    keys=layer.keys[:, :, start:end].detach().clone(),
                    values=layer.values[:, :, start:end].detach().clone(),
                )
                for layer in self._layers
            ),
        )

    def restore(self, snapshot: KVCacheSnapshot) -> None:
        if (
            not isinstance(snapshot.length, int)
            or isinstance(snapshot.length, bool)
            or not 0 <= snapshot.length <= self.capacity
        ):
            raise ValueError(
                f"Snapshot length must fit within the {self.capacity:,}-token cache."
            )
        if len(snapshot.layers) != len(self._layers):
            raise ValueError("Snapshot layer count does not match this Qwen3 model.")

        for destination, source in zip(self._layers, snapshot.layers, strict=True):
            expected_shape = (
                destination.keys.shape[0],
                destination.keys.shape[1],
                snapshot.length,
                destination.keys.shape[3],
            )
            if (
                source.keys.shape != expected_shape
                or source.values.shape != expected_shape
            ):
                raise ValueError("Snapshot tensor shape does not match this KV cache.")
            if (
                source.keys.dtype != destination.keys.dtype
                or source.values.dtype != destination.values.dtype
            ):
                raise ValueError("Snapshot tensor dtype does not match this KV cache.")
            if (
                source.keys.device != destination.keys.device
                or source.values.device != destination.values.device
            ):
                raise ValueError("Snapshot tensor device does not match this KV cache.")

        for destination, source in zip(self._layers, snapshot.layers, strict=True):
            destination.keys[:, :, :snapshot.length].copy_(source.keys)
            destination.values[:, :, :snapshot.length].copy_(source.values)
        self.length = snapshot.length

    def restore_blocks(self, blocks: Sequence[KVBlockSnapshot]) -> None:
        total_length = 0
        for block in blocks:
            if (
                not isinstance(block.length, int)
                or isinstance(block.length, bool)
                or block.length < 1
            ):
                raise ValueError("KV block length must be a positive integer.")
            total_length += block.length
            if total_length > self.capacity:
                raise ValueError(
                    f"KV blocks must fit within the {self.capacity:,}-token cache."
                )
            if len(block.layers) != len(self._layers):
                raise ValueError("KV block layer count does not match this Qwen3 model.")

            for destination, source in zip(
                self._layers, block.layers, strict=True
            ):
                expected_shape = (
                    destination.keys.shape[0],
                    destination.keys.shape[1],
                    block.length,
                    destination.keys.shape[3],
                )
                if (
                    source.keys.shape != expected_shape
                    or source.values.shape != expected_shape
                ):
                    raise ValueError("KV block tensor shape does not match this cache.")
                if (
                    source.keys.dtype != destination.keys.dtype
                    or source.values.dtype != destination.values.dtype
                ):
                    raise ValueError("KV block tensor dtype does not match this cache.")
                if (
                    source.keys.device != destination.keys.device
                    or source.values.device != destination.values.device
                ):
                    raise ValueError("KV block tensor device does not match this cache.")

        offset = 0
        for block in blocks:
            end = offset + block.length
            for destination, source in zip(
                self._layers, block.layers, strict=True
            ):
                destination.keys[:, :, offset:end].copy_(source.keys)
                destination.values[:, :, offset:end].copy_(source.values)
            offset = end
        self.length = total_length
