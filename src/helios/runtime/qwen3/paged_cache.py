from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch.nn.attention.varlen import varlen_attn

from helios.runtime.qwen3.config import Qwen3Config


class KVPage:
    __slots__ = ("__weakref__", "_finalizer", "index", "pool")

    def __init__(self, pool: KVPagePool, index: int) -> None:
        self.index = index
        self.pool = pool
        self._finalizer = weakref.finalize(self, pool._release, index)


@dataclass(frozen=True)
class PagedKVBlockSnapshot:
    length: int
    pages: tuple[KVPage, ...]
    memory_bytes: int


class KVPagePool:
    def __init__(
        self,
        config: Qwen3Config,
        num_pages: int,
        *,
        device: torch.device,
        page_size: int = 256,
    ) -> None:
        if num_pages < 1:
            raise ValueError("KV page pool must contain at least one page.")
        if page_size < 1 or page_size % 256:
            raise ValueError("KV page size must be a positive multiple of 256.")
        self.config = config
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        shape = (num_pages, page_size, config.n_kv_heads, config.head_dim)
        self.keys = tuple(
            torch.empty(shape, dtype=config.dtype, device=device)
            for _ in range(config.n_layers)
        )
        self.values = tuple(
            torch.empty(shape, dtype=config.dtype, device=device)
            for _ in range(config.n_layers)
        )
        self._free = list(range(num_pages - 1, -1, -1))

    @property
    def bytes_per_token(self) -> int:
        return (
            self.config.n_layers
            * 2
            * self.config.n_kv_heads
            * self.config.head_dim
            * self.keys[0].element_size()
        )

    @property
    def bytes_per_page(self) -> int:
        return self.bytes_per_token * self.page_size

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def acquire(self, count: int) -> list[KVPage]:
        if count < 0:
            raise ValueError("Requested page count must not be negative.")
        if count > len(self._free):
            raise RuntimeError(
                f"KV page pool needs {count} pages but only {len(self._free)} are free."
            )
        return [KVPage(self, self._free.pop()) for _ in range(count)]

    def _release(self, index: int) -> None:
        self._free.append(index)


class PagedKVCache:
    batch_size = 1

    def __init__(self, pool: KVPagePool, capacity: int) -> None:
        if not 1 <= capacity <= pool.config.context_length:
            raise ValueError(
                f"KV-cache capacity must be between 1 and {pool.config.context_length:,} tokens."
            )
        self.pool = pool
        self.capacity = capacity
        self.pages: list[KVPage] = []
        self.length = 0
        self._pending_tokens: int | None = None

    @property
    def memory_bytes_per_token(self) -> int:
        return self.pool.bytes_per_token

    @property
    def memory_bytes_per_slot_token(self) -> int:
        return self.pool.bytes_per_token

    def slot_length(self, slot: int) -> int:
        if slot != 0:
            raise ValueError("Single-request paged caches have only slot 0.")
        return self.length

    def snapshot_block_slot(
        self, slot: int, start: int, end: int
    ) -> PagedKVBlockSnapshot:
        if slot != 0:
            raise ValueError("Single-request paged caches have only slot 0.")
        page_size = self.pool.page_size
        if (
            start < 0
            or end <= start
            or end > self.length
            or start % page_size
            or end % page_size
        ):
            raise ValueError("Paged K/V snapshots require complete aligned pages.")
        pages = tuple(self.pages[start // page_size : end // page_size])
        if len(pages) * page_size != end - start:
            raise RuntimeError("Paged K/V snapshot is missing a logical page.")
        return PagedKVBlockSnapshot(
            length=end - start,
            pages=pages,
            memory_bytes=len(pages) * self.pool.bytes_per_page,
        )

    def snapshot_block(self, start: int, end: int) -> PagedKVBlockSnapshot:
        return self.snapshot_block_slot(0, start, end)

    def restore_blocks(self, blocks: Sequence[PagedKVBlockSnapshot]) -> None:
        if self.length or self.pages or self._pending_tokens is not None:
            raise RuntimeError("Paged K/V blocks can only restore into an empty cache.")
        pages: list[KVPage] = []
        for block in blocks:
            if (
                block.length < 1
                or block.length % self.pool.page_size
                or len(block.pages) * self.pool.page_size != block.length
                or any(page.pool is not self.pool for page in block.pages)
            ):
                raise ValueError(
                    "Paged K/V block does not belong to this aligned pool."
                )
            pages.extend(block.pages)
        length = len(pages) * self.pool.page_size
        if length > self.capacity:
            raise ValueError(
                "Paged K/V blocks do not fit within this request capacity."
            )
        self.pages = pages
        self.length = length

    def close(self) -> None:
        self.pages.clear()
        self.length = 0
        self._pending_tokens = None


class PagedBatchCache:
    def __init__(self, caches: Sequence[PagedKVCache]) -> None:
        if not caches:
            raise ValueError("Paged batch cache needs at least one request cache.")
        pool = caches[0].pool
        if any(cache.pool is not pool for cache in caches):
            raise ValueError("Paged batch caches must share one KV page pool.")
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("Paged batches require distinct request caches.")
        self._buffers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._metadata_values: dict[str, list] = {}
        self._block_signature = None
        self.caches = tuple(caches)
        self.pool = pool
        self.batch_size = len(caches)

    @property
    def length(self) -> int:
        if self.batch_size != 1:
            raise RuntimeError("Batched paged caches have independent lengths.")
        return self.caches[0].length

    def slot_ids(self, slots: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        if isinstance(slots, torch.Tensor):
            if slots.ndim != 1:
                raise ValueError("Paged batch slots must be one-dimensional.")
            rows = tuple(int(value) for value in slots.cpu().tolist())
        else:
            rows = tuple(slots)
        if rows != tuple(range(self.batch_size)):
            raise ValueError("Paged batches require every active cache in order.")
        return rows

    def slot_length(self, slot: int) -> int:
        if not 0 <= slot < self.batch_size:
            raise ValueError("Paged batch slot is outside the active batch.")
        return self.caches[slot].length

    def slot_lengths(self, slots: Sequence[int] | torch.Tensor) -> torch.Tensor:
        rows = self.slot_ids(slots)
        return self.metadata(
            "positions", [self.caches[row].length for row in rows], torch.long
        )

    def prepare(self, tokens: int) -> None:
        if tokens < 1:
            raise ValueError("Paged KV append needs at least one token.")
        page_size = self.pool.page_size
        missing_pages: list[int] = []
        for cache in self.caches:
            if cache._pending_tokens is not None:
                raise RuntimeError("Paged KV cache already has a pending append.")
            end = cache.length + tokens
            if end > cache.capacity:
                raise ValueError("Paged KV append exceeds its request capacity.")
            missing_pages.append(max(0, _pages_for(end, page_size) - len(cache.pages)))
        total_missing = sum(missing_pages)
        if total_missing > self.pool.free_pages:
            raise RuntimeError(
                f"KV page pool needs {total_missing} pages but only {self.pool.free_pages} are free."
            )
        for cache, missing in zip(self.caches, missing_pages, strict=True):
            cache.pages.extend(self.pool.acquire(missing))
            cache._pending_tokens = tokens
        tables = [[page.index for page in cache.pages] for cache in self.caches]
        width = max(_pages_for(cache.capacity, page_size) for cache in self.caches)
        signature = (width, tuple(tuple(table) for table in tables))
        if signature != self._block_signature:
            self.block_table = self.metadata(
                "block_table",
                [table + [0] * (width - len(table)) for table in tables],
                torch.int32,
            )
            self._block_signature = signature
        ends = [cache.length + tokens for cache in self.caches]
        self.seqused_k = self.metadata("seqused_k", ends, torch.int32)
        self.cu_seq_q = self.metadata(
            "cu_seq_q",
            [row * tokens for row in range(self.batch_size + 1)],
            torch.int32,
        )
        self.cu_seq_k = self.metadata("cu_seq_k", [0, *accumulate(ends)], torch.int32)
        self.write_slots = self.metadata(
            "write_slots",
            [
                cache.pages[position // page_size].index * page_size
                + position % page_size
                for cache in self.caches
                for position in range(cache.length, cache.length + tokens)
            ],
            torch.long,
        )
        self.max_q = tokens
        self.max_k = max(ends)

    def metadata(self, name: str, values: list, dtype: torch.dtype) -> torch.Tensor:
        shape = (
            (len(values), len(values[0]))
            if isinstance(values[0], list)
            else (len(values),)
        )
        pair = self._buffers.get(name)
        if pair is None or pair[0].shape != shape:
            host = torch.empty(shape, dtype=dtype, device="cpu")
            device = torch.empty(shape, dtype=dtype, device=self.pool.device)
            pair = self._buffers[name] = (host, device)
        host, device = pair
        if self._metadata_values.get(name) != values:
            host.numpy()[...] = values
            device.copy_(host)
            self._metadata_values[name] = values
        return device

    def advance(
        self, tokens: int, *, slots: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        if slots is not None:
            self.slot_ids(slots)
        if any(cache._pending_tokens != tokens for cache in self.caches):
            raise RuntimeError("Paged KV advance must match the prepared token count.")
        for cache in self.caches:
            cache.length += tokens
            cache._pending_tokens = None

    def attend(
        self,
        layer: int,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        tokens = _validate_attention_inputs(self, layer, queries, keys, values)
        if any(cache._pending_tokens != tokens for cache in self.caches):
            raise RuntimeError("Call prepare(tokens) before paged attention.")
        self._write(layer, keys, values)
        if self.pool.device.type == "cuda":
            packed_queries = queries.transpose(1, 2).reshape(
                -1, queries.shape[1], queries.shape[3]
            )
            output = varlen_attn(
                packed_queries,
                self.pool.keys[layer],
                self.pool.values[layer],
                self.cu_seq_q,
                self.cu_seq_k,
                self.max_q,
                self.max_k,
                window_size=(-1, 0),
                enable_gqa=queries.shape[1] > self.pool.config.n_kv_heads,
                seqused_k=self.seqused_k,
                block_table=self.block_table,
                num_splits=1,
            )
            return output.reshape(
                self.batch_size, tokens, queries.shape[1], queries.shape[3]
            ).transpose(1, 2)
        return _cpu_attention(self, layer, queries)

    def _write(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        packed_keys = keys.transpose(1, 2).reshape(-1, keys.shape[1], keys.shape[3])
        packed_values = values.transpose(1, 2).reshape(
            -1, values.shape[1], values.shape[3]
        )
        self.pool.keys[layer].view(-1, keys.shape[1], keys.shape[3]).index_copy_(
            0, self.write_slots, packed_keys
        )
        self.pool.values[layer].view(-1, values.shape[1], values.shape[3]).index_copy_(
            0, self.write_slots, packed_values
        )


def _pages_for(tokens: int, page_size: int) -> int:
    return (tokens + page_size - 1) // page_size


def _validate_attention_inputs(
    cache: PagedBatchCache,
    layer: int,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> int:
    config = cache.pool.config
    if not 0 <= layer < config.n_layers:
        raise ValueError("Paged attention layer index is outside the model.")
    if (
        queries.ndim != 4
        or keys.ndim != 4
        or values.shape != keys.shape
        or queries.shape[0] != cache.batch_size
        or keys.shape[0] != cache.batch_size
        or queries.shape[1] != config.n_heads
        or queries.shape[2] != keys.shape[2]
        or queries.shape[-1] != config.head_dim
        or keys.shape[1] != config.n_kv_heads
        or keys.shape[-1] != config.head_dim
    ):
        raise ValueError("Paged attention tensors do not match the Qwen3 KV shape.")
    return keys.shape[2]


def _cpu_attention(
    cache: PagedBatchCache, layer: int, queries: torch.Tensor
) -> torch.Tensor:
    outputs = []
    for row, request in enumerate(cache.caches):
        end = request.length + queries.shape[2]
        keys = _gather_cpu(cache.pool.keys[layer], request.pages, end)
        values = _gather_cpu(cache.pool.values[layer], request.pages, end)
        if queries.shape[1] != keys.shape[1]:
            repeat = queries.shape[1] // keys.shape[1]
            if repeat * keys.shape[1] != queries.shape[1]:
                raise ValueError("Query heads must be a multiple of KV heads for GQA.")
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)
        positions = torch.arange(end, device=queries.device)
        query_positions = request.length + torch.arange(
            queries.shape[2], device=queries.device
        )
        mask = positions[None, None, None, :] <= query_positions[None, None, :, None]
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                queries[row : row + 1], keys, values, attn_mask=mask, dropout_p=0.0
            )
        )
    return torch.cat(outputs, dim=0)


def _gather_cpu(
    storage: torch.Tensor, pages: Sequence[KVPage], length: int
) -> torch.Tensor:
    indices = torch.tensor([page.index for page in pages], device=storage.device)
    return (
        storage.index_select(0, indices)
        .reshape(-1, *storage.shape[2:])[:length]
        .permute(1, 0, 2)
        .unsqueeze(0)
    )


class PackedBatchCache(PagedBatchCache):
    def prepare_packed(self, counts: Sequence[int]) -> None:
        if len(counts) != len(self.caches) or any(count < 1 for count in counts):
            raise ValueError("Each packed sequence needs at least one query token.")
        missing = []
        for cache, count in zip(self.caches, counts, strict=True):
            if (
                cache._pending_tokens is not None
                or cache.length + count > cache.capacity
            ):
                raise ValueError(
                    "Packed append exceeds capacity or an append is pending."
                )
            missing.append(
                max(
                    0,
                    _pages_for(cache.length + count, self.pool.page_size)
                    - len(cache.pages),
                )
            )
        if sum(missing) > self.pool.free_pages:
            raise RuntimeError("Insufficient free pages for packed append.")
        for cache, count, pages in zip(self.caches, counts, missing, strict=True):
            cache.pages.extend(self.pool.acquire(pages))
            cache._pending_tokens = count
        self.counts = tuple(counts)
        offsets = [0, *accumulate(counts)]
        ends = [
            cache.length + count
            for cache, count in zip(self.caches, counts, strict=True)
        ]
        tables = [[page.index for page in cache.pages] for cache in self.caches]
        width = max(
            _pages_for(cache.capacity, self.pool.page_size) for cache in self.caches
        )
        self.block_table = self.metadata(
            "block_table",
            [row + [0] * (width - len(row)) for row in tables],
            torch.int32,
        )
        self.cu_seq_q = self.metadata("cu_seq_q", offsets, torch.int32)
        self.cu_seq_k = self.metadata("cu_seq_k", [0, *accumulate(ends)], torch.int32)
        self.seqused_k = self.metadata("seqused_k", ends, torch.int32)
        positions = [
            position
            for cache, end in zip(self.caches, ends, strict=True)
            for position in range(cache.length, end)
        ]
        self.positions = self.metadata("positions", positions, torch.long).unsqueeze(0)
        self.last_indices = self.metadata(
            "last_indices", [offset - 1 for offset in offsets[1:]], torch.long
        )
        self.write_slots = self.metadata(
            "write_slots",
            [
                cache.pages[position // self.pool.page_size].index * self.pool.page_size
                + position % self.pool.page_size
                for cache, end in zip(self.caches, ends, strict=True)
                for position in range(cache.length, end)
            ],
            torch.long,
        )
        self.max_q, self.max_k = max(counts), max(ends)

    def advance_packed(self) -> None:
        for cache, count in zip(self.caches, self.counts, strict=True):
            cache.length += count
            cache._pending_tokens = None

    def attend(self, layer, queries, keys, values):
        if queries.shape[0] != 1 or queries.shape[2] != sum(self.counts):
            raise ValueError("Packed attention expects one flattened token dimension.")
        self._write(layer, keys, values)
        if self.pool.device.type == "cuda":
            output = varlen_attn(
                queries[0].transpose(0, 1),
                self.pool.keys[layer],
                self.pool.values[layer],
                self.cu_seq_q,
                self.cu_seq_k,
                self.max_q,
                self.max_k,
                window_size=(-1, 0),
                enable_gqa=queries.shape[1] > self.pool.config.n_kv_heads,
                seqused_k=self.seqused_k,
                block_table=self.block_table,
                num_splits=1,
            )
            return output.transpose(0, 1).unsqueeze(0)
        outputs = []
        offset = 0
        for cache, count in zip(self.caches, self.counts, strict=True):
            end = cache.length + count
            k = _gather_cpu(self.pool.keys[layer], cache.pages, end)
            v = _gather_cpu(self.pool.values[layer], cache.pages, end)
            repeat = queries.shape[1] // k.shape[1]
            k, v = (
                k.repeat_interleave(repeat, dim=1),
                v.repeat_interleave(repeat, dim=1),
            )
            mask = (
                torch.arange(end)[None, :]
                <= (cache.length + torch.arange(count))[:, None]
            )
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    queries[:, :, offset : offset + count],
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=0.0,
                )
            )
            offset += count
        return torch.cat(outputs, dim=2)
