import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from helios.runtime.qwen3.cache import KVBlockSnapshot, KVCache


@dataclass(frozen=True)
class TokenBlock:
    tokens: tuple[int, ...]
    hash: str
    parent_hash: str


@dataclass
class CachedBlock:
    tokens: tuple[int, ...]
    snapshot: KVBlockSnapshot
    hash: str = ""
    parent_hash: str = ""
    hit_count: int = 0


@dataclass(frozen=True)
class PrefixCacheHit:
    blocks: tuple[CachedBlock, ...]

    @property
    def length(self) -> int:
        return sum(block.snapshot.length for block in self.blocks)


@dataclass(frozen=True)
class PromptBlockView:
    index: int
    token_count: int
    hash: str
    parent_hash: str
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "token_count": self.token_count,
            "hash": self.hash,
            "parent_hash": self.parent_hash,
            "status": self.status,
        }


@dataclass(frozen=True)
class PrefixBlockInfo:
    hash: str
    parent_hash: str
    token_count: int
    hit_count: int
    memory_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "hash": self.hash,
            "parent_hash": self.parent_hash,
            "token_count": self.token_count,
            "hit_count": self.hit_count,
            "memory_bytes": self.memory_bytes,
        }


def split_token_stream(
    token_ids: Sequence[int], block_size: int
) -> list[tuple[int, ...]]:
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    return [
        tuple(token_ids[start : start + block_size])
        for start in range(0, len(token_ids), block_size)
    ]


def hash_token_blocks(blocks: Sequence[Sequence[int]]) -> list[TokenBlock]:
    hashed_blocks: list[TokenBlock] = []
    parent_digest = b""
    parent_hex = ""

    for block in blocks:
        tokens = tuple(block)
        if not tokens:
            raise ValueError("Token blocks must not be empty.")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in tokens
        ):
            raise ValueError("Token IDs must be non-negative integers.")

        payload = json.dumps(tokens, separators=(",", ":")).encode("ascii")
        digest = hashlib.sha256(parent_digest + payload).digest()
        hashed_blocks.append(
            TokenBlock(tokens=tokens, hash=digest.hex(), parent_hash=parent_hex)
        )
        parent_digest = digest
        parent_hex = digest.hex()

    return hashed_blocks


def build_token_blocks(token_ids: Sequence[int], block_size: int) -> list[TokenBlock]:
    return hash_token_blocks(split_token_stream(token_ids, block_size))


def describe_prompt_blocks(
    token_ids: Sequence[int],
    block_size: int,
    hit: PrefixCacheHit | None,
) -> tuple[PromptBlockView, ...]:
    hit_blocks = 0 if hit is None else len(hit.blocks)
    views: list[PromptBlockView] = []
    for index, block in enumerate(build_token_blocks(token_ids, block_size)):
        if len(block.tokens) != block_size:
            status = "partial"
        elif index < hit_blocks:
            status = "hit"
        else:
            status = "miss"
        views.append(
            PromptBlockView(
                index=index,
                token_count=len(block.tokens),
                hash=block.hash,
                parent_hash=block.parent_hash,
                status=status,
            )
        )
    return tuple(views)


class PrefixCache:
    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        self.block_size = block_size
        self._blocks: dict[str, CachedBlock] = {}

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def token_count(self) -> int:
        return sum(len(block.tokens) for block in self._blocks.values())

    @property
    def memory_bytes(self) -> int:
        return sum(
            self._snapshot_bytes(block.snapshot) for block in self._blocks.values()
        )

    def get(self, block_hash: str) -> CachedBlock | None:
        return self._blocks.get(block_hash)

    def longest_prefix(self, token_ids: Sequence[int]) -> PrefixCacheHit | None:
        matched: list[CachedBlock] = []
        for block in build_token_blocks(token_ids, self.block_size):
            if len(block.tokens) != self.block_size:
                break
            cached = self._blocks.get(block.hash)
            if cached is None:
                break
            cached.hit_count += 1
            matched.append(cached)
        return PrefixCacheHit(tuple(matched)) if matched else None

    def store_completed_blocks(self, token_ids: Sequence[int], cache: KVCache) -> int:
        completed_length = len(token_ids) // self.block_size * self.block_size
        if completed_length > cache.length:
            raise ValueError("KV cache has not processed every completed token block.")

        blocks = build_token_blocks(token_ids[:completed_length], self.block_size)
        stored = 0
        for index, block in enumerate(blocks):
            if block.hash in self._blocks:
                continue
            start = index * self.block_size
            end = start + self.block_size
            self._blocks[block.hash] = CachedBlock(
                tokens=block.tokens,
                snapshot=cache.snapshot_block(start, end),
                hash=block.hash,
                parent_hash=block.parent_hash,
            )
            stored += 1
        return stored

    def blocks(self) -> list[PrefixBlockInfo]:
        return [
            PrefixBlockInfo(
                hash=block.hash or key,
                parent_hash=block.parent_hash,
                token_count=len(block.tokens),
                hit_count=block.hit_count,
                memory_bytes=self._snapshot_bytes(block.snapshot),
            )
            for key, block in self._blocks.items()
        ]

    @staticmethod
    def _snapshot_bytes(snapshot: KVBlockSnapshot) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in snapshot.layers
            for tensor in (layer.keys, layer.values)
        )
