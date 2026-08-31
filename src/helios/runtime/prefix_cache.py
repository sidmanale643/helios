import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from helios.runtime.qwen3.cache import KVBlockSnapshot, KVCache


@dataclass(frozen=True)
class TokenBlock:
    tokens: tuple[int, ...]
    hash: str


@dataclass(frozen=True)
class CachedBlock:
    tokens: tuple[int, ...]
    snapshot: KVBlockSnapshot


@dataclass(frozen=True)
class PrefixCacheHit:
    blocks: tuple[CachedBlock, ...]

    @property
    def length(self) -> int:
        return sum(block.snapshot.length for block in self.blocks)


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
    parent_hash = b""

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
        parent_hash = hashlib.sha256(parent_hash + payload).digest()
        hashed_blocks.append(TokenBlock(tokens=tokens, hash=parent_hash.hex()))

    return hashed_blocks


def build_token_blocks(token_ids: Sequence[int], block_size: int) -> list[TokenBlock]:
    return hash_token_blocks(split_token_stream(token_ids, block_size))


class PrefixCache:
    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        self.block_size = block_size
        self._blocks: dict[str, CachedBlock] = {}

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
            )
            stored += 1
        return stored
