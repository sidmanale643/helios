MAX_NGRAM = 4
LOOKUP_WINDOW = 4096
MAX_COOLDOWN_TICKS = 32


class NgramDrafter:
    """Per-request prompt-lookup drafter with an incremental n-gram index.

    Each indexed n-gram maps to the position just after its latest occurrence,
    so a lookup costs a few dict probes instead of rescanning the history.
    The draft length adapts to recent acceptance, and repeated full rejections
    back off exponentially so poor matches stop paying for verification.
    """

    def __init__(
        self, history: list[int], *, max_tokens: int, min_ngram: int
    ) -> None:
        self.tokens: list[int] = []
        self.max_tokens = max_tokens
        self.min_ngram = min_ngram
        self.draft_tokens = max_tokens
        self._positions: dict[tuple[int, ...], int] = {}
        self._misses = 0
        self._cooldown = 0
        self.extend(history[-LOOKUP_WINDOW:])

    def extend(self, tokens: list[int]) -> None:
        for token in tokens:
            self.append(token)

    def append(self, token: int) -> None:
        # Index the n-grams ending at the current tail now that their
        # continuation exists. The current suffix is indexed only after the
        # next token arrives, so lookups always find an earlier occurrence.
        end = len(self.tokens)
        for size in range(self.min_ngram, min(MAX_NGRAM, end) + 1):
            self._positions[tuple(self.tokens[end - size : end])] = end
        self.tokens.append(token)

    def propose(self, limit: int) -> list[int]:
        if self._cooldown:
            self._cooldown -= 1
            return []
        limit = min(limit, self.draft_tokens)
        if limit <= 0:
            return []
        end = len(self.tokens)
        for size in range(min(MAX_NGRAM, end), self.min_ngram - 1, -1):
            start = self._positions.get(tuple(self.tokens[end - size :]))
            if start is not None:
                return self.tokens[start : start + limit]
        return []

    def record(self, proposed: int, accepted: int) -> None:
        if not proposed:
            return
        if accepted == proposed:
            self.draft_tokens = min(self.max_tokens, self.draft_tokens + 2)
        else:
            self.draft_tokens = max(1, self.draft_tokens - 1)
        if accepted:
            self._misses = 0
        else:
            self._misses += 1
            self._cooldown = min(2 ** (self._misses - 1), MAX_COOLDOWN_TICKS)
