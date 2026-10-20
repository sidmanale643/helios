def propose_ngram(history: list[int], limit: int) -> list[int]:
    if limit <= 0:
        return []
    # Bound CPU lookup work even for long prompts.
    window = history[-4096:]
    for size in range(min(4, len(window) - 1), 0, -1):
        suffix = window[-size:]
        for start in range(len(window) - size - 1, -1, -1):
            if window[start : start + size] == suffix:
                return window[start + size : min(start + size + limit, len(window))]
    return []
