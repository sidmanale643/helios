from typing import TYPE_CHECKING

import torch

from helios.runtime.types import Sampling

if TYPE_CHECKING:
    from helios.runtime.qwen3.decode import Decoder


WARMUP_PROMPT = """
A regional library system has eight branches, a shared catalog, self-checkout kiosks, and a
mobile app. Patrons report that newly returned books sometimes remain unavailable for several
minutes, while staff occasionally see the same hold assigned twice during busy evenings. The
system uses one API, PostgreSQL, Redis, and a background worker. The team can make focused
changes but cannot replace these components. Recommend a staged reliability improvement that
includes data integrity, cache invalidation, observability, rollout safety, and user-facing error
handling. State the tradeoffs and define concrete metrics.
""".strip()

WARMUP_OUTPUT_TOKENS = 4


DECODE_WARMUP_STEPS = 4


def warm_decode(
    decoder: "Decoder",
    input_ids: list[int],
    *,
    max_batch_size: int,
    max_tokens: int,
    budget_tokens: int,
    prefill_chunk_size: int = 256,
) -> tuple[int, tuple[int, ...]]:
    sampling = Sampling(temperature=0, top_p=1, max_new_tokens=DECODE_WARMUP_STEPS)
    device = decoder.device
    batch_sizes = tuple(range(1, max_batch_size + 1))
    activation_peak = 0
    for measured in (False, True):
        for batch_size in batch_sizes:
            allocation_copies = 2 if decoder.page_pool is None and batch_size > 1 else 1
            capacity = min(
                max_tokens, budget_tokens // (batch_size * allocation_copies)
            )
            if decoder.page_pool is not None:
                page_size = decoder.page_pool.page_size
                pages = min(budget_tokens // page_size, decoder.page_pool.free_pages)
                capacity = min(capacity, pages // batch_size * page_size)
            prompt_limit = min(len(input_ids), capacity - DECODE_WARMUP_STEPS)
            if prompt_limit < 4:
                raise RuntimeError(
                    f"Decode warmup cannot fit batch size {batch_size} in the KV budget. "
                    "Reduce HELIOS_MAX_BATCH_SIZE."
                )
            for mixed in (False, True):
                caches = []
                result = None
                if measured and device.type == "cuda":
                    decoder._synchronize()
                    torch.cuda.empty_cache()
                    baseline = torch.cuda.memory_reserved(device)
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    packed_counts = [
                        min(
                            max(1, prefill_chunk_size - batch_size + 1)
                            if row == 0
                            else 1,
                            max(1, capacity - DECODE_WARMUP_STEPS - 4),
                        )
                        for row in range(batch_size)
                    ]
                    for row in range(batch_size):
                        length = (
                            prompt_limit - int(measured) - (row % 2 if mixed else 0)
                        )
                        if mixed and decoder.page_pool is not None:
                            length = min(
                                length,
                                capacity - DECODE_WARMUP_STEPS - packed_counts[row],
                            )
                        result = decoder.prefill(
                            input_ids[:length],
                            sampling,
                            max_total_tokens=max_tokens,
                        )
                        if decoder.page_pool is not None:
                            result.cache.capacity = capacity
                        caches.append(result.cache)
                        result = None
                    if mixed and decoder.page_pool is not None:
                        decoder.packed_caches(
                            caches, [[input_ids[-1]] * count for count in packed_counts]
                        )
                    for _ in range(DECODE_WARMUP_STEPS):
                        decoder.decode_caches(caches, [input_ids[-1]] * batch_size)
                    decoder._synchronize()
                    if measured and device.type == "cuda":
                        kv_bytes = sum(
                            cache.capacity * cache.memory_bytes_per_token
                            for cache in caches
                        )
                        if decoder.page_pool is not None:
                            kv_bytes = 0
                        peak = torch.cuda.max_memory_reserved(device) - baseline
                        activation_peak = max(activation_peak, peak - kv_bytes)
                finally:
                    result = None
                    for cache in caches:
                        decoder.release_cache(cache)
                    caches.clear()
                    cache = None
    return activation_peak, batch_sizes
