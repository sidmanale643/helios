COMPILE_WARMUP_PROMPT = """
A regional library system has eight branches, a shared catalog, self-checkout kiosks, and a
mobile app. Patrons report that newly returned books sometimes remain unavailable for several
minutes, while staff occasionally see the same hold assigned twice during busy evenings. The
system uses one API, PostgreSQL, Redis, and a background worker. The team can make focused
changes but cannot replace these components. Recommend a staged reliability improvement that
includes data integrity, cache invalidation, observability, rollout safety, and user-facing error
handling. State the tradeoffs and define concrete metrics.
""".strip()

COMPILE_WARMUP_OUTPUT_TOKENS = 4
