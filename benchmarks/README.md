# Fixed benchmark suite

Run this suite before and after every major inference change:

```bash
uv run python benchmarks/run.py --label baseline
uv run python benchmarks/run.py --label kv-cache
uv run python benchmarks/run.py --label prefix-cache
uv run python benchmarks/run.py --label torch-compile
```

The label describes the implementation being measured. It does not change the
workload. Keep the same model, model revision, machine, accelerator, environment,
and default `--runs 3` when comparing results.

The suite loads the model once, warms every request shape outside the measurement,
then runs four fixed workloads:

| Workload | Shape | What it isolates |
| --- | --- | --- |
| `prefill-long` | Long prompt, 8 output tokens | Attention and prompt-processing changes |
| `decode-long` | Short 2,000-word essay request, up to 2,048 output tokens | KV cache and token-by-token decode changes |
| `balanced` | Medium prompt, up to 64 output tokens | Typical mixed generation behavior |
| `agent-prefix` | Growing simulated tool-agent transcript, 48 output tokens per step | Prefix reuse across an agent loop |

Every uncached or cold repetition starts with a fresh prefix cache and fails if
Helios restores any tokens. Each `agent-prefix` repetition then appends simulated
assistant tool calls and simulated tool results to the same transcript in order:
customer lookup, order listing, and order details. No tool is actually invoked.
Every appended step must restore cached prefix tokens. The cache resets before the
next repetition so earlier runs cannot bias later samples.

The terminal table reports exact input/output token counts, end-to-end latency,
time to first token (TTFT), inter-token latency (ITL), generation throughput, and
restored-token cache hit rate for every workload/phase. The JSON result additionally
records:

- End-to-end, tokenization, detokenization, lookup, restore, prefill, decode, and
  cache-store timings.
- Prompt/output token counts, finish reason, restored and matched prefix tokens,
  stored block count, and request-level cache hits.
- Git commit and dirty state, model revision, Python and PyTorch versions, host,
  platform, and accelerator.
- The complete suite definition and its SHA-256 identity.

Results are written to `benchmarks/results/`. A run compares itself with the latest
result having the same suite hash and model revision. Same-machine and
same-accelerator results are preferred; cross-machine comparisons are labeled.

`--runs` exists for noisier machines, but it is part of the suite identity. A run
with a different repetition count establishes a separate baseline:

```bash
uv run python benchmarks/run.py --label baseline-long --runs 10
```

Do not edit prompts or output limits between an optimization's before and after
runs. If the fixed suite intentionally changes, increment `SUITE_VERSION` in
`benchmarks/suite.py`; the new hash will prevent comparison with old results.
