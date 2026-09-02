# Fixed benchmark suite

Run the suite to load the model once and execute every workload in order:

```bash
uv run python benchmarks/run.py --label baseline
uv run python benchmarks/run.py --label kv-cache
uv run python benchmarks/run.py --label prefix-cache
uv run python benchmarks/run.py --label torch-compile
```

The label only names the saved result. It does not change the workload.

The suite loads the model once, sends every workload input to the model in order,
and saves each generated output with its metrics. All four workloads run together:

| Workload | Shape | What it isolates |
| --- | --- | --- |
| `prefill-long` | Long prompt, 8 output tokens | Attention and prompt-processing changes |
| `decode-long` | Short 2,000-word essay request, up to 2,048 output tokens | KV cache and token-by-token decode changes |
| `balanced` | Medium prompt, up to 64 output tokens | Typical mixed generation behavior |
| `agent-prefix` | Agent input followed by simulated tool call/result exchanges, 48 output tokens per step | Prefix reuse across an agent loop |

Only `agent-prefix` simulates a tool workflow: it appends an assistant tool call
and its returned tool result to the same transcript in order:
customer lookup, order listing, and order details. No tool is actually invoked.
The model and cache remain loaded for the full sequence.

The terminal table reports input/output token counts, end-to-end latency, time to
first token (TTFT), generation throughput, and restored-token cache hit rate for
every request. The JSON result additionally records the complete input and output
for each request, plus:

- Model revision, host, platform, and accelerator.

Results are written to `benchmarks/results/`.
