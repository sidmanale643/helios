# Fixed benchmark suite

Start Helios in one terminal. Model loading and the one-time compile warmup belong
to the server process:

```bash
HELIOS_TORCH_COMPILE=1 uv run helios
```

Then run the HTTP-only benchmark client in another terminal:

```bash
uv run python benchmarks/run.py --label baseline
uv run python benchmarks/run.py --label kv-cache
uv run python benchmarks/run.py --label prefix-cache
uv run python benchmarks/run.py --label torch-compile
```

The benchmark never loads, starts, stops, or owns the model. It only calls the
running server's `/health` and `/v1/chat/completions` endpoints. Use `--base-url`
or `HELIOS_BASE_URL` when Helios is not listening on `http://127.0.0.1:8000`.
The label only names the saved result; it does not change the workload.

The client sends every workload input to the server in order and saves each
generated output with its metrics. All four workloads run together:

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
