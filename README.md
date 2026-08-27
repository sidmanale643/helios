# Helios

Helios runs as two local processes separated by one ZeroMQ connection:

- `helios` owns FastAPI and CPU-only tokenization/detokenization.
- `helios-scheduler` owns the FIFO scheduler, model, KV cache, and GPU sampling.

Set `HF_TOKEN` (or `HF_API_KEY`) in the environment, then start Helios:

```bash
uv run helios
```

This one command starts the GPU scheduler, waits for the model to load, then
starts the HTTP frontend. Press `Ctrl-C` once to stop both processes.

`helios-scheduler` remains available when you deliberately want to run the
scheduler independently.

The ASGI application is exposed as `helios.main:app`. The legacy
`main:app` import path remains available for existing deployments.

The ASGI process loads only the tokenizer. The scheduler process loads the model
and binds to `tcp://127.0.0.1:5555` by default. Once both are ready, check them
through the HTTP frontend:

```bash
curl http://127.0.0.1:8000/health
```

Before loading weights, Helios selects CUDA, Apple Metal, or CPU, measures
currently available memory, and counts the configured model's parameters with
meta tensors. It reserves 20% headroom for loader buffers and refuses startup
when the estimated model weights cannot fit. After the weights load, Helios
profiles the remaining device memory, reserves further headroom, and derives a
single-request KV-cache token limit from the model's layers, KV heads, head
dimension, and dtype. The health response includes that limit; requests that
would exceed it are rejected before generation.

The frontend converts text to CPU token IDs and sends only those IDs, the EOS
token ID, and sampling settings to the scheduler. The scheduler returns only
generated token IDs, which the frontend detokenizes on the CPU. Model logits
never cross the ZeroMQ boundary.

Each generation gets an isolated fixed-capacity KV cache. Helios pre-fills it
once with the prompt, then runs each generated token against cached keys and
values rather than recomputing the full prompt. The cache stores only Qwen3's
8 KV heads per layer (not the expanded 32 query heads), and is released when
the request finishes.

Helios's primary model is the native Qwen3-4B architecture, with weights from
`Qwen/Qwen3-4B`. The native runtime separates the architecture, safetensor
weight mapping, tokenizer/chat template, and decoding under `runtime/qwen3/`.
Its architecture follows the [Qwen3 standalone reference](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb):
36 layers, 2,560 hidden width, 32 query heads, 8 KV heads, and a 9,728-wide
SwiGLU MLP. `HELIOS_MODEL_ID` must remain `Qwen/Qwen3-4B` until another native
architecture is implemented. Set `HELIOS_WEIGHT_HEADROOM_RATIO` and
`HELIOS_KV_CACHE_HEADROOM_RATIO` as needed; both headroom ratios default to
`0.20`.

Generate text with:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Explain solar power simply.","sampling":{"temperature":0.2,"max_new_tokens":128}}'
```

The HTTP layer is in `api/`. The CPU tokenizer worker, ZeroMQ protocol/client,
FIFO scheduler, model admission, and generation runtime are in `runtime/`. The
HTTP API exposes `POST /generate`; `prompt` is accepted as an input alias for
`text`. Flat sampling fields remain accepted.

The model and tokenizer processes must resolve the same Hugging Face snapshot.
Set `HELIOS_MODEL_REVISION` to pin one explicitly. `HELIOS_SCHEDULER_ENDPOINT`
changes the ZeroMQ endpoint, and `HELIOS_SCHEDULER_TIMEOUT_MS` changes the
generation response timeout.

## Run a batch from Python

With `helios-scheduler` running, the CPU frontend can also be used directly:

```python
from helios.config import get_config
from helios.runtime.client import SchedulerClient
from helios.runtime.frontend import TextGenerator
from helios.runtime.types import GenerateRequest
from helios.runtime.worker import Tokenizer

config = get_config()
generator = TextGenerator(Tokenizer.load(config), SchedulerClient(config))
prompts = ["Explain solar power simply.", "Explain photosynthesis simply."]
responses = [generator.run(GenerateRequest(text=prompt)) for prompt in prompts]
```

For per-prompt sampling settings, pass `GenerateRequest` objects instead of strings.

For a reusable editable batch, update the `prompts` list in `run_batch.py` and
run:

```bash
uv run python run_batch.py
```

## Benchmarks

Use the repeatable scheduler benchmark to track throughput after a code change
or across machines and accelerators:

```bash
uv run python benchmarks/run.py --name decode-128
```

Each invocation stores its full environment and result under `benchmarks/results/`
and reports improvement or worsening against the latest equivalent run. See
[`benchmarks/README.md`](benchmarks/README.md) for workload options and comparison rules.
