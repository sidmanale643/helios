# Helios

Helios is a small, readable inference engine for running
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on a local NVIDIA GPU. It
implements the model, weight loading, tokenization, generation loop, KV cache,
and HTTP serving path directly in PyTorch.

The project is an inference-engineering learning environment. It favors code
that is easy to inspect and change over production features or maximum
throughput.

## What is included

- A native PyTorch implementation of Qwen3-4B
- Grouped-query attention, rotary position embeddings, RMS normalization, and
  SwiGLU feed-forward layers
- Hugging Face tokenizer and safetensor loading
- Autoregressive generation with a request-local KV cache
- Reuse of completed prompt blocks through a persistent prefix cache
- Memory checks before model loading and KV-cache sizing after loading
- A non-streaming OpenAI-style chat completions endpoint
- Repeatable benchmarks for latency, TTFT, inter-token latency, and throughput

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A CUDA-capable NVIDIA GPU with enough memory for Qwen3-4B and its KV cache
- Internet access on first run to download the model snapshot from Hugging Face

Apple Metal and CPU execution are not supported by the current loader.

## Quick start

```bash
git clone https://github.com/sidmanale643/helios.git
cd helios
uv sync
uv run helios
```

The first start downloads the tokenizer and model weights. If Hugging Face
requires authentication in your environment, set `HF_TOKEN` before starting
Helios.

Once the model is loaded, the server listens on `http://127.0.0.1:8000`.

```bash
curl http://127.0.0.1:8000/health
```

Interactive API documentation is available at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) while the server is
running.

## Chat completions API

Helios implements a focused, non-streaming subset of
`POST /v1/chat/completions`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  --header 'content-type: application/json' \
  --data '{
    "model": "Qwen/Qwen3-4B",
    "messages": [
      {"role": "system", "content": "Answer clearly and briefly."},
      {"role": "user", "content": "Why does a KV cache speed up decoding?"}
    ],
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 128,
    "stream": false
  }'
```

The response uses the OpenAI chat-completion shape and includes one assistant
choice plus prompt, completion, and total token counts.

Supported request fields:

| Field | Notes |
| --- | --- |
| `model` | Must be `Qwen/Qwen3-4B`. |
| `messages` | 1–128 `developer`, `system`, `user`, or `assistant` messages. |
| `max_tokens` | 1–2,048; defaults to 256. `max_completion_tokens` is accepted as an alias. |
| `temperature` | 0–2; defaults to 0.2. |
| `top_p` | Greater than 0 and at most 1; defaults to 0.95. |
| `stream` | Must be `false`. |

Streaming, tool calls, structured outputs, authentication, and the rest of the
OpenAI API are not implemented.

## Python usage

The same runtime can be called without HTTP:

```python
from helios.config import get_config
from helios.runtime.engine import Engine
from helios.runtime.frontend import TextGenerator
from helios.runtime.types import GenerateRequest
from helios.runtime.worker import Tokenizer

config = get_config()
generator = TextGenerator(Tokenizer.load(config), Engine(config))

text = generator.run(
    GenerateRequest(
        text="Explain speculative decoding in simple terms.",
        sampling={"temperature": 0.2, "max_new_tokens": 128},
    )
)
print(text)
```

For a small multi-prompt example:

```bash
uv run python run_batch.py
```

## How it works

```mermaid
flowchart LR
    Client --> FastAPI
    FastAPI --> Tokenizer
    Tokenizer --> Engine
    Engine --> PrefixCache[Prefix cache]
    Engine --> Qwen3[Qwen3 decoder]
    Qwen3 --> KVCache[Request KV cache]
    Qwen3 --> CUDA
    Engine --> Tokenizer
    FastAPI --> Client
```

At startup, Helios resolves one Hugging Face snapshot for both the tokenizer
and model, checks GPU memory, loads the safetensors into the native Qwen3
implementation, and reserves the remaining capacity for KV-cache tokens.

For each request, the tokenizer applies the Qwen3 chat template. The engine
looks up the longest complete cached prompt prefix, restores its per-layer K/V
state, prefills only the unmatched tokens, and then decodes one token at a time.
A process-local lock serializes generation because the decoder owns mutable
cache state.

## Configuration

Helios loads a local `.env` file automatically.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HELIOS_MODEL_ID` | `Qwen/Qwen3-4B` | Model repository. No other architecture is currently supported. |
| `HELIOS_MODEL_REVISION` | latest resolved snapshot | Pins model and tokenizer files to a Hugging Face revision. |
| `HF_TOKEN` / `HF_API_KEY` | unset | Hugging Face authentication. |
| `HELIOS_WEIGHT_HEADROOM_RATIO` | `0.20` | Additional free-memory requirement before weight loading. |
| `HELIOS_KV_CACHE_HEADROOM_RATIO` | `0.20` | Activation headroom retained when sizing the KV cache. |

Both headroom ratios must be at least `0` and less than `1`.

## Benchmarks

Run the fixed prompt suite and save a comparable result:

```bash
uv run python benchmarks/run.py --name decode-128
```

The benchmark warms every prompt before measuring it, so the current protocol
measures warm-prefix behavior. Results are written to `benchmarks/results/`
with the model revision, Git state, host, accelerator, workload, raw samples,
and aggregate metrics. Equivalent runs are compared automatically.

See [benchmarks/README.md](benchmarks/README.md) for the protocol and options.

## Project structure

```text
src/helios/
├── api/                 # FastAPI routes, schemas, and dependencies
├── runtime/
│   ├── qwen3/           # Model, layers, tokenizer, weights, decoding, and KV cache
│   ├── engine.py        # Serialized model execution
│   ├── frontend.py      # Text and chat frontend
│   ├── generate.py      # Prefill, decode, and prefix-cache orchestration
│   └── prefix_cache.py  # Hashed prompt blocks and cached K/V snapshots
├── config.py            # Environment-backed runtime configuration
└── main.py              # Local server entry point
benchmarks/              # Repeatable performance runner
run_batch.py             # Direct Python example
```

## Current scope

Helios deliberately keeps the serving design small. It does not currently
provide continuous batching, streaming, paged attention, quantization,
multi-model serving, distributed execution, optimized custom kernels, or a
bounded prefix-cache policy.

The goal is to make each inference mechanism understandable, verify it against
a simple baseline, and measure the effect before adding the next optimization.

## Acknowledgements

- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) for the model weights and tokenizer
- [LLMs from Scratch: Qwen3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) for a readable architecture reference
- [PyTorch](https://pytorch.org/) and [FastAPI](https://fastapi.tiangolo.com/)

## License

This repository does not currently include a license. The source is publicly
visible, but no permission to use, modify, or redistribute it is granted until
a license is added.
