import argparse
import json
import os
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from suite import SUITE_VERSION, WORKLOADS, RequestSpec, requests_for

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every benchmark workload against an existing Helios server."
    )
    parser.add_argument("--label", default="run", help="Name for the saved result.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("HELIOS_BASE_URL", "http://127.0.0.1:8000"),
        help="URL of an already-running Helios server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1_800,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run every benchmark request in one model-side static batch.",
    )
    return parser.parse_args()


def request_json(
    base_url: str,
    path: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"Helios returned HTTP {error.code} for {path}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Cannot reach Helios at {base_url}. Start it first with `uv run helios`."
        ) from error


def run_request(
    base_url: str,
    model: str,
    spec: RequestSpec,
    *,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = request_json(
        base_url,
        "/v1/chat/completions",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {"role": role, "content": content}
                for role, content in spec.messages
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": spec.workload.max_new_tokens,
            "stream": False,
        },
    )
    end_to_end_seconds = time.perf_counter() - started
    usage = response["usage"]
    timings = response["timings"]
    generation_seconds = timings["prefill_seconds"] + timings["decode_seconds"]
    prompt_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]

    return {
        "workload": spec.workload.name,
        "phase": spec.phase,
        "input": spec.messages,
        "output": response["choices"][0]["message"]["content"],
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "end_to_end_seconds": end_to_end_seconds,
            "time_to_first_token_seconds": (
                timings["tokenize_seconds"]
                + timings["queue_seconds"]
                + timings["prefix_lookup_seconds"]
                + timings["restore_seconds"]
                + timings["prefill_seconds"]
            ),
            "generation_tokens_per_second": (
                output_tokens / generation_seconds
                if generation_seconds > 0
                else None
            ),
            "restored_tokens": usage["prompt_tokens_details"]["cached_tokens"],
            "cache_hit_rate": (
                usage["prompt_tokens_details"]["cached_tokens"] / prompt_tokens
            ),
        },
    }


def run_batch(
    base_url: str,
    model: str,
    specs: list[RequestSpec],
    *,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    response = request_json(
        base_url,
        "/v1/chat/completions/batch",
        timeout=timeout,
        payload={
            "requests": [
                {
                    "model": model,
                    "messages": [
                        {"role": role, "content": content}
                        for role, content in spec.messages
                    ],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": spec.workload.max_new_tokens,
                    "stream": False,
                }
                for spec in specs
            ]
        },
    )
    batch_seconds = time.perf_counter() - started
    items = sorted(response["items"], key=lambda item: item["index"])
    samples = [
        {
            "workload": spec.workload.name,
            "phase": spec.phase,
            "input": spec.messages,
            "output": item["content"],
            "metrics": {
                "prompt_tokens": item["prompt_tokens"],
                "output_tokens": item["completion_tokens"],
                "end_to_end_seconds": batch_seconds,
                "time_to_first_token_seconds": None,
                "generation_tokens_per_second": None,
                "restored_tokens": 0,
                "cache_hit_rate": 0,
            },
        }
        for spec, item in zip(specs, items, strict=True)
    ]
    total_output_tokens = sum(item["completion_tokens"] for item in items)
    return samples, {
        "batch_size": len(specs),
        "end_to_end_seconds": batch_seconds,
        "output_tokens": total_output_tokens,
        "output_tokens_per_second": total_output_tokens / batch_seconds,
    }


def duration(seconds: float) -> str:
    return f"{seconds * 1_000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"


def request_name(sample: dict[str, Any]) -> str:
    name = sample["workload"]
    if sample["phase"] not in {"uncached", "cold"}:
        name = f"{name}/{sample['phase']}"
    return name


def print_response(index: int, total: int, sample: dict[str, Any]) -> None:
    print(
        f"\n[{index}/{total}] {request_name(sample)} response:\n"
        f"{sample['output']}",
        flush=True,
    )


def report(
    samples: list[dict[str, Any]],
    path: Path,
    batch_metrics: dict[str, Any] | None = None,
) -> str:
    lines = [
        "",
        "Helios benchmark results",
        "",
        f"{'request':<28} {'in':>6} {'out':>6} {'E2E':>9} {'TTFT':>9} {'tok/s':>10} {'cache':>7}",
        "-" * 85,
    ]
    for sample in samples:
        metrics = sample["metrics"]
        name = request_name(sample)
        rate = metrics["generation_tokens_per_second"]
        rate_display = f"{rate:.2f}" if rate is not None else "—"
        ttft = metrics["time_to_first_token_seconds"]
        ttft_display = duration(ttft) if ttft is not None else "—"
        lines.append(
            f"{name:<28} {metrics['prompt_tokens']:>6} {metrics['output_tokens']:>6} "
            f"{duration(metrics['end_to_end_seconds']):>9} "
            f"{ttft_display:>9} "
            f"{rate_display:>10} {metrics['cache_hit_rate'] * 100:>6.0f}%"
        )
    if batch_metrics is not None:
        lines.extend(
            [
                "",
                "Static batch metrics",
                f"Batch size: {batch_metrics['batch_size']}",
                f"Output tokens: {batch_metrics['output_tokens']}",
                f"End-to-end: {duration(batch_metrics['end_to_end_seconds'])}",
                (
                    "Output throughput: "
                    f"{batch_metrics['output_tokens_per_second']:.2f} tok/s"
                ),
            ]
        )
    lines.extend(["", f"Saved outputs and metrics: {path.relative_to(ROOT)}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    print(f"Checking Helios at {args.base_url} ...", flush=True)
    health = request_json(args.base_url, "/health", timeout=args.timeout)
    model = health["model"]
    print(f"Model loaded: {model}", flush=True)

    requests = [
        request
        for workload in WORKLOADS
        for request in requests_for(workload)
    ]

    if args.batch:
        print(
            f"Running one batch of {len(requests)} requests ...",
            flush=True,
        )
        samples, batch_metrics = run_batch(
            args.base_url, model, requests, timeout=args.timeout
        )
        for index, sample in enumerate(samples, start=1):
            print_response(index, len(samples), sample)
    else:
        samples = []
        for index, request in enumerate(requests, start=1):
            print(
                f"\n[{index}/{len(requests)}] Running "
                f"{request.workload.name}/{request.phase} ...",
                flush=True,
            )
            sample = run_request(
                args.base_url, model, request, timeout=args.timeout
            )
            samples.append(sample)
            print_response(index, len(requests), sample)
        batch_metrics = None

    now = datetime.now(UTC)
    record = {
        "schema_version": 4,
        "timestamp": now.isoformat(),
        "label": args.label,
        "machine": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "server": args.base_url,
        "accelerator": {"kind": "cuda", "name": health["memory"]["gpu"]},
        "model": {"id": health["model"], "revision": health["model_revision"]},
        "suite_version": SUITE_VERSION,
        "execution_mode": "static-batch" if args.batch else "sequential",
        "batch_metrics": batch_metrics,
        "torch_compile": health["torch_compile"],
        "samples": samples,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in args.label
    )
    path = RESULTS / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{safe_label}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(report(samples, path, batch_metrics))


if __name__ == "__main__":
    main()
