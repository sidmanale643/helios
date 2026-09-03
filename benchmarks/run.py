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


def duration(seconds: float) -> str:
    return f"{seconds * 1_000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"


def report(samples: list[dict[str, Any]], path: Path) -> str:
    lines = [
        "",
        "Helios benchmark results",
        "",
        f"{'request':<28} {'in':>6} {'out':>6} {'E2E':>9} {'TTFT':>9} {'tok/s':>10} {'cache':>7}",
        "-" * 85,
    ]
    for sample in samples:
        metrics = sample["metrics"]
        name = sample["workload"]
        if sample["phase"] not in {"uncached", "cold"}:
            name = f"{name}/{sample['phase']}"
        rate = metrics["generation_tokens_per_second"]
        rate_display = f"{rate:.2f}" if rate is not None else "—"
        lines.append(
            f"{name:<28} {metrics['prompt_tokens']:>6} {metrics['output_tokens']:>6} "
            f"{duration(metrics['end_to_end_seconds']):>9} "
            f"{duration(metrics['time_to_first_token_seconds']):>9} "
            f"{rate_display:>10} {metrics['cache_hit_rate'] * 100:>6.0f}%"
        )
    lines.extend(["", f"Saved outputs and metrics: {path.relative_to(ROOT)}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    health = request_json(args.base_url, "/health", timeout=args.timeout)
    model = health["model"]

    samples = []
    for workload in WORKLOADS:
        for request in requests_for(workload):
            samples.append(
                run_request(
                    args.base_url,
                    model,
                    request,
                    timeout=args.timeout,
                )
            )

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
    print(report(samples, path))


if __name__ == "__main__":
    main()
