import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from suite import WARMUP_REQUEST


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
MESSAGE_ROLES = {"developer", "system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    category: str
    messages: tuple[tuple[str, str], ...]
    max_tokens: int
    temperature: float
    top_p: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic mixed-workload benchmark against an existing "
            "Helios server."
        )
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
        "--dataset",
        type=Path,
        default=ROOT / "dataset.json",
        help="Versioned request dataset to run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum simultaneous HTTP requests (default: 8).",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help=(
            "Run only this dataset category. Repeat the flag to select multiple "
            "categories, e.g. --category decode_heavy --category balanced."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle request order deterministically before submission.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by --shuffle (default: 42).",
    )
    parser.add_argument(
        "--print-outputs",
        action="store_true",
        help="Print full model outputs. Disabled by default to reduce client overhead.",
    )
    return parser.parse_args()


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def load_dataset(path: Path) -> tuple[int, list[BenchmarkRequest]]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Dataset does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Dataset is not valid JSON: {path}: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("schema_version"), int):
        raise TypeError("Dataset must be an object with an integer schema_version.")

    rows = data.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset requests must be a non-empty array.")

    requests: list[BenchmarkRequest] = []
    request_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Dataset request {index} must be an object.")

        request_id = row.get("id")
        category = row.get("category")
        messages = row.get("messages")
        max_tokens = row.get("max_tokens")
        temperature = row.get("temperature")
        top_p = row.get("top_p")

        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"Dataset request {index} needs a non-empty id.")
        if request_id in request_ids:
            raise ValueError(f"Dataset request id is duplicated: {request_id}")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Dataset request {request_id} needs a category.")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Dataset request {request_id} needs messages.")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 2_048
        ):
            raise ValueError(f"Dataset request {request_id} has invalid max_tokens.")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError(f"Dataset request {request_id} has invalid temperature.")
        if (
            not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError(f"Dataset request {request_id} has invalid top_p.")

        normalized_messages: list[tuple[str, str]] = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in MESSAGE_ROLES
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise ValueError(
                    f"Dataset request {request_id} has an invalid message."
                )
            normalized_messages.append((message["role"], message["content"]))

        request_ids.add(request_id)
        requests.append(
            BenchmarkRequest(
                request_id=request_id,
                category=category,
                messages=tuple(normalized_messages),
                max_tokens=max_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        )

    return data["schema_version"], requests


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
    spec: BenchmarkRequest,
    *,
    timeout: float,
    batch_started: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    start_offset = None if batch_started is None else started - batch_started

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
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": spec.max_tokens,
            "stream": False,
        },
    )

    finished = time.perf_counter()
    end_to_end_seconds = finished - started
    finish_offset = None if batch_started is None else finished - batch_started

    usage = response["usage"]
    timings = response["timings"]
    prompt_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]
    choice = response["choices"][0]
    finish_reason = choice.get("finish_reason")

    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens", 0) or 0

    return {
        "id": spec.request_id,
        "category": spec.category,
        "input": spec.messages,
        "output": choice["message"]["content"],
        "finish_reason": finish_reason,
        "request": {
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
        },
        "client_timeline": {
            "start_offset_seconds": start_offset,
            "finish_offset_seconds": finish_offset,
        },
        "timings": timings,
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "max_tokens": spec.max_tokens,
            "completion_ratio": output_tokens / spec.max_tokens,
            "hit_max_tokens": output_tokens >= spec.max_tokens,
            "end_to_end_seconds": end_to_end_seconds,
            "time_to_first_token_seconds": timings.get(
                "time_to_first_token_seconds"
            ),
            "generation_tokens_per_second": timings.get(
                "generation_tokens_per_second"
            ),
            "prefill_tokens_per_second": timings.get("prefill_tokens_per_second"),
            "decode_tokens_per_second": timings.get("decode_tokens_per_second"),
            "decode_compute_tokens_per_second": timings.get("decode_compute_tokens_per_second"),
            "restored_tokens": cached_tokens,
            "cache_hit_rate": timings.get("cache_hit_rate", 0.0),
        },
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_metric(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample["metrics"].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                values.append(float(value))
    return values


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = numeric_metric(samples, "time_to_first_token_seconds")
    e2e = numeric_metric(samples, "end_to_end_seconds")
    decode_rate = numeric_metric(samples, "decode_tokens_per_second")
    decode_compute_rate = numeric_metric(
        samples, "decode_compute_tokens_per_second"
    )
    prefill_rate = numeric_metric(samples, "prefill_tokens_per_second")

    total_max_tokens = sum(sample["metrics"]["max_tokens"] for sample in samples)
    total_output_tokens = sum(sample["metrics"]["output_tokens"] for sample in samples)
    hit_cap = sum(bool(sample["metrics"]["hit_max_tokens"]) for sample in samples)

    return {
        "request_count": len(samples),
        "prompt_tokens": sum(
            sample["metrics"]["prompt_tokens"] for sample in samples
        ),
        "output_tokens": total_output_tokens,
        "requested_max_tokens": total_max_tokens,
        "aggregate_completion_ratio": (
            total_output_tokens / total_max_tokens if total_max_tokens else 0.0
        ),
        "requests_hitting_max_tokens": hit_cap,
        "requests_hitting_max_tokens_rate": hit_cap / len(samples),
        "finish_reasons": dict(Counter(sample["finish_reason"] for sample in samples)),
        "ttft_seconds": {
            "p50": percentile(ttft, 0.50),
            "p95": percentile(ttft, 0.95),
        },
        "e2e_seconds": {
            "p50": percentile(e2e, 0.50),
            "p95": percentile(e2e, 0.95),
        },
        "decode_tokens_per_second": {
            "p50": percentile(decode_rate, 0.50),
            "p95": percentile(decode_rate, 0.95),
        },
        "decode_compute_tokens_per_second": {
            "p50": percentile(decode_compute_rate, 0.50),
            "p95": percentile(decode_compute_rate, 0.95),
        },
        "prefill_tokens_per_second": {
            "p50": percentile(prefill_rate, 0.50),
            "p95": percentile(prefill_rate, 0.95),
        },
    }


def peak_client_concurrency(samples: list[dict[str, Any]]) -> int:
    events: list[tuple[float, int]] = []

    for sample in samples:
        timeline = sample["client_timeline"]
        start = timeline["start_offset_seconds"]
        finish = timeline["finish_offset_seconds"]
        if start is None or finish is None:
            continue
        events.append((float(start), 1))
        events.append((float(finish), -1))

    active = 0
    peak = 0
    # End events (-1) sort before start events (+1) at identical timestamps.
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)

    return peak


def run_continuous_batch(
    base_url: str,
    model: str,
    specs: list[BenchmarkRequest],
    *,
    timeout: float,
    concurrency: int,
    print_outputs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1.")

    batch_started = time.perf_counter()
    samples_by_id: dict[str, dict[str, Any]] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=min(concurrency, len(specs))) as executor:
        futures = [
            executor.submit(
                run_request,
                base_url,
                model,
                spec,
                timeout=timeout,
                batch_started=batch_started,
            )
            for spec in specs
        ]

        for future in as_completed(futures):
            sample = future.result()
            completed += 1
            print_response(
                completed,
                len(specs),
                sample,
                print_output=print_outputs,
            )
            samples_by_id[sample["id"]] = sample

    elapsed_seconds = time.perf_counter() - batch_started
    samples = [samples_by_id[spec.request_id] for spec in specs]

    total_output_tokens = sum(
        sample["metrics"]["output_tokens"] for sample in samples
    )
    total_prompt_tokens = sum(
        sample["metrics"]["prompt_tokens"] for sample in samples
    )
    total_request_time = sum(
        sample["metrics"]["end_to_end_seconds"] for sample in samples
    )

    start_offsets = [
        sample["client_timeline"]["start_offset_seconds"]
        for sample in samples
        if sample["client_timeline"]["start_offset_seconds"] is not None
    ]
    last_request_start = max(start_offsets, default=0.0)
    drain_seconds = max(0.0, elapsed_seconds - last_request_start)

    aggregate = summarize_samples(samples)

    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({sample["category"] for sample in samples}):
        category_samples = [
            sample for sample in samples if sample["category"] == category
        ]
        categories[category] = summarize_samples(category_samples)

    return samples, {
        "request_count": len(specs),
        "configured_concurrency": concurrency,
        "effective_concurrency": min(concurrency, len(specs)),
        "peak_client_concurrency": peak_client_concurrency(samples),
        "average_client_concurrency": (
            total_request_time / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "elapsed_seconds": elapsed_seconds,
        "last_request_start_seconds": last_request_start,
        "drain_seconds": drain_seconds,
        "drain_fraction": drain_seconds / elapsed_seconds if elapsed_seconds else 0.0,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "output_tokens_per_second": (
            total_output_tokens / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "requests_per_second": len(specs) / elapsed_seconds if elapsed_seconds else 0.0,
        "aggregate": aggregate,
        "categories": categories,
    }


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds * 1_000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"


def rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def request_name(sample: dict[str, Any]) -> str:
    return f"{sample['category']}/{sample['id']}"


def print_response(
    index: int,
    total: int,
    sample: dict[str, Any],
    *,
    print_output: bool,
) -> None:
    metrics = sample["metrics"]
    print(
        f"[{index:>2}/{total}] {request_name(sample):<30} "
        f"in={metrics['prompt_tokens']:>5} "
        f"out={metrics['output_tokens']:>4}/{metrics['max_tokens']:<4} "
        f"E2E={duration(metrics['end_to_end_seconds']):>8} "
        f"TTFT={duration(metrics['time_to_first_token_seconds']):>8} "
        f"dec={rate(metrics['decode_tokens_per_second']):>8} "
        f"finish={sample['finish_reason']}",
        flush=True,
    )

    if print_output:
        print(sample["output"], flush=True)


def report(
    samples: list[dict[str, Any]],
    path: Path,
    warmup: dict[str, Any],
    continuous_batch_metrics: dict[str, Any],
) -> str:
    lines = [
        "",
        "Helios benchmark results",
        "",
        "Post-health warmup",
        f"End-to-end: {duration(warmup['metrics']['end_to_end_seconds'])}",
        f"Server total: {duration(warmup['timings'].get('total_seconds'))}",
        f"TTFT: {duration(warmup['timings'].get('time_to_first_token_seconds'))}",
        "",
        (
            f"{'request':<30} {'in':>6} {'out':>9} {'E2E':>9} {'TTFT':>9} "
            f"{'pre tok/s':>10} {'dec tok/s':>10} {'cache':>7} {'finish':>9}"
        ),
        "-" * 124,
    ]

    for sample in samples:
        metrics = sample["metrics"]
        name = request_name(sample)
        out_display = f"{metrics['output_tokens']}/{metrics['max_tokens']}"
        lines.append(
            f"{name:<30} "
            f"{metrics['prompt_tokens']:>6} "
            f"{out_display:>9} "
            f"{duration(metrics['end_to_end_seconds']):>9} "
            f"{duration(metrics['time_to_first_token_seconds']):>9} "
            f"{rate(metrics['prefill_tokens_per_second']):>10} "
            f"{rate(metrics['decode_tokens_per_second']):>10} "
            f"{metrics['cache_hit_rate'] * 100:>6.0f}% "
            f"{str(sample['finish_reason']):>9}"
        )

    aggregate = continuous_batch_metrics["aggregate"]

    lines.extend(
        [
            "",
            "Continuous-batch metrics",
            f"Request count: {continuous_batch_metrics['request_count']}",
            (
                "Client concurrency: "
                f"{continuous_batch_metrics['effective_concurrency']} "
                f"(peak {continuous_batch_metrics['peak_client_concurrency']}, "
                f"avg {continuous_batch_metrics['average_client_concurrency']:.2f})"
            ),
            f"Prompt tokens: {continuous_batch_metrics['prompt_tokens']}",
            f"Output tokens: {continuous_batch_metrics['output_tokens']}",
            f"Elapsed: {duration(continuous_batch_metrics['elapsed_seconds'])}",
            (
                "Output throughput: "
                f"{continuous_batch_metrics['output_tokens_per_second']:.2f} tok/s"
            ),
            (
                "Request throughput: "
                f"{continuous_batch_metrics['requests_per_second']:.3f} req/s"
            ),
            (
                "Tail drain: "
                f"{duration(continuous_batch_metrics['drain_seconds'])} "
                f"({continuous_batch_metrics['drain_fraction'] * 100:.1f}% of run)"
            ),
            (
                "Requests hitting max_tokens: "
                f"{aggregate['requests_hitting_max_tokens']}/"
                f"{aggregate['request_count']} "
                f"({aggregate['requests_hitting_max_tokens_rate'] * 100:.1f}%)"
            ),
            (
                "Generated/requested token ratio: "
                f"{aggregate['aggregate_completion_ratio'] * 100:.1f}%"
            ),
            (
                "TTFT p50 / p95: "
                f"{duration(aggregate['ttft_seconds']['p50'])} / "
                f"{duration(aggregate['ttft_seconds']['p95'])}"
            ),
            (
                "E2E p50 / p95: "
                f"{duration(aggregate['e2e_seconds']['p50'])} / "
                f"{duration(aggregate['e2e_seconds']['p95'])}"
            ),
            (
                "Per-request wall-clock decode tok/s p50 / p95: "
                f"{rate(aggregate['decode_tokens_per_second']['p50'])} / "
                f"{rate(aggregate['decode_tokens_per_second']['p95'])}"
            ),
            (
                "Per-request compute-only decode tok/s p50 / p95: "
                f"{rate(aggregate['decode_compute_tokens_per_second']['p50'])} / "
                f"{rate(aggregate['decode_compute_tokens_per_second']['p95'])}"
            ),
            "",
            "Category summary",
            (
                f"{'category':<18} {'reqs':>5} {'out':>7} {'cap':>7} "
                f"{'TTFT p50':>10} {'TTFT p95':>10} {'wall p50':>9} "
                f"{'compute p50':>12}"
            ),
            "-" * 98,
        ]
    )

    for category, summary in continuous_batch_metrics["categories"].items():
        lines.append(
            f"{category:<18} "
            f"{summary['request_count']:>5} "
            f"{summary['output_tokens']:>7} "
            f"{summary['requests_hitting_max_tokens_rate'] * 100:>6.1f}% "
            f"{duration(summary['ttft_seconds']['p50']):>10} "
            f"{duration(summary['ttft_seconds']['p95']):>10} "
            f"{rate(summary['decode_tokens_per_second']['p50']):>9} "
            f"{rate(summary['decode_compute_tokens_per_second']['p50']):>12}"
        )

    try:
        result_path = path.relative_to(ROOT)
    except ValueError:
        result_path = path

    lines.extend(["", f"Saved outputs and metrics: {result_path}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")

    dataset_version, requests = load_dataset(args.dataset)

    if args.category:
        wanted = set(args.category)
        known = {request.category for request in requests}
        unknown = wanted - known
        if unknown:
            raise ValueError(
                f"Unknown categories: {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(known))}"
            )
        requests = [
            request for request in requests if request.category in wanted
        ]

    if not requests:
        raise ValueError("No requests selected.")

    if args.shuffle:
        random.Random(args.seed).shuffle(requests)

    print(f"Checking Helios at {args.base_url} ...", flush=True)
    health = request_json(args.base_url, "/health", timeout=args.timeout)
    model = health["model"]
    print(f"Model loaded: {model}", flush=True)

    print("Running post-health warmup ...", flush=True)
    warmup = run_request(
        args.base_url,
        model,
        BenchmarkRequest(
            request_id="warmup",
            category="warmup",
            messages=WARMUP_REQUEST.messages,
            max_tokens=WARMUP_REQUEST.workload.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        ),
        timeout=args.timeout,
    )

    selected_categories = sorted({request.category for request in requests})
    order = "shuffled" if args.shuffle else "dataset"
    print(
        f"Running {len(requests)} requests at client concurrency "
        f"{min(args.concurrency, len(requests))} "
        f"(order={order}, categories={','.join(selected_categories)}) ...",
        flush=True,
    )

    samples, continuous_batch_metrics = run_continuous_batch(
        args.base_url,
        model,
        requests,
        timeout=args.timeout,
        concurrency=args.concurrency,
        print_outputs=args.print_outputs,
    )

    now = datetime.now(UTC)
    record = {
        "schema_version": 7,
        "timestamp": now.isoformat(),
        "label": args.label,
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "git": {"revision": git_revision()},
        "server": args.base_url,
        "accelerator": {
            "kind": "cuda",
            "name": health.get("memory", {}).get("gpu"),
        },
        "model": {
            "id": health["model"],
            "revision": health.get("model_revision"),
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": dataset_sha256(args.dataset),
            "schema_version": dataset_version,
            "request_count": len(requests),
            "categories": selected_categories,
            "order": order,
            "shuffle_seed": args.seed if args.shuffle else None,
            "request_order": [request.request_id for request in requests],
        },
        "execution_mode": "continuous-batch",
        "continuous_batch_metrics": continuous_batch_metrics,
        "warmup": warmup,
        "samples": samples,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in args.label
    )
    path = RESULTS / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{safe_label}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")

    print(report(samples, path, warmup, continuous_batch_metrics))


if __name__ == "__main__":
    main()
