import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import torch
from suite import SUITE_VERSION, WORKLOADS, RequestSpec, requests_for

from helios.config import get_config
from helios.runtime.engine import Engine
from helios.runtime.frontend import TextGenerator
from helios.runtime.types import Sampling
from helios.runtime.worker import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed Helios benchmark suite."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Change being measured, such as baseline, kv-cache, or torch-compile.",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Measured repetitions per workload."
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    args.label = args.label.strip()
    if not args.label:
        parser.error("--label must not be empty")
    return args


def git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def accelerator() -> dict[str, str]:
    if torch.cuda.is_available():
        return {
            "kind": "cuda",
            "name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        }
    if torch.backends.mps.is_available():
        return {
            "kind": "mps",
            "name": "Apple Metal",
            "torch_version": torch.__version__,
        }
    return {
        "kind": "cpu",
        "name": platform.processor() or "unknown",
        "torch_version": torch.__version__,
    }


def suite_definition(runs: int) -> dict[str, object]:
    return {
        "version": SUITE_VERSION,
        "runs": runs,
        "workloads": [
            {
                "name": workload.name,
                "kind": workload.kind,
                "description": workload.description,
                "max_new_tokens": workload.max_new_tokens,
                "request_shapes": [
                    {"phase": request.phase, "messages": request.messages}
                    for request in requests_for(workload)
                ],
            }
            for workload in WORKLOADS
        ],
    }


def suite_hash(definition: dict[str, object]) -> str:
    payload = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_results() -> list[dict[str, Any]]:
    records = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"Skipping invalid result: {path}", file=sys.stderr)
            continue
        if isinstance(record, dict) and record.get("schema_version") == 2:
            records.append(record)
    return records


def previous_result(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    matches = [
        item
        for item in load_results()
        if item.get("suite", {}).get("sha256") == record["suite"]["sha256"]
        and item.get("model") == record["model"]
    ]
    if not matches:
        return None, None
    machine_matches = [
        item
        for item in matches
        if item.get("machine") == record["machine"]
        and item.get("accelerator") == record["accelerator"]
    ]
    if machine_matches:
        return machine_matches[-1], "same machine and accelerator"
    return matches[-1], "different machine or accelerator"


def run_request(generator: TextGenerator, spec: RequestSpec) -> dict[str, object]:
    sampling = Sampling(
        temperature=0,
        top_p=1,
        max_new_tokens=spec.workload.max_new_tokens,
    )
    started = time.perf_counter()
    tokenize_started = time.perf_counter()
    input_ids = generator.tokenizer.tokenize_chat(list(spec.messages))
    tokenize_seconds = time.perf_counter() - tokenize_started
    result = generator.engine.run(
        input_ids,
        generator.tokenizer.eos_token_id,
        sampling,
    )
    detokenize_started = time.perf_counter()
    generator.tokenizer.detokenize(result.output_ids)
    detokenize_seconds = time.perf_counter() - detokenize_started
    end_to_end = time.perf_counter() - started
    decode_seconds = sum(result.inter_token_seconds)
    generation_seconds = result.prefill_seconds + decode_seconds
    time_to_first_token = (
        tokenize_seconds
        + result.queue_seconds
        + result.prefix_lookup_seconds
        + result.restore_seconds
        + result.prefill_seconds
    )
    return {
        "workload": spec.workload.name,
        "phase": spec.phase,
        "prompt_tokens": len(input_ids),
        "output_tokens": len(result.output_ids),
        "finish_reason": result.finish_reason,
        "output_sha256": hashlib.sha256(
            json.dumps(result.output_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "end_to_end_seconds": end_to_end,
        "time_to_first_token_seconds": time_to_first_token,
        "prefill_seconds": result.prefill_seconds,
        "decode_seconds": decode_seconds,
        "inter_token_seconds": result.inter_token_seconds,
        "tokenize_seconds": tokenize_seconds,
        "detokenize_seconds": detokenize_seconds,
        "prefix_lookup_seconds": result.prefix_lookup_seconds,
        "restore_seconds": result.restore_seconds,
        "store_seconds": result.store_seconds,
        "prefix_hit_tokens": result.prefix.hit_tokens,
        "restored_tokens": result.prefix.restored_tokens,
        "stored_blocks": result.prefix.stored_blocks,
        "cache_hit_rate": result.prefix.restored_tokens / len(input_ids),
        "generation_tokens_per_second": (
            len(result.output_ids) / generation_seconds
            if generation_seconds > 0
            else None
        ),
        "end_to_end_tokens_per_second": (
            len(result.output_ids) / end_to_end if end_to_end > 0 else None
        ),
    }


def reset_prefix_cache(generator: TextGenerator) -> None:
    generator.engine.generator.prefix_cache.clear()


def validate_samples(samples: list[dict[str, Any]]) -> None:
    for workload in WORKLOADS:
        phases = {
            sample["phase"] for sample in samples if sample["workload"] == workload.name
        }
        for phase in phases:
            matching = [
                sample
                for sample in samples
                if sample["workload"] == workload.name and sample["phase"] == phase
            ]
            if any(sample["output_tokens"] == 0 for sample in matching):
                raise RuntimeError(
                    f"{workload.name}/{phase} produced no output tokens."
                )
            outputs = {
                (
                    sample["output_tokens"],
                    sample["finish_reason"],
                    sample["output_sha256"],
                )
                for sample in matching
            }
            if len(outputs) != 1:
                raise RuntimeError(
                    f"{workload.name}/{phase} produced different deterministic outputs across runs."
                )


def outputs_match(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    fields = ("workload", "phase", "output_tokens", "finish_reason", "output_sha256")
    current_outputs = [
        tuple(sample.get(field) for field in fields) for sample in current["samples"]
    ]
    previous_outputs = [
        tuple(sample.get(field) for field in fields) for sample in previous["samples"]
    ]
    return current_outputs == previous_outputs


def aggregate(samples: list[dict[str, Any]]) -> dict[str, object]:
    inter_token = [
        seconds for sample in samples for seconds in sample["inter_token_seconds"]
    ]
    generation_rates = [
        sample["generation_tokens_per_second"]
        for sample in samples
        if sample["generation_tokens_per_second"] is not None
    ]
    end_to_end_rates = [sample["end_to_end_tokens_per_second"] for sample in samples]
    return {
        "requests": len(samples),
        "median_prompt_tokens": median(sample["prompt_tokens"] for sample in samples),
        "median_output_tokens": median(sample["output_tokens"] for sample in samples),
        "median_end_to_end_seconds": median(
            sample["end_to_end_seconds"] for sample in samples
        ),
        "median_time_to_first_token_seconds": median(
            sample["time_to_first_token_seconds"] for sample in samples
        ),
        "median_prefill_seconds": median(
            sample["prefill_seconds"] for sample in samples
        ),
        "median_inter_token_latency_seconds": median(inter_token)
        if inter_token
        else None,
        "generation_tokens_per_second": median(generation_rates)
        if generation_rates
        else None,
        "end_to_end_tokens_per_second": median(end_to_end_rates),
        "median_restored_tokens": median(
            sample["restored_tokens"] for sample in samples
        ),
        "median_cache_hit_rate": median(sample["cache_hit_rate"] for sample in samples),
        "request_cache_hit_rate": sum(
            sample["restored_tokens"] > 0 for sample in samples
        )
        / len(samples),
    }


def aggregate_by_workload(
    samples: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    metrics = {}
    for workload in WORKLOADS:
        phases = [request.phase for request in requests_for(workload)]
        metrics[workload.name] = {
            phase: aggregate(
                [
                    sample
                    for sample in samples
                    if sample["workload"] == workload.name and sample["phase"] == phase
                ]
            )
            for phase in phases
        }
    return metrics


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def percentage(value: float) -> str:
    return f"{value * 100:.0f}%"


def report(
    record: dict[str, Any],
    path: Path,
    previous: dict[str, Any] | None,
    scope: str | None,
) -> str:
    lines = [
        "",
        "Helios fixed benchmark suite",
        f"label={record['label']}  suite=v{record['suite']['version']}  runs={record['suite']['runs']}",
        f"model={record['model']['id']}@{record['model']['revision'][:12]}",
        "",
        (
            f"{'workload':<28} {'in':>6} {'out':>6} {'E2E':>9} {'TTFT':>9} "
            f"{'ITL':>9} {'gen tok/s':>10} {'cache':>7}"
        ),
        "-" * 99,
    ]
    for workload in WORKLOADS:
        for phase, metrics in record["metrics"][workload.name].items():
            name = workload.name if phase == "uncached" else f"{workload.name}/{phase}"
            lines.append(
                f"{name:<28} "
                f"{metrics['median_prompt_tokens']:>6.0f} "
                f"{metrics['median_output_tokens']:>6.0f} "
                f"{duration(metrics['median_end_to_end_seconds']):>9} "
                f"{duration(metrics['median_time_to_first_token_seconds']):>9} "
                f"{duration(metrics['median_inter_token_latency_seconds']):>9} "
                f"{metrics['generation_tokens_per_second']:>10.2f} "
                f"{percentage(metrics['median_cache_hit_rate']):>7}"
            )
    agent_requests = requests_for(
        next(workload for workload in WORKLOADS if workload.name == "agent-prefix")
    )
    cold = record["metrics"]["agent-prefix"][agent_requests[0].phase]
    final = record["metrics"]["agent-prefix"][agent_requests[-1].phase]
    final_ttft_speedup = (
        cold["median_time_to_first_token_seconds"]
        / final["median_time_to_first_token_seconds"]
    )
    lines.extend(
        [
            "",
            (
                f"Agent final step: {final_ttft_speedup:.2f}x TTFT speedup, "
                f"{percentage(final['median_cache_hit_rate'])} restored-token hit rate."
            ),
        ]
    )
    if previous and outputs_match(record, previous):
        lines.extend(["", f"Previous: {previous['label']} ({scope})"])
        for workload in WORKLOADS:
            phase = requests_for(workload)[-1].phase
            current_metrics = record["metrics"][workload.name][phase]
            previous_metrics = previous["metrics"][workload.name][phase]
            current_rate = current_metrics["generation_tokens_per_second"]
            previous_rate = previous_metrics["generation_tokens_per_second"]
            rate_change = (current_rate / previous_rate - 1) * 100
            current_ttft = current_metrics["median_time_to_first_token_seconds"]
            previous_ttft = previous_metrics["median_time_to_first_token_seconds"]
            ttft_saved = (1 - current_ttft / previous_ttft) * 100
            lines.append(
                f"  {workload.name:<16} TTFT time saved {ttft_saved:+.1f}%  "
                f"generation throughput {rate_change:+.1f}%"
            )
    elif previous:
        lines.extend(
            [
                "",
                f"Previous: {previous['label']} ({scope})",
                "Comparison skipped because deterministic outputs changed.",
            ]
        )
    else:
        lines.extend(["", "Baseline established for this suite and model."])
    lines.extend(["", f"Saved: {path.resolve().relative_to(ROOT)}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = get_config()
    tokenizer = Tokenizer.load(config)
    generator = TextGenerator(tokenizer, Engine(config))
    health = generator.health()

    for workload in WORKLOADS:
        reset_prefix_cache(generator)
        for request in requests_for(workload):
            run_request(generator, request)

    samples = []
    for workload in WORKLOADS:
        for _ in range(args.runs):
            reset_prefix_cache(generator)
            for request in requests_for(workload):
                sample = run_request(generator, request)
                if (
                    request.phase in {"uncached", "cold"}
                    and sample["restored_tokens"] != 0
                ):
                    raise RuntimeError(
                        f"{workload.name}/{request.phase} expected an empty prefix cache but restored "
                        f"{sample['restored_tokens']} tokens."
                    )
                if (
                    workload.kind == "prefix"
                    and request.phase != "cold"
                    and sample["restored_tokens"] == 0
                ):
                    raise RuntimeError(
                        f"agent-prefix/{request.phase} did not restore cached tokens."
                    )
                samples.append(sample)

    validate_samples(samples)
    now = datetime.now(UTC)
    definition = suite_definition(args.runs)
    record = {
        "schema_version": 2,
        "timestamp": now.isoformat(),
        "label": args.label,
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "accelerator": accelerator(),
        "git": {
            "commit": git(["rev-parse", "HEAD"]),
            "dirty": bool(git(["status", "--porcelain"])),
        },
        "model": {"id": health["model"], "revision": health["model_revision"]},
        "suite": {**definition, "sha256": suite_hash(definition)},
        "samples": samples,
        "metrics": aggregate_by_workload(samples),
    }
    previous, scope = previous_result(record)
    RESULTS.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in args.label
    )
    filename = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{safe_label}.json"
    path = RESULTS / filename
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(report(record, path, previous, scope))


if __name__ == "__main__":
    main()
