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

from helios.config import HeliosConfig, get_config
from helios.runtime.client import SchedulerClient
from helios.runtime.types import Sampling
from helios.runtime.worker import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
PROMPTS = (
    "Reply with one word: ready.",
    "Explain the value of a key-value cache during autoregressive language-model decoding.",
    "Explain solar power simply, using one practical example.",
    (
        "A small team runs an online booking service for local clinics. Patients can search "
        "for appointments, receive reminder messages, reschedule visits, and upload insurance "
        "documents. The team has seen duplicate bookings during traffic spikes, occasional slow "
        "searches after importing new provider schedules, and confusing error messages when a "
        "reminder provider is unavailable. They have one backend service, a PostgreSQL database, "
        "a background worker for reminders, and basic request logs. They cannot afford a full "
        "rewrite or a large operations team. Propose a practical reliability plan for the next "
        "three months. Prioritize the work, explain the tradeoffs, identify what should be "
        "measured, and give concrete examples of changes the team can make to prevent duplicate "
        "bookings, keep search responsive, and handle reminder failures gracefully. Keep the "
        "answer structured and specific enough for an engineer to begin implementation."
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a repeatable Helios scheduler benchmark."
    )
    parser.add_argument("--name", required=True, help="Stable name for this workload.")
    parser.add_argument("--runs", type=int, default=5, help="Measured requests after warmup.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--endpoint", help="Override HELIOS_SCHEDULER_ENDPOINT.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
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
        return {"kind": "mps", "name": "Apple Metal", "torch_version": torch.__version__}
    return {"kind": "cpu", "name": platform.processor() or "unknown", "torch_version": torch.__version__}


def load_results() -> list[dict[str, Any]]:
    records = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"Skipping invalid result: {path}", file=sys.stderr)
    return records


def previous_result(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    matches = [
        item
        for item in load_results()
        if item.get("name") == record["name"]
        and item.get("workload") == record["workload"]
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


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1_000:.1f} ms"
    return f"{seconds:.3f} s"


def report(
    record: dict[str, Any], path: Path, previous: dict[str, Any] | None, scope: str | None
) -> str:
    metrics = record["metrics"]
    machine = record["machine"]
    accelerator_info = record["accelerator"]
    lines = [
        "",
        "╭─ Helios benchmark ─────────────────────────────────────",
        f"│  {record['name']}  ·  {record['model']['id']}@{record['model']['revision'][:12]}",
        f"│  {accelerator_info['name']} ({accelerator_info['kind']})  ·  {machine['hostname']}",
        f"│  {record['workload']['prompts']} prompts × {record['workload']['runs']} runs",
        "├────────────────────────────────────────────────────────",
        f"│  End-to-end latency  {duration(metrics['median_end_to_end_seconds']):>10}",
        f"│  Time to first token {duration(metrics['median_time_to_first_token_seconds']):>10}",
        f"│  Inter-token latency {duration(metrics['median_inter_token_latency_seconds']):>10}",
        f"│  Throughput          {metrics['tokens_per_second']:>8.2f} tok/s",
        "├────────────────────────────────────────────────────────",
    ]
    if previous:
        before = previous["metrics"]["tokens_per_second"]
        change = (metrics["tokens_per_second"] / before - 1) * 100
        trend = "▲ improvement" if change >= 0 else "▼ worsening"
        lines.append(f"│  {trend:<16} {change:+.1f}% vs {before:.2f} tok/s")
        lines.append(f"│  Comparison: {scope}")
    else:
        lines.append("│  ● Baseline established")
    lines.extend(
        [
            "╰────────────────────────────────────────────────────────",
            f"  Saved: {path.resolve().relative_to(ROOT)}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = get_config()
    if args.endpoint:
        config = HeliosConfig(
            **{**config.__dict__, "scheduler_endpoint": args.endpoint}
        )
    tokenizer = Tokenizer.load(config)
    client = SchedulerClient(config)
    health = client.health()
    prompt_tokens = [len(tokenizer.tokenize(prompt)) for prompt in PROMPTS]
    sampling = Sampling(temperature=0, top_p=1, max_new_tokens=args.max_new_tokens)

    def generate(prompt: str) -> tuple[int, float, float, list[float]]:
        started = time.perf_counter()
        prompt_ids = tokenizer.tokenize(prompt)
        result = client.generate(
            model_id=tokenizer.model_id,
            model_revision=tokenizer.model_revision,
            input_ids=prompt_ids,
            eos_token_id=tokenizer.eos_token_id,
            sampling=sampling,
        )
        if result.timing is None:
            raise RuntimeError(
                "Scheduler did not return token timings. Restart it with this version of Helios."
            )
        return (
            len(result.output_ids),
            time.perf_counter() - started,
            result.timing.prefill_seconds,
            result.timing.inter_token_seconds,
        )

    for prompt in PROMPTS:
        generate(prompt)
    samples = [
        (prompt_index, *generate(prompt))
        for _ in range(args.runs)
        for prompt_index, prompt in enumerate(PROMPTS)
    ]
    tokens = [count for _, count, _, _, _ in samples]
    durations = [duration for _, _, duration, _, _ in samples]
    prefill_times = [prefill for _, _, _, prefill, _ in samples]
    inter_token_times = [time for _, _, _, _, times in samples for time in times]
    throughput = median(count / duration for count, duration in zip(tokens, durations))
    now = datetime.now(UTC)
    record = {
        "schema_version": 1,
        "timestamp": now.isoformat(),
        "name": args.name,
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "accelerator": accelerator(),
        "git": {"commit": git(["rev-parse", "HEAD"]), "dirty": bool(git(["status", "--porcelain"]))},
        "model": {"id": health.model_id, "revision": health.model_revision},
        "workload": {
            "prompt_sha256": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
            "prompt_tokens": prompt_tokens,
            "prompts": len(PROMPTS),
            "max_new_tokens": args.max_new_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "runs": args.runs,
        },
        "samples": [
            {
                "prompt_index": prompt_index,
                "output_tokens": count,
                "end_to_end_seconds": duration,
                "time_to_first_token_seconds": prefill,
                "inter_token_seconds": times,
            }
            for prompt_index, count, duration, prefill, times in samples
        ],
        "metrics": {
            "median_output_tokens": median(tokens),
            "median_end_to_end_seconds": median(durations),
            "median_time_to_first_token_seconds": median(prefill_times),
            "median_inter_token_latency_seconds": (
                median(inter_token_times) if inter_token_times else None
            ),
            "tokens_per_second": throughput,
        },
    }
    previous, scope = previous_result(record)
    RESULTS.mkdir(parents=True, exist_ok=True)
    filename = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{args.name}.json"
    path = RESULTS / filename
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(report(record, path, previous, scope))


if __name__ == "__main__":
    main()
