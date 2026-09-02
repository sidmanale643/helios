import argparse
import json
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
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
        description="Load Helios once and run every benchmark workload in order."
    )
    parser.add_argument("--label", default="run", help="Name for the saved result.")
    return parser.parse_args()


def accelerator() -> dict[str, str]:
    if torch.cuda.is_available():
        return {"kind": "cuda", "name": torch.cuda.get_device_name(0)}
    if torch.backends.mps.is_available():
        return {"kind": "mps", "name": "Apple Metal"}
    return {"kind": "cpu", "name": platform.processor() or "unknown"}


def run_request(generator: TextGenerator, spec: RequestSpec) -> dict[str, Any]:
    sampling = Sampling(temperature=0, top_p=1, max_new_tokens=spec.workload.max_new_tokens)
    started = time.perf_counter()
    tokenize_started = time.perf_counter()
    input_ids = generator.tokenizer.tokenize_chat(list(spec.messages))
    tokenize_seconds = time.perf_counter() - tokenize_started
    result = generator.engine.run(input_ids, generator.tokenizer.eos_token_id, sampling)
    output = generator.tokenizer.detokenize(result.output_ids)
    end_to_end_seconds = time.perf_counter() - started
    generation_seconds = result.prefill_seconds + sum(result.inter_token_seconds)

    return {
        "workload": spec.workload.name,
        "phase": spec.phase,
        "input": spec.messages,
        "output": output,
        "metrics": {
            "prompt_tokens": len(input_ids),
            "output_tokens": len(result.output_ids),
            "end_to_end_seconds": end_to_end_seconds,
            "time_to_first_token_seconds": (
                tokenize_seconds
                + result.queue_seconds
                + result.prefix_lookup_seconds
                + result.restore_seconds
                + result.prefill_seconds
            ),
            "generation_tokens_per_second": (
                len(result.output_ids) / generation_seconds
                if generation_seconds > 0
                else None
            ),
            "restored_tokens": result.prefix.restored_tokens,
            "cache_hit_rate": result.prefix.restored_tokens / len(input_ids),
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
    config = get_config()
    tokenizer = Tokenizer.load(config)
    generator = TextGenerator(tokenizer, Engine(config))
    health = generator.health()

    samples = []
    for workload in WORKLOADS:
        for request in requests_for(workload):
            samples.append(run_request(generator, request))

    now = datetime.now(UTC)
    record = {
        "schema_version": 3,
        "timestamp": now.isoformat(),
        "label": args.label,
        "machine": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "accelerator": accelerator(),
        "model": {"id": health["model"], "revision": health["model_revision"]},
        "suite_version": SUITE_VERSION,
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
