# Benchmarks

Record a repeatable decode benchmark:

```bash
uv run python benchmarks/run.py --name decode-128
```

The first request warms the model and is excluded. By default, the benchmark
runs five measured requests against the fixed prompt and reports:

- End-to-end latency: CPU tokenization and GPU/Metal execution.
- Time to first token: prompt prefill plus selection of the first output token
  on the model device.
- Inter-token latency: the median time to produce each subsequent output token
  on the model device.
- Throughput: generated output tokens divided by median end-to-end latency.

Each run creates a JSON file in `benchmarks/results/`. It includes the model
revision, commit and dirty state, hostname, platform, detected accelerator,
workload, individual timings, and aggregate metrics. The command prints the
change from the most recent matching workload. It compares the same host and
accelerator first; when none exists, it falls back to the most recent result
from another machine and labels that comparison.

Keep the benchmark name and options unchanged when comparing implementation
changes. Use a new name when changing the workload:

```bash
uv run python benchmarks/run.py --name decode-128 --runs 10
uv run python benchmarks/run.py --name decode-256 --max-new-tokens 256
```

Results are versioned with the code so benchmarks run on another machine can
be compared after syncing the repository. The script fails without writing a
result if a measured request produces a different number of tokens than the
others.
