#!/usr/bin/env python3
"""Collapse criterion's output into one committed CSV.

Criterion writes per-benchmark JSON and raw sample CSVs under ``target/criterion``, which is
build output and therefore gitignored. Phase 7 needs to draw charts without re-running a
multi-minute suite on the same machine, so this extracts the point estimates into a single
file that *is* committed.

Usage::

    cargo bench
    python scripts/summarize_bench.py results/scalar.csv

    cargo +nightly bench --features simd
    python scripts/summarize_bench.py results/simd.csv

The raw per-sample data stays in ``target/criterion/**/new/raw.csv`` for anyone who wants
distributions rather than means.
"""

import csv
import glob
import json
import os
import sys


def collect(root="target/criterion"):
    """Yield (group, benchmark, parameter, mean_ms, median_ms) for every measured benchmark."""
    pattern = os.path.join(root, "**", "new", "estimates.json")

    for path in glob.glob(pattern, recursive=True):
        parts = path.replace(os.sep, "/").split("/")
        try:
            end = parts.index("new")
        except ValueError:
            continue

        # <root>/<group>/<benchmark>/<parameter>/new/estimates.json, with the parameter
        # segment absent for benchmarks that take no input.
        key = parts[parts.index("criterion") + 1:end]
        if not key:
            continue
        group = key[0]
        benchmark = key[1] if len(key) > 1 else ""
        parameter = key[2] if len(key) > 2 else ""

        with open(path) as handle:
            estimates = json.load(handle)

        yield (
            group,
            benchmark,
            parameter,
            round(estimates["mean"]["point_estimate"] / 1e6, 4),
            round(estimates["median"]["point_estimate"] / 1e6, 4),
        )


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <output.csv>")

    destination = sys.argv[1]
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

    rows = sorted(collect())
    if not rows:
        sys.exit("no criterion estimates found; run `cargo bench` first")

    with open(destination, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "benchmark", "parameter", "mean_ms", "median_ms"])
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
