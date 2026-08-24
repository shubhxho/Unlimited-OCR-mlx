#!/usr/bin/env python3
"""Paired, seeded comparison of two locked-manifest OCR evaluation reports."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Sequence


def load_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report.get("rows"), list) or not report["rows"]:
        raise ValueError(f"Invalid or empty evaluation report: {path}")
    rows = {row.get("id"): row for row in report["rows"]}
    if None in rows or len(rows) != len(report["rows"]):
        raise ValueError(f"Report has missing or duplicate row IDs: {path}")
    return report


def paired_bootstrap(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    """Return a two-sided 95% percentile interval for mean paired difference."""

    if not values:
        raise ValueError("Cannot bootstrap an empty comparison")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def compare(base: dict, candidate: dict, samples: int = 5000, seed: int = 42) -> dict:
    base_rows = {row["id"]: row for row in base["rows"]}
    candidate_rows = {row["id"]: row for row in candidate["rows"]}
    if base_rows.keys() != candidate_rows.keys():
        only_base = sorted(base_rows.keys() - candidate_rows.keys())
        only_candidate = sorted(candidate_rows.keys() - base_rows.keys())
        raise ValueError(f"Reports evaluate different IDs; only_base={only_base[:5]}, only_candidate={only_candidate[:5]}")
    incomplete = [
        identifier for identifier in base_rows
        if not base_rows[identifier].get("complete") or not candidate_rows[identifier].get("complete")
    ]
    deltas = [candidate_rows[identifier]["cer"] - base_rows[identifier]["cer"] for identifier in sorted(base_rows)]
    low, high = paired_bootstrap(deltas, samples, seed)
    return {
        "examples": len(deltas),
        "base_mean_cer": mean(row["cer"] for row in base_rows.values()),
        "candidate_mean_cer": mean(row["cer"] for row in candidate_rows.values()),
        "mean_cer_delta_candidate_minus_base": mean(deltas),
        "bootstrap_95_ci": [low, high],
        "incomplete_ids": incomplete,
        "seed": seed,
        "bootstrap_samples": samples,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate OCR reports using paired bootstrap CER.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        if args.samples < 100:
            raise ValueError("samples must be at least 100")
        output = compare(load_report(args.base), load_report(args.candidate), args.samples, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
