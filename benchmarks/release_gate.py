#!/usr/bin/env python3
"""Promotion gate for a candidate Unlimited-OCR adapter/checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve a candidate only when locked benchmark evidence passes.")
    parser.add_argument("--comparison", type=Path, required=True, help="Output of benchmarks.compare")
    parser.add_argument("--candidate", type=Path, required=True, help="Adapter/checkpoint directory to promote")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--max-cer-regression", type=float, default=0.0)
    parser.add_argument("--require-statistical-improvement", action="store_true", help="Require CI upper bound below zero")
    return parser.parse_args(argv)


def decide(comparison: dict, max_cer_regression: float, require_statistical_improvement: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if comparison.get("incomplete_ids"):
        reasons.append("at least one locked evaluation example was incomplete")
    delta = comparison.get("mean_cer_delta_candidate_minus_base")
    interval = comparison.get("bootstrap_95_ci")
    if not isinstance(delta, (int, float)) or not isinstance(interval, list) or len(interval) != 2:
        reasons.append("comparison report is missing paired CER statistics")
    else:
        if delta > max_cer_regression:
            reasons.append(f"mean CER regressed by {delta:.6f}, allowed {max_cer_regression:.6f}")
        if require_statistical_improvement and interval[1] >= 0:
            reasons.append("95% CI does not establish a CER improvement")
    return not reasons, reasons


def main() -> int:
    args = parse_args()
    try:
        if not args.candidate.exists():
            raise FileNotFoundError(f"Candidate artifact not found: {args.candidate}")
        comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
        passed, reasons = decide(comparison, args.max_cer_regression, args.require_statistical_improvement)
        receipt = {
            "passed": passed,
            "candidate": str(args.candidate),
            "comparison": str(args.comparison),
            "criteria": {
                "max_cer_regression": args.max_cer_regression,
                "require_statistical_improvement": args.require_statistical_improvement,
            },
            "reasons": reasons,
        }
        args.release_dir.mkdir(parents=True, exist_ok=True)
        (args.release_dir / "promotion_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0 if passed else 3
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
