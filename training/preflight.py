#!/usr/bin/env python3
"""Fail early when an Unlimited-OCR MLX training run is not safe to start."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from training.manifest import load_manifest, require_provenance, validate_manifest_images


def memory_gb() -> float | None:
    try:
        value = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(value) / (1024**3)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check data and hardware before Unlimited-OCR MLX training.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Disk volume used for model/data/run artifacts")
    parser.add_argument("--min-memory-gb", type=float, default=32.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--allow-unverified-provenance", action="store_true", help="Development-only: do not require source, license, and hash fields")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.min_memory_gb <= 0 or args.min_free_disk_gb <= 0:
        raise ValueError("minimum memory and disk requirements must be positive")
    examples = load_manifest(args.manifest)
    validate_manifest_images(examples, args.image_root, args.verify_hashes)
    if not args.allow_unverified_provenance:
        require_provenance(examples)
    total_memory = memory_gb()
    free_disk = shutil.disk_usage(args.workspace).free / (1024**3)
    checks = {
        "examples": len(examples),
        "memory_gb": total_memory,
        "free_disk_gb": free_disk,
        "minimum_memory_gb": args.min_memory_gb,
        "minimum_free_disk_gb": args.min_free_disk_gb,
        "memory_ok": total_memory is not None and total_memory >= args.min_memory_gb,
        "disk_ok": free_disk >= args.min_free_disk_gb,
    }
    checks["ready"] = checks["memory_ok"] and checks["disk_ok"]
    print(json.dumps(checks, indent=2))
    return checks


def main() -> int:
    try:
        checks = run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    if not checks["ready"]:
        print("Training not started: satisfy the reported hardware requirements first.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
