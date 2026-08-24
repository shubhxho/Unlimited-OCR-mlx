#!/usr/bin/env python3
"""Locked-manifest OCR evaluation for base and LoRA-adapted Unlimited-OCR."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Sequence

from training.manifest import load_manifest, resolve_image, validate_manifest_images
from unlimited_ocr_mlx import ocr_images


def normalize(text: str) -> str:
    """Conservative text normalization for CER; raw outputs are always retained."""

    return " ".join(unicodedata.normalize("NFKC", text).split())


def levenshtein(left: str, right: str) -> int:
    """Memory-bounded edit distance."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, char_left in enumerate(left, 1):
        current = [index]
        for column, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def character_error_rate(prediction: str, reference: str) -> float:
    reference = normalize(reference)
    prediction = normalize(prediction)
    return levenshtein(prediction, reference) / max(1, len(reference))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Unlimited-OCR on a frozen JSONL manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument("--model-revision", default="07dea832e22aefee32ad281d4b80551282e1c168")
    parser.add_argument("--adapter-path", type=Path, help="MLX LoRA adapter directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    from mlx_vlm import load
    if args.adapter_path:
        from mlx_vlm.trainer.utils import apply_lora_layers

    examples = load_manifest(args.manifest)
    validate_manifest_images(examples, args.image_root, args.verify_hashes)
    model, processor = load(args.model, revision=args.model_revision)
    if args.adapter_path:
        model = apply_lora_layers(model, str(args.adapter_path))

    rows = []
    for item in examples:
        result = ocr_images(
            model, processor, [resolve_image(item, args.image_root)], prompt=item.prompt,
            max_tokens=args.max_tokens,
        )
        rows.append({
            "id": item.id,
            "source_document": item.source_document,
            "cer": character_error_rate(result.text, item.target),
            "complete": result.finish_reason != "length",
            "finish_reason": result.finish_reason,
            "output": result.text,
            "reference": item.target,
            "metrics": result.metadata(),
        })
    summary = {
        "examples": len(rows),
        "mean_cer": sum(row["cer"] for row in rows) / len(rows),
        "complete_rate": sum(row["complete"] for row in rows) / len(rows),
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
    }
    report = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return report


def main() -> int:
    try:
        run(parse_args())
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
