#!/usr/bin/env python3
"""LoRA fine-tuning for the native MLX Unlimited-OCR model.

This intentionally supports batch size one. Unlimited-OCR uses a variable
number of visual tiles, so ordinary tensor batching can silently corrupt image
placeholder alignment. Gradient accumulation provides a larger effective batch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

from training.manifest import (
    OCRExample, assert_disjoint_source_documents, load_manifest, require_provenance,
    resolve_image, validate_manifest_images,
)

IGNORE_INDEX = -100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Unlimited-OCR with MLX LoRA adapters.")
    parser.add_argument("--train", required=True, type=Path, help="Training JSONL manifest")
    parser.add_argument("--validation", type=Path, help="Held-out validation JSONL manifest")
    parser.add_argument("--image-root", required=True, type=Path, help="Root directory containing manifest images")
    parser.add_argument("--model", default="baidu/Unlimited-OCR", help="BF16 Unlimited-OCR checkpoint or local MLX checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/unlimited-ocr-lora"))
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true", help="Verify manifest image SHA-256 values before training")
    parser.add_argument("--allow-unverified-provenance", action="store_true", help="Development-only: allow missing license/source/hash fields")
    return parser.parse_args(argv)


def _require_positive(args: argparse.Namespace) -> None:
    for name in ("iters", "rank", "alpha", "max_seq_length", "gradient_accumulation", "eval_every", "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def _eos_text(processor: Any) -> str:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos, str) or not eos:
        raise RuntimeError("Unlimited-OCR tokenizer has no EOS token")
    return eos


def prepare_example(
    example: OCRExample,
    image_root: str | Path,
    processor: Any,
    config: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    """Create model inputs and completion-only labels for one image.

    The prefix is processed separately solely to find its exact expanded visual
    token boundary. This avoids training the prompt/image placeholders and
    prevents the common image-feature/placeholder misalignment failure.
    """

    from PIL import Image
    from mlx_vlm.prompt_utils import apply_chat_template
    import mlx.core as mx
    import numpy as np

    image_path = resolve_image(example, image_root)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    prompt = apply_chat_template(processor, config, example.prompt, num_images=1)
    if prompt.count("<image>") != 1:
        raise RuntimeError(f"Example {example.id}: expected exactly one image marker")
    # The native processor has the same image geometry for both calls.
    prefix = processor.process_one(prompt, [image], base_size=1024, image_size=640, cropping=True)
    full = processor.process_one(
        prompt + example.target + _eos_text(processor),
        [image],
        base_size=1024,
        image_size=640,
        cropping=True,
    )
    sequence_length = int(full["input_ids"].shape[-1])
    if sequence_length > max_seq_length:
        raise ValueError(
            f"Example {example.id}: {sequence_length} tokens exceeds max_seq_length={max_seq_length}; "
            "do not truncate image-token examples. Split or filter this sample."
        )
    prefix_length = int(prefix["input_ids"].shape[-1])
    labels = np.asarray(full["labels"]).copy()
    labels[:prefix_length] = IGNORE_INDEX
    full["labels"] = mx.array(labels)[None, :]
    full["id"] = example.id
    return full


def completion_loss(model: Any, batch: dict[str, Any]):
    """Causal cross entropy over target tokens only, with visual inputs intact."""

    import mlx.core as mx
    import mlx.nn as nn

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if input_ids.shape[-1] < 2:
        raise ValueError("Training sequence must contain at least two tokens")
    result = model(
        input_ids[:, :-1],
        batch["images"],
        images_spatial_crop=batch["images_spatial_crop"],
        images_seq_mask=batch["images_seq_mask"][:, :-1],
    )
    logits = result.logits.astype(mx.float32)
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    safe_targets = mx.where(mask, targets, mx.zeros_like(targets))
    token_loss = nn.losses.cross_entropy(logits, safe_targets)
    return (token_loss * mask).sum() / mx.maximum(mask.sum(), 1)


def _install_lora(model: Any, rank: int, alpha: float, dropout: float) -> None:
    """Freeze base weights and adapt all text decoder layers, including MoE."""

    from mlx_vlm.trainer.adapter_utils import linear_to_lora_layers

    model.freeze()
    lora_parameters = {"rank": rank, "scale": alpha / rank, "dropout": dropout}
    # LanguageModel.layers exposes the decoder blocks. The MLX-VLM helper knows
    # SwitchLinear, so MoE expert/routing projections are not accidentally skipped.
    linear_to_lora_layers(model.language_model, num_layers=-1, config=lora_parameters)
    model.config.lora = {
        "fine_tune_type": "lora",
        "num_layers": -1,
        "lora_parameters": lora_parameters,
    }


def _mean_validation_loss(model: Any, examples: list[OCRExample], args: argparse.Namespace, processor: Any) -> float:
    import mlx.core as mx

    if not examples:
        return float("nan")
    model.eval()
    total = 0.0
    for example in examples:
        batch = prepare_example(example, args.image_root, processor, model.config, args.max_seq_length)
        loss = completion_loss(model, batch)
        mx.eval(loss)
        total += loss.item()
        mx.clear_cache()
    model.train()
    return total / len(examples)


def run(args: argparse.Namespace) -> None:
    _require_positive(args)
    import mlx.core as mx
    import mlx.optimizers as optim
    import mlx.nn as nn
    from mlx_vlm import load
    from mlx_vlm.trainer.utils import save_adapter

    random.seed(args.seed)
    mx.random.seed(args.seed)
    train_examples = load_manifest(args.train)
    validation_examples = load_manifest(args.validation) if args.validation else []
    validate_manifest_images(train_examples + validation_examples, args.image_root, args.verify_hashes)
    assert_disjoint_source_documents(train=train_examples, validation=validation_examples)
    if not args.allow_unverified_provenance:
        require_provenance(train_examples + validation_examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    print(f"Loading {args.model} …")
    model, processor = load(args.model)
    _install_lora(model, args.rank, args.alpha, args.dropout)
    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, completion_loss)
    model.train()
    grad_accum = None
    order = list(range(len(train_examples)))

    for step in range(1, args.iters + 1):
        if (step - 1) % len(order) == 0:
            random.shuffle(order)
        example = train_examples[order[(step - 1) % len(order)]]
        started = time.perf_counter()
        batch = prepare_example(example, args.image_root, processor, model.config, args.max_seq_length)
        loss, grad = loss_and_grad(model, batch)
        if grad_accum is None:
            grad_accum = grad
        else:
            from mlx.utils import tree_map
            grad_accum = tree_map(lambda left, right: left + right, grad_accum, grad)
        if step % args.gradient_accumulation == 0:
            from mlx.utils import tree_map
            optimizer.update(model, tree_map(lambda value: value / args.gradient_accumulation, grad_accum))
            grad_accum = None
        mx.eval(loss, model.state, optimizer.state)
        print(f"step={step} example={example.id} loss={loss.item():.6f} seconds={time.perf_counter() - started:.2f}", flush=True)
        mx.clear_cache()

        if validation_examples and (step % args.eval_every == 0 or step == args.iters):
            validation_loss = _mean_validation_loss(model, validation_examples, args, processor)
            print(f"step={step} validation_loss={validation_loss:.6f}", flush=True)
        if step % args.save_every == 0 or step == args.iters:
            checkpoint = args.output_dir / f"step-{step:07d}.safetensors"
            save_adapter(model, checkpoint)
            save_adapter(model, args.output_dir / "adapters.safetensors")
            print(f"saved {checkpoint}", flush=True)


def main() -> int:
    try:
        run(parse_args())
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
