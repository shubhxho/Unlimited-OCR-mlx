# Training and Evaluation

This project improves **Baidu Unlimited-OCR** through reproducible MLX experiments. It does not claim a new architecture or benchmark lead without an independently reproducible result.

## Training data contract

Put images outside Git and make one JSON object per line:

```json
{"id":"doc-001-p001","image":"train/doc-001-p001.png","target":"Canonical transcription", "prompt":"document parsing.", "source_document":"doc-001", "license":"CC-BY-4.0", "sha256":"..."}
```

Use document-level source IDs. Pages, alternate renders, templates, or near duplicates from a source must remain in one split. Keep public test manifests read-only and never use them for prompt selection, data synthesis, or hyperparameter choice.

## Hardware preflight

Unlimited-OCR has a 3.3B-parameter dual-vision/MoE design. Its BF16 weights, dynamic visual activations, optimizer state, and LoRA gradients do not fit safely in a 16 GB unified-memory Mac. Use a 32 GB machine only for careful small LoRA experiments; 64 GB+ is the practical minimum for the full OCR training/evaluation loop. Keep at least 20 GB free disk for the source checkpoint, cache, adapters, reports, and conversion artifacts.

```bash
uv run unlimited-ocr-mlx-preflight \
  --manifest manifests/train.jsonl --image-root data/images --workspace . \
  --verify-hashes
```

The command validates every image and optional checksum before loading the model, then exits nonzero rather than starting a run that will OOM or exhaust disk.

## LoRA baseline

```bash
uv run python -m training.train_lora \
  --train manifests/train.jsonl \
  --validation manifests/validation.jsonl \
  --image-root data/images \
  --output-dir runs/ocr-lora-001 \
  --model-revision 07dea832e22aefee32ad281d4b80551282e1c168 \
  --iters 200 --rank 16 --alpha 32 --max-seq-length 8192 \
  --verify-hashes
```

The trainer deliberately uses one example per optimization step. Unlimited-OCR has variable dynamic visual crops; generic batching can misalign image features and `<image>` positions. Increase effective batch size only through `--gradient-accumulation` after validating memory and output parity.

The trainer:

- keeps the official single-page `gundam` preprocessing;
- masks the prompt and image placeholders, training only canonical target tokens;
- rejects over-length image examples instead of truncating visual placeholders;
- freezes the base model and trains MLX LoRA modules in the language decoder;
- saves adapter weights and their matching `adapter_config.json`.

## Evaluation gate

```bash
# Base model, then adapter. Compare reports on the same immutable manifest.
uv run python -m benchmarks.evaluate \
  --manifest manifests/eval-public.jsonl --image-root data/images \
  --output reports/base.json --verify-hashes

uv run python -m benchmarks.evaluate \
  --manifest manifests/eval-public.jsonl --image-root data/images \
  --adapter-path runs/ocr-lora-001 \
  --output reports/lora-001.json --verify-hashes
```

The generic evaluator reports conservative normalized CER, completion rate, raw output, reference, finish reason, and runtime metrics. It is a smoke/evidence layer, not a substitute for official benchmark evaluators.

Before release, run the pinned official OmniDocBench evaluator on a locked version, report every component and difficult slice, and compare against the official BF16 Unlimited-OCR baseline with identical images, prompts, DPI, page grouping, and decoding settings. Publish an adapter only when it improves the held-out validation decision metric and does not regress the locked public test or MLX parity gate.

## Candidate comparison and promotion

A release requires a paired comparison against a baseline report generated from the **same locked manifest**:

```bash
uv run unlimited-ocr-mlx-compare \
  --base reports/base.json --candidate reports/lora-001.json \
  --output reports/lora-001-vs-base.json --seed 42

uv run unlimited-ocr-mlx-release-gate \
  --comparison reports/lora-001-vs-base.json \
  --candidate runs/ocr-lora-001 --release-dir releases/ocr-lora-001 \
  --require-statistical-improvement
```

The gate rejects incomplete generations, mismatched manifests, CER regressions, and adapters whose paired bootstrap interval does not establish improvement. Passing this generic gate is necessary but not sufficient: release still requires official OmniDocBench and task-specific evaluation with the predeclared margins.

## Architecture roadmap

1. **Baseline and parity:** BF16 MLX must first match the official checkpoint.
2. **Data:** add only licensed, deduplicated examples with source-document splits and a licence ledger.
3. **LoRA:** target evidence-backed failures (small text, tables, formulas, reading order) with separate ablations.
4. **Vision changes:** only after LoRA/data plateaus, train and test multi-scale vision/projector changes from a reproducible checkpoint. This is an architecture experiment, not an inference flag.
5. **Release:** require public benchmark scores, document-level bootstrap confidence intervals, raw outputs, model/data revisions, and an explicit comparison table.

Do not train on OmniDocBench public test data or claim a new state of the art from internal validation alone.
