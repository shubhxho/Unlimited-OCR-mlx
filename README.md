---
library_name: mlx
license: mit
base_model: baidu/Unlimited-OCR
tags:
- ocr
- document-parsing
- mlx
- apple-silicon
---

# Unlimited-OCR for MLX

Native Apple Silicon inference for [Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR). This project uses MLX-VLM's dedicated `unlimited_ocr` implementation. It is not a generic Transformers wrapper.

The MLX backend preserves the released model design:

- SAM ViT-B + CLIP-L dual vision encoder
- 2048 → 1280 visual projector
- DeepSeek-V2-style top-6 MoE decoder
- Unlimited-OCR's R-SWA sliding decode cache
- official `gundam` dynamic-crop and `base` global-view preprocessing
- deterministic decoding with the released sliding no-repeat 35-gram guard

## Checkpoint status

This repository publishes the native MLX runner and model card. It resolves the official BF16 checkpoint from [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) at run time, so it does **not** yet duplicate a standalone MLX-weight artifact. A standalone quantized checkpoint will only be published after conversion and output-parity evaluation against the official checkpoint.

## Requirements

- macOS on Apple Silicon
- Python 3.10+
- enough unified memory for the 6.7 GB BF16 checkpoint and document activations; 32 GB+ is recommended for large pages or long PDFs

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

`mlx-vlm==0.6.16` is pinned because it includes the dedicated Unlimited-OCR architecture. Do not substitute an older arbitrary MLX-VLM version.

## Run

```bash
# A single image: official gundam mode (1024 global view + 640 dynamic crops)
uv run unlimited-ocr-mlx --image receipt.png --output-dir outputs

# A PDF: render at 300 DPI, then parse all ordered pages in one request
uv run unlimited-ocr-mlx --pdf report.pdf --output-dir outputs

# An image directory: each image is a separate document
uv run unlimited-ocr-mlx --image-dir scans/ --output-dir outputs

# Treat an ordered directory as one multi-page document
uv run unlimited-ocr-mlx --image-dir pages/ --join-images --output-dir outputs
```

Each request produces Markdown and a JSON sidecar with timing, token counts, memory, and finish reason. Existing pairs are skipped, so re-run the command to resume. Use `--overwrite` to regenerate.

### Important settings

| Input | Default mode | Released preprocessing | Default prompt |
|---|---|---|---|
| One image | `gundam` | 1024² global view plus 640² aspect-ratio tiles | `document parsing.` |
| Multi-page PDF / `--join-images` | `base` | one 1024² global view per page | `Multi page parsing.` |

```bash
# Split a very long PDF into independent multi-page requests.
uv run unlimited-ocr-mlx --pdf book.pdf --max-pages-per-request 8

# Reduce render size or select a PDF page range.
uv run unlimited-ocr-mlx --pdf book.pdf --dpi 200 --first-page 5 --last-page 20

# Pass another task prompt or disable the repeat guard for comparison.
uv run unlimited-ocr-mlx --image page.png --prompt "Free OCR." --no-repeat-ngram-size 0

# Lower peak prefill memory on long visual prompts.
uv run unlimited-ocr-mlx --pdf book.pdf --prefill-step-size 512
```

`--max-pages-per-request` deliberately creates independent OCR requests. Use `0` (the default) for true one-shot long-horizon parsing.

## Python API

```python
from unlimited_ocr_mlx import load_model, ocr_images, write_result

model, processor = load_model("baidu/Unlimited-OCR")
result = ocr_images(model, processor, ["page_0001.png", "page_0002.png"])
print(result.text)
write_result(result, "outputs/document.md")
```

For one page, `ocr_images` selects `gundam`; for two or more it selects `base`. It inserts exactly one `<image>` marker as required by Unlimited-OCR's multi-page protocol.

## Validation

```bash
uv run --with pytest pytest -q
uv run unlimited-ocr-mlx --help
```

The automated tests validate input selection, PDF range rendering, image ordering, output mapping, and the upstream-equivalent n-gram rule. A full model run downloads the model checkpoint from Hugging Face and requires Apple Silicon, so it is not run in unit tests.

## Credits

Model: [Baidu Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), MIT license. MLX model implementation: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
