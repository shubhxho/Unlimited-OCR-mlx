#!/usr/bin/env python3
"""Native Apple Silicon OCR with Baidu Unlimited-OCR and MLX-VLM."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from unlimited_ocr_mlx import (
    BASE,
    GUNDAM,
    MODEL_ID,
    collect_images,
    load_model,
    ocr_images,
    write_result,
)


def render_pdf(pdf_path: Path, work_dir: Path, dpi: int, first_page: int, last_page: int | None) -> list[Path]:
    """Render a selected, ordered PDF page range to lossless PNG files."""

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF input requires PyMuPDF. Install with `uv sync`.") from exc
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    if first_page < 1:
        raise ValueError("first_page must be at least 1")

    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    scale = dpi / 72
    work_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    with fitz.open(pdf_path) as document:
        end = len(document) if last_page is None else min(last_page, len(document))
        if first_page > end:
            raise ValueError(f"Requested page range {first_page}-{last_page} is outside this {len(document)}-page PDF")
        for number in range(first_page, end + 1):
            path = work_dir / f"page_{number:05d}.png"
            document[number - 1].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(path)
            rendered.append(path)
    return rendered


def chunks(items: Sequence[Path], size: int) -> list[list[Path]]:
    if size <= 0:
        return [list(items)] if items else []
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def output_for_single(image: Path, image_dir: Path | None, output_dir: Path) -> Path:
    """Create collision-free output paths for recursive image-directory input."""

    if image_dir is None:
        return output_dir / f"{image.stem}.md"
    return output_dir / image.relative_to(image_dir).with_suffix(".md")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-of-the-art document OCR on Apple Silicon using Baidu Unlimited-OCR + MLX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One image to parse with gundam mode by default")
    source.add_argument("--image-dir", type=Path, help="Recursively process images; each image is its own document by default")
    source.add_argument("--pdf", type=Path, help="Render a PDF and parse its pages together in base mode")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for Markdown and JSON metrics")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local MLX-compatible checkpoint")
    parser.add_argument("--prompt", default=None, help="OCR instruction; defaults to Baidu's single or multi-page prompt")
    parser.add_argument("--mode", choices=("auto", "gundam", "base"), default="auto", help="gundam is single-image crop mode; base is global-view mode")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum generated tokens per OCR request")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35, help="0 disables Baidu's sliding no-repeat guard")
    parser.add_argument("--ngram-window", type=int, default=None, help="Override guard window; default is 128 gundam / 1024 base")
    parser.add_argument("--prefill-step-size", type=int, default=None, help="Chunk long prompt prefill to reduce peak memory")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--first-page", type=int, default=1, help="First PDF page, one-indexed")
    parser.add_argument("--last-page", type=int, default=None, help="Last PDF page, inclusive")
    parser.add_argument("--join-images", action="store_true", help="Treat ordered --image-dir files as one multi-page document")
    parser.add_argument("--max-pages-per-request", type=int, default=0, help="Split joined pages into chunks; 0 sends all pages together")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already have Markdown and metrics")
    return parser.parse_args(argv)


def _request(model, processor, pages: list[Path], args: argparse.Namespace, output: Path) -> bool:
    metrics_path = output.with_suffix(".json")
    if output.exists() and metrics_path.exists() and not args.overwrite:
        print(f"SKIP  {output} (use --overwrite to regenerate)")
        return False
    result = ocr_images(
        model,
        processor,
        pages,
        prompt=args.prompt,
        requested_mode=args.mode,
        max_tokens=args.max_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
        prefill_step_size=args.prefill_step_size,
    )
    markdown_path, metadata_path = write_result(result, output)
    print(
        f"DONE  {markdown_path} | {result.image_count} image(s), {result.mode}, "
        f"{result.generation_tokens} generated tokens, {result.elapsed_seconds:.1f}s, "
        f"{result.generation_tps:.1f} tok/s | metrics: {metadata_path}"
    )
    return True


def run(args: argparse.Namespace) -> int:
    if args.max_pages_per_request < 0:
        raise ValueError("max_pages_per_request cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The actual MLX model is loaded only after cheap input validation/rendering.
    if args.image:
        image = args.image.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image}")
        jobs = [([image], output_for_single(image, None, args.output_dir))]
    elif args.image_dir:
        image_dir = args.image_dir.expanduser().resolve()
        images = collect_images(image_dir)
        if not images:
            raise ValueError(f"No supported images found in {image_dir}")
        if args.join_images:
            jobs = [
                (group, args.output_dir / f"document_{index:04d}.md")
                for index, group in enumerate(chunks(images, args.max_pages_per_request), 1)
            ]
        else:
            if args.mode == "gundam" or args.mode == "auto":
                pass
            jobs = [([image], output_for_single(image, image_dir, args.output_dir)) for image in images]
    else:
        # The temporary PNGs must survive through all MLX calls, so PDF work is
        # handled in this scope rather than returned to the caller.
        with tempfile.TemporaryDirectory(prefix="unlimited_ocr_pages_") as temp:
            pages = render_pdf(args.pdf, Path(temp), args.dpi, args.first_page, args.last_page)
            jobs = [
                (group, args.output_dir / f"{args.pdf.stem}_pages_{group[0].stem[-5:]}-{group[-1].stem[-5:]}.md")
                for group in chunks(pages, args.max_pages_per_request)
            ]
            return _run_jobs(jobs, args)
    return _run_jobs(jobs, args)


def _run_jobs(jobs: list[tuple[list[Path], Path]], args: argparse.Namespace) -> int:
    pending = [
        job
        for job in jobs
        if args.overwrite or not (job[1].exists() and job[1].with_suffix(".json").exists())
    ]
    skipped = len(jobs) - len(pending)
    if not pending:
        print(f"All {len(jobs)} output(s) already exist. Use --overwrite to regenerate.")
        return 0

    print(f"Loading {args.model} with MLX-VLM …")
    model, processor = load_model(args.model)
    print(f"Loaded. Processing {len(pending)} OCR request(s).")
    complete = 0
    for index, (pages, output) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] " + ", ".join(page.name for page in pages[:3]) + (" …" if len(pages) > 3 else ""))
        if _request(model, processor, pages, args, output):
            complete += 1
    print(f"Finished {complete} generated request(s); {skipped} skipped.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
