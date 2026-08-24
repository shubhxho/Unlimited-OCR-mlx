"""Production MLX runner for Baidu Unlimited-OCR.

The model implementation lives in mlx-vlm (``unlimited_ocr``).  This module
keeps the model-specific inference contract in one place: exact prompt format,
image mode, R-SWA-friendly no-repeat guard, PDF rendering, and one-shot
multi-page parsing.
"""

from __future__ import annotations

import json
import os
import re
import time
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

MODEL_ID = "baidu/Unlimited-OCR"
MODEL_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
SINGLE_IMAGE_PROMPT = "document parsing."
MULTI_PAGE_PROMPT = "Multi page parsing."
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class OCRMode:
    """The two image modes released by Baidu for Unlimited-OCR."""

    name: str
    cropping: bool
    base_size: int
    image_size: int
    ngram_window: int


GUNDAM = OCRMode("gundam", cropping=True, base_size=1024, image_size=640, ngram_window=128)
BASE = OCRMode("base", cropping=False, base_size=1024, image_size=1024, ngram_window=1024)


@dataclass
class OCRResult:
    text: str
    elapsed_seconds: float
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    finish_reason: str | None
    image_count: int
    mode: str

    @property
    def complete(self) -> bool:
        """False when generation exhausted its budget and output may be truncated."""

        return self.finish_reason != "length"

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        data["complete"] = self.complete
        return data


def natural_sort_key(path: Path) -> list[object]:
    """Sort page_2 before page_10 without depending on locale."""

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def collect_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_sort_key,
    )


def validate_images(paths: Sequence[str | Path]) -> list[Path]:
    images = [Path(path).expanduser().resolve() for path in paths]
    if not images:
        raise ValueError("At least one image is required.")
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image file(s) not found: " + ", ".join(missing))
    unsupported = [str(path) for path in images if path.suffix.lower() not in IMAGE_EXTENSIONS]
    if unsupported:
        raise ValueError("Unsupported image type(s): " + ", ".join(unsupported))
    return images


def _banned_tokens(sequence: Sequence[int], ngram_size: int, window: int, whitelist: set[int]) -> set[int]:
    """Return continuations that repeat the current n-gram in the recent window.

    This exactly follows Unlimited-OCR's released
    ``SlidingWindowNoRepeatNgramProcessor``.  It is deliberately small and
    runs only once per generated token; n=35 makes the lookup cheap.
    """

    if ngram_size <= 0 or window <= 0 or len(sequence) < ngram_size:
        return set()
    start = max(0, len(sequence) - window)
    end = len(sequence) - ngram_size + 1
    if end <= start:
        return set()
    prefix = tuple(sequence[-(ngram_size - 1) :]) if ngram_size > 1 else ()
    banned = {
        sequence[index + ngram_size - 1]
        for index in range(start, end)
        if ngram_size == 1 or tuple(sequence[index : index + ngram_size - 1]) == prefix
    }
    return banned - whitelist


class SlidingWindowNoRepeatNgramProcessor:
    """MLX logits processor compatible with Baidu's released repeat guard."""

    def __init__(self, ngram_size: int = 35, window: int = 128, whitelist_token_ids: Iterable[int] = ()):
        if ngram_size < 1:
            raise ValueError("ngram_size must be at least 1")
        if window < 1:
            raise ValueError("window must be at least 1")
        self.ngram_size = ngram_size
        self.window = window
        self.whitelist = set(whitelist_token_ids)

    def banned_tokens(self, token_ids: Sequence[int]) -> set[int]:
        return _banned_tokens(token_ids, self.ngram_size, self.window, self.whitelist)

    def __call__(self, input_ids, logits):
        # Import lazily so file discovery, CLI help, and unit tests do not need MLX.
        import mlx.core as mx

        sequences = input_ids.tolist()
        unbatched = input_ids.ndim == 1
        if unbatched:
            sequences = [sequences]
            logits = logits[None, :] if logits.ndim == 1 else logits

        rows = []
        vocab_ids = mx.arange(logits.shape[-1])
        for sequence, row in zip(sequences, logits):
            banned = sorted(self.banned_tokens(sequence))
            if not banned:
                rows.append(row)
                continue
            banned_ids = mx.array(banned)
            mask = mx.any(vocab_ids[:, None] == banned_ids[None, :], axis=1)
            rows.append(mx.where(mask, mx.array(float("-inf"), dtype=row.dtype), row))
        output = mx.stack(rows)
        return output[0] if unbatched else output


def mode_for(image_count: int, requested_mode: str = "auto") -> OCRMode:
    if requested_mode == "auto":
        return GUNDAM if image_count == 1 else BASE
    if requested_mode == "gundam":
        if image_count != 1:
            raise ValueError("gundam mode is for one image. Use base mode for multi-page OCR.")
        return GUNDAM
    if requested_mode == "base":
        return BASE
    raise ValueError(f"Unknown image mode: {requested_mode}")


def load_model(
    model_id: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
    adapter_path: str | Path | None = None,
):
    """Load the native MLX architecture at a reproducible revision and optional adapter."""

    try:
        from mlx_vlm import load
    except ImportError as exc:
        raise RuntimeError(
            "MLX-VLM is required. Install the project with `uv sync` or run "
            "`uv run unlimited-ocr-mlx --help`."
        ) from exc
    return load(model_id, revision=revision, adapter_path=str(adapter_path) if adapter_path else None)


def ocr_images(
    model: Any,
    processor: Any,
    images: Sequence[str | Path],
    *,
    prompt: str | None = None,
    requested_mode: str = "auto",
    max_tokens: int = 32768,
    no_repeat_ngram_size: int = 35,
    ngram_window: int | None = None,
    prefill_step_size: int | None = None,
) -> OCRResult:
    """Run a native MLX OCR request, preserving Unlimited-OCR's input contract."""

    image_paths = validate_images(images)
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    mode = mode_for(len(image_paths), requested_mode)
    if prompt is None:
        prompt = SINGLE_IMAGE_PROMPT if len(image_paths) == 1 else MULTI_PAGE_PROMPT
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted_prompt = apply_chat_template(
        processor, model.config, prompt, num_images=len(image_paths)
    )
    # Unlimited-OCR accepts one <image> marker for all pages.  The MLX-VLM
    # template implements that model-specific behavior; assert it so a future
    # incompatible template does not silently corrupt the visual token layout.
    if formatted_prompt.count("<image>") != 1:
        raise RuntimeError("Unlimited-OCR prompt must contain exactly one <image> token.")

    window = mode.ngram_window if ngram_window is None else ngram_window
    kwargs: dict[str, Any] = {
        "image": [str(path) for path in image_paths],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "verbose": False,
        "cropping": mode.cropping,
        "base_size": mode.base_size,
        "image_size": mode.image_size,
    }
    if prefill_step_size is not None:
        if prefill_step_size < 1:
            raise ValueError("prefill_step_size must be at least 1")
        kwargs["prefill_step_size"] = prefill_step_size
    if no_repeat_ngram_size:
        if no_repeat_ngram_size < 1 or window < 1:
            raise ValueError("no-repeat ngram size and window must be positive, or ngram size must be 0")
        kwargs["logits_processors"] = [
            SlidingWindowNoRepeatNgramProcessor(no_repeat_ngram_size, window)
        ]

    started = time.perf_counter()
    response = generate(model, processor, formatted_prompt, **kwargs)
    elapsed = time.perf_counter() - started
    return OCRResult(
        text=response.text.strip(),
        elapsed_seconds=elapsed,
        prompt_tokens=response.prompt_tokens,
        generation_tokens=response.generation_tokens,
        prompt_tps=response.prompt_tps,
        generation_tps=response.generation_tps,
        peak_memory_gb=response.peak_memory,
        finish_reason=response.finish_reason,
        image_count=len(image_paths),
        mode=mode.name,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace one UTF-8 file without leaving a partially written result."""

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def write_result(result: OCRResult, output: str | Path) -> tuple[Path, Path]:
    """Write the Markdown result and JSON metrics with atomic file replacement."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output, result.text + ("\n" if result.text else ""))
    metadata_path = output.with_suffix(".json")
    _atomic_write_text(metadata_path, json.dumps(result.metadata(), indent=2) + "\n")
    return output, metadata_path
