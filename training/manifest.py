"""Strict, portable JSONL manifests for OCR fine-tuning and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_PROMPT = "document parsing."


@dataclass(frozen=True)
class OCRExample:
    """One licensed, single-page supervised OCR example.

    ``target`` is the canonical text/layout transcription. Keep raw images out
    of Git and use a source-document-level split so pages from the same work
    never cross train/validation/test.
    """

    id: str
    image: str
    target: str
    prompt: str = DEFAULT_PROMPT
    source_document: str = ""
    license: str = ""
    sha256: str = ""

    @classmethod
    def from_dict(cls, value: dict, line_number: int) -> "OCRExample":
        required = ("id", "image", "target")
        missing = [key for key in required if not isinstance(value.get(key), str) or not value[key].strip()]
        if missing:
            raise ValueError(f"Manifest line {line_number}: missing non-empty " + ", ".join(missing))
        prompt = value.get("prompt", DEFAULT_PROMPT)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Manifest line {line_number}: prompt must be a non-empty string")
        for key in ("source_document", "license", "sha256"):
            if key in value and not isinstance(value[key], str):
                raise ValueError(f"Manifest line {line_number}: {key} must be a string")
        return cls(
            id=value["id"], image=value["image"], target=value["target"], prompt=prompt,
            source_document=value.get("source_document", ""), license=value.get("license", ""),
            sha256=value.get("sha256", ""),
        )


def load_manifest(path: str | Path) -> list[OCRExample]:
    path = Path(path)
    examples: list[OCRExample] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Manifest line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number}: expected a JSON object")
            example = OCRExample.from_dict(value, line_number)
            if example.id in ids:
                raise ValueError(f"Manifest line {line_number}: duplicate id {example.id!r}")
            ids.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"Manifest is empty: {path}")
    return examples


def resolve_image(example: OCRExample, image_root: str | Path) -> Path:
    root = Path(image_root).resolve()
    image = (root / example.image).resolve()
    if root not in image.parents and image != root:
        raise ValueError(f"Example {example.id}: image path escapes image root")
    if not image.is_file():
        raise FileNotFoundError(f"Example {example.id}: image not found: {image}")
    return image


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_images(examples: list[OCRExample], image_root: str | Path, verify_hashes: bool = False) -> None:
    """Fail before training if an image is missing, escapes root, or changed."""

    for example in examples:
        image = resolve_image(example, image_root)
        if verify_hashes and example.sha256 and sha256_file(image) != example.sha256:
            raise ValueError(f"Example {example.id}: SHA-256 does not match manifest")


def write_manifest(examples: Iterator[OCRExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
