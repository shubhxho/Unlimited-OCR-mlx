import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from infer_mlx import output_for_single, parse_args, render_pdf
from unlimited_ocr_mlx import (
    BASE,
    GUNDAM,
    OCRResult,
    SlidingWindowNoRepeatNgramProcessor,
    collect_images,
    mode_for,
    write_result,
)


class TestUnlimitedOCRMLX(unittest.TestCase):
    def test_sliding_no_repeat_matches_upstream_rule(self):
        processor = SlidingWindowNoRepeatNgramProcessor(ngram_size=3, window=6)
        # The current prefix is (1, 2); prior 1,2,3 blocks continuation 3.
        self.assertEqual(processor.banned_tokens([1, 2, 3, 1, 2]), {3})
        self.assertEqual(processor.banned_tokens([1, 2]), set())
        self.assertEqual(
            SlidingWindowNoRepeatNgramProcessor(3, 6, whitelist_token_ids=[3]).banned_tokens([1, 2, 3, 1, 2]),
            set(),
        )

    def test_mode_selection_enforces_official_workflows(self):
        self.assertEqual(mode_for(1), GUNDAM)
        self.assertEqual(mode_for(2), BASE)
        self.assertEqual(mode_for(2, "base"), BASE)
        with self.assertRaises(ValueError):
            mode_for(2, "gundam")

    def test_collect_images_uses_natural_order_and_supported_extensions(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "page_10.png").touch()
            (root / "page_2.png").touch()
            (root / "ignored.txt").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "page_1.JPG").touch()
            self.assertEqual(
                [path.name for path in collect_images(root)],
                ["page_1.JPG", "page_2.png", "page_10.png"],
            )
            self.assertEqual(output_for_single(nested / "page_1.JPG", root, root / "out"), root / "out" / "nested" / "page_1.md")

    def test_result_writes_markdown_and_metrics(self):
        result = OCRResult(
            text="# parsed",
            elapsed_seconds=1.25,
            prompt_tokens=10,
            generation_tokens=5,
            prompt_tps=8.0,
            generation_tps=4.0,
            peak_memory_gb=2.0,
            finish_reason="eos",
            image_count=1,
            mode="gundam",
        )
        with tempfile.TemporaryDirectory() as root:
            markdown, metadata = write_result(result, Path(root) / "nested" / "page.md")
            self.assertEqual(markdown.read_text(), "# parsed\n")
            self.assertIn('"generation_tokens": 5', metadata.read_text())
            self.assertNotIn('"text"', metadata.read_text())

    def test_cli_requires_exactly_one_input_source(self):
        args = parse_args(["--image", "page.png"])
        self.assertEqual(args.image, Path("page.png"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([])
            with self.assertRaises(SystemExit):
                parse_args(["--image", "a.png", "--pdf", "b.pdf"])

    def test_render_pdf_preserves_requested_page_order(self):
        saved = []

        class FakePage:
            def __init__(self, number):
                self.number = number

            def get_pixmap(self, matrix, alpha):
                self.matrix, self.alpha = matrix, alpha
                return self

            def save(self, path):
                saved.append((self.number, Path(path)))
                Path(path).touch()

        class FakeDocument:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return FakePage(index + 1)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        fake_fitz = SimpleNamespace(open=lambda _: FakeDocument(), Matrix=lambda x, y: (x, y))
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"fitz": fake_fitz}):
            pdf = Path(root) / "book.pdf"
            pdf.touch()
            pages = render_pdf(pdf, Path(root) / "pages", dpi=144, first_page=2, last_page=3)
        self.assertEqual([number for number, _ in saved], [2, 3])
        self.assertEqual([path.name for path in pages], ["page_00002.png", "page_00003.png"])


if __name__ == "__main__":
    unittest.main()
