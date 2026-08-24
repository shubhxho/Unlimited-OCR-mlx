"""Adapter save/load support for Unlimited-OCR's MoE decoder.

MLX-VLM's generic adapter loader assumes a top-level ``model.layers`` layout
and cannot reconstruct Unlimited-OCR's language-model ``SwitchLinear`` experts.
This module replays the exact same LoRA replacement on ``language_model``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ADAPTER_ARCHITECTURE = "unlimited-ocr-mlx-lora-v1"


def install_unlimited_ocr_lora(model: Any, rank: int, alpha: float, dropout: float, num_layers: int = -1) -> dict:
    """Freeze the base and install LoRA on all decoder modules, including MoE."""

    from mlx_vlm.trainer.adapter_utils import linear_to_lora_layers

    model.freeze()
    parameters = {"rank": rank, "scale": alpha / rank, "dropout": dropout}
    linear_to_lora_layers(model.language_model, num_layers=num_layers, config=parameters)
    config = {
        "architecture": ADAPTER_ARCHITECTURE,
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": parameters,
    }
    model.config.lora = config
    return config


def load_unlimited_ocr_adapter(model: Any, adapter_path: str | Path) -> Any:
    """Apply an adapter emitted by :func:`install_unlimited_ocr_lora`."""

    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapters.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Adapter directory must contain adapter_config.json and adapters.safetensors")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architecture") != ADAPTER_ARCHITECTURE:
        raise ValueError(
            "This is not an Unlimited-OCR MLX adapter. Use a matching adapter "
            f"with architecture={ADAPTER_ARCHITECTURE!r}."
        )
    parameters = config.get("lora_parameters")
    if not isinstance(parameters, dict) or not {"rank", "scale", "dropout"} <= parameters.keys():
        raise ValueError("Adapter has invalid lora_parameters")
    from mlx_vlm.trainer.adapter_utils import linear_to_lora_layers

    linear_to_lora_layers(
        model.language_model,
        num_layers=int(config.get("num_layers", -1)),
        config=parameters,
    )
    model.config.lora = config
    model.load_weights(str(weights_path), strict=False)
    model.eval()
    return model
