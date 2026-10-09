"""LayoutLMv3 model and processor helpers for the receipt token-classification task."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from transformers import AutoProcessor, LayoutLMv3ForTokenClassification


def build_model(
    model_name: str,
    label2id: Mapping[str, int],
    id2label: Mapping[int, str],
) -> LayoutLMv3ForTokenClassification:
    """Load pretrained LayoutLMv3 weights with a task-specific token head."""
    return LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=dict(label2id),
        id2label={int(key): value for key, value in id2label.items()},
        ignore_mismatched_sizes=True,
    )


def load_processor(model_name_or_path: str | Path):
    """Load processor with built-in OCR disabled; pass words and boxes explicitly."""
    return AutoProcessor.from_pretrained(model_name_or_path, apply_ocr=False)


def save_model_bundle(
    model,
    processor,
    output_dir: str | Path,
    label2id: Mapping[str, int],
    id2label: Mapping[int, str],
    training_config: Mapping[str, object] | None = None,
) -> Path:
    """Save model weights/config, processor, labels, and training configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {"label2id": dict(label2id), "id2label": {str(k): v for k, v in id2label.items()}},
            indent=2,
        ),
        encoding="utf-8",
    )
    if training_config is not None:
        (output_dir / "training_config.json").write_text(
            json.dumps(dict(training_config), indent=2), encoding="utf-8"
        )
    return output_dir


def load_model_bundle(model_dir: str | Path):
    """Reload a saved token classifier and its matching processor."""
    model_dir = Path(model_dir)
    processor = load_processor(model_dir)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    return model, processor
