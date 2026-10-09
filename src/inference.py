"""Local OCR and LayoutLMv3 inference for receipts and invoices."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

try:
    from .preprocessing import image_to_ocr_tokens
    from .postprocessing import FIELDS, extract_baseline_fields
except ImportError:  # Supports running from the repository root as a script.
    from preprocessing import image_to_ocr_tokens
    from postprocessing import FIELDS, extract_baseline_fields


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIDOC_MODEL_DIR = PROJECT_ROOT / "models" / "layoutlmv3-multidoc-best"
RECEIPT_MODEL_DIR = PROJECT_ROOT / "models" / "layoutlmv3-sroie-best"


def _default_model_dir() -> Path:
    """Prefer the combined checkpoint, falling back to the receipt checkpoint."""
    if (MULTIDOC_MODEL_DIR / "config.json").is_file():
        return MULTIDOC_MODEL_DIR
    if (RECEIPT_MODEL_DIR / "config.json").is_file():
        return RECEIPT_MODEL_DIR
    return MULTIDOC_MODEL_DIR


DEFAULT_MODEL_DIR = _default_model_dir()
INVOICE_PREFIX = "INV_"


def _label_field(tag: str) -> str | None:
    """Return the entity name from a BIO label, or None for O/special labels."""
    if tag == "O" or "-" not in tag:
        return None
    prefix, field = tag.split("-", 1)
    if prefix not in {"B", "I"} or not field:
        return None
    return field


def _reconstruct_fields(words: list[str], tags: list[str]) -> dict[str, str]:
    """Join predicted BIO spans into fields, retaining separate repeated spans."""
    spans: dict[str, list[str]] = {}
    active_field: str | None = None
    active_words: list[str] = []

    def finish() -> None:
        nonlocal active_field, active_words
        if active_field and active_words:
            spans.setdefault(active_field, []).append(" ".join(active_words))
        active_field, active_words = None, []

    for word, tag in zip(words, tags):
        field = _label_field(tag)
        if field is None:
            finish()
        elif tag.startswith("B-") or field != active_field:
            finish()
            active_field, active_words = field, [word]
        else:
            active_words.append(word)
    finish()
    return {field: " ".join(values).strip() for field, values in spans.items()}


def _schema_fields(id2label: dict[int, str], invoice: bool) -> list[str]:
    raw_fields = {
        field
        for tag in id2label.values()
        if (field := _label_field(tag)) is not None
        and field.startswith(INVOICE_PREFIX) == invoice
    }
    if invoice:
        return sorted(field[len(INVOICE_PREFIX):].lower() for field in raw_fields)
    return sorted(raw_fields)


class DocumentExtractor:
    """Load one receipt/invoice model once and extract fields from document images."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        configured_dir = model_dir or os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
        self.model_dir = Path(configured_dir).expanduser()
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Model config not found in {self.model_dir}. Copy "
                "layoutlmv3-multidoc-best there, or set MODEL_DIR to the model folder."
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, apply_ocr=False, local_files_only=True
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = min(512, int(self.model.config.max_position_embeddings))
        self.id2label = {
            int(index): str(label) for index, label in self.model.config.id2label.items()
        }
        self.invoice_fields = _schema_fields(self.id2label, invoice=True)
        self.receipt_fields = _schema_fields(self.id2label, invoice=False) or list(FIELDS)
        self.supported_document_types = (
            ["receipt", "invoice"] if self.invoice_fields else ["receipt"]
        )

    def extract(self, image: Image.Image) -> dict[str, Any]:
        """Run OCR and infer receipt or invoice fields from one RGB page."""
        image = image.convert("RGB")
        ocr_tokens = image_to_ocr_tokens(image)
        words = [token["text"] for token in ocr_tokens]
        boxes = [token["bbox"] for token in ocr_tokens]

        if not words:
            return {
                "document_type": "unknown",
                "fields": {},
                "field_sources": {},
                "tokens": [],
                "truncated": False,
                "warning": "No text was detected. Try a sharper, well-lit image.",
            }

        encoded = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits[0]
        predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()

        # Use the first subword prediction, matching the training label convention.
        predictions_by_word: dict[int, str] = {}
        for word_id, label_id in zip(word_ids, predicted_ids):
            if word_id is not None and word_id not in predictions_by_word:
                predictions_by_word[word_id] = self.id2label[int(label_id)]

        visible_words: list[str] = []
        visible_tags: list[str] = []
        output_tokens: list[dict[str, Any]] = []
        for word_id, token in enumerate(ocr_tokens):
            tag = predictions_by_word.get(word_id)
            if tag is None:
                continue
            visible_words.append(token["text"])
            visible_tags.append(tag)
            output_tokens.append({
                "text": token["text"],
                "bbox": token["bbox"],
                "confidence": round(float(token["confidence"]), 2),
                "tag": tag,
            })

        raw_fields = _reconstruct_fields(visible_words, visible_tags)
        invoice_fields = getattr(self, "invoice_fields", _schema_fields(self.id2label, invoice=True))
        receipt_fields = getattr(self, "receipt_fields", _schema_fields(self.id2label, invoice=False) or list(FIELDS))
        invoice_predicted = any(field.startswith(INVOICE_PREFIX) for field in raw_fields)
        invoice_text_hint = bool(
            re.search(r"\b(?:TAX\s+)?INVOICE\b", " ".join(words), flags=re.IGNORECASE)
        )
        receipt_predicted = any(not field.startswith(INVOICE_PREFIX) for field in raw_fields)
        document_type = (
            "invoice" if invoice_predicted or (invoice_text_hint and not receipt_predicted)
            else "receipt"
        )

        fields: dict[str, str] = {}
        field_sources: dict[str, str] = {}
        if document_type == "invoice":
            for field in invoice_fields:
                raw_name = INVOICE_PREFIX + field.upper()
                value = raw_fields.get(raw_name, "").strip()
                fields[field] = value
                field_sources[field] = "layoutlmv3" if value else "not_found"
        else:
            for field in receipt_fields:
                value = raw_fields.get(field, "").strip()
                fields[field] = value
                field_sources[field] = "layoutlmv3" if value else "not_found"

            # Preserve the receipt-only OCR fallback for blank receipt fields.
            baseline_fields = extract_baseline_fields(ocr_tokens)
            for field in receipt_fields:
                fallback_value = baseline_fields.get(field, "").strip()
                if not fields.get(field) and fallback_value:
                    fields[field] = fallback_value
                    field_sources[field] = "ocr_rule_fallback"

        warnings = []
        fallback_fields = [
            field for field, source in field_sources.items()
            if source == "ocr_rule_fallback"
        ]
        if fallback_fields:
            warnings.append(
                "Rule-based OCR fallback filled: " + ", ".join(fallback_fields)
                + ". Review these values."
            )
        if len(visible_words) < len(words):
            warnings.append(
                f"The model input limit truncated {len(words) - len(visible_words)} "
                "OCR word(s). Results may omit text near the end of the document."
            )
        return {
            "document_type": document_type,
            "fields": fields,
            "field_sources": field_sources,
            "tokens": output_tokens,
            "truncated": len(visible_words) < len(words),
            "warning": " ".join(warnings) or None,
        }


# Keep the old import name working for existing code and scripts.
ReceiptExtractor = DocumentExtractor
