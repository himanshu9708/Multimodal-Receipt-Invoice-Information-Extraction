"""Image preprocessing and OCR helpers shared by the Colab and local pipelines."""
from __future__ import annotations

from typing import Any, Sequence


def normalize_bbox(bbox: Sequence[float], image_size: tuple[int, int]) -> list[int]:
    """Convert pixel xyxy coordinates to LayoutLM's 0–1000 range."""
    if len(bbox) != 4:
        raise ValueError(f"Expected an xyxy box with four values, got {bbox!r}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {image_size!r}")
    x0, y0, x1, y1 = bbox
    result = [
        round(1000 * x0 / width), round(1000 * y0 / height),
        round(1000 * x1 / width), round(1000 * y1 / height),
    ]
    result = [max(0, min(1000, value)) for value in result]
    if result[0] > result[2] or result[1] > result[3]:
        raise ValueError(f"Invalid xyxy box ordering: {bbox!r}")
    return result


def preprocess_receipt_image(image: Any, min_long_edge: int = 1800) -> Any:
    """Grayscale, autocontrast, and modestly upscale a PIL receipt image."""
    from PIL import ImageEnhance, ImageOps

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    if max(gray.size) < min_long_edge:
        scale = min_long_edge / max(gray.size)
        gray = gray.resize((round(gray.width * scale), round(gray.height * scale)))
    return ImageEnhance.Contrast(gray).enhance(1.15)


def image_to_ocr_tokens(
    image: Any,
    lang: str = "eng",
    config: str = "--oem 3 --psm 6",
) -> list[dict[str, Any]]:
    """Run pytesseract and return text, confidence, pixel, and normalized boxes."""
    from PIL import Image
    import pytesseract
    from pytesseract import Output

    original = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    processed = preprocess_receipt_image(original)
    data = pytesseract.image_to_data(
        processed, lang=lang, config=config, output_type=Output.DICT
    )
    scale_x = original.width / processed.width
    scale_y = original.height / processed.height
    tokens = []
    for index, value in enumerate(data["text"]):
        text = value.strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if not text or confidence < 0:
            continue
        left = round(data["left"][index] * scale_x)
        top = round(data["top"][index] * scale_y)
        right = round((data["left"][index] + data["width"][index]) * scale_x)
        bottom = round((data["top"][index] + data["height"][index]) * scale_y)
        pixel_box = [left, top, right, bottom]
        tokens.append({
            "text": text,
            "confidence": confidence,
            "bbox": normalize_bbox(pixel_box, original.size),
            "pixel_bbox": pixel_box,
            "line_id": (
                data["block_num"][index], data["par_num"][index], data["line_num"][index]
            ),
        })
    return tokens


def group_ocr_tokens_into_lines(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine OCR tokens by Tesseract line identifiers in reading order."""
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for token in tokens:
        grouped.setdefault(tuple(token["line_id"]), []).append(token)
    lines = []
    for line_tokens in grouped.values():
        line_tokens.sort(key=lambda item: item["pixel_bbox"][0])
        lines.append({
            "text": " ".join(item["text"] for item in line_tokens),
            "bbox": [
                min(item["bbox"][0] for item in line_tokens),
                min(item["bbox"][1] for item in line_tokens),
                max(item["bbox"][2] for item in line_tokens),
                max(item["bbox"][3] for item in line_tokens),
            ],
            "confidence": sum(item["confidence"] for item in line_tokens) / len(line_tokens),
        })
    return sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def validate_document_record(record, label_names: set[str] | None = None) -> None:
    """Raise ValueError if a prepared words/boxes/tags record is malformed."""
    required = {"document_id", "split", "image_path", "words", "bboxes", "ner_tags"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Record is missing fields: {sorted(missing)}")
    words, boxes, tags = record["words"], record["bboxes"], record["ner_tags"]
    if not (len(words) == len(boxes) == len(tags)):
        raise ValueError("words, bboxes, and ner_tags must have equal lengths")
    for index, (word, box, tag) in enumerate(zip(words, boxes, tags)):
        if not isinstance(word, str) or not word.strip():
            raise ValueError(f"Empty or invalid word at index {index}")
        if len(box) != 4:
            raise ValueError(f"Box at index {index} must contain four values")
        x0, y0, x1, y1 = box
        if not (0 <= x0 <= x1 <= 1000 and 0 <= y0 <= y1 <= 1000):
            raise ValueError(f"Invalid normalized box at index {index}: {box}")
        if label_names is not None and tag not in label_names:
            raise ValueError(f"Unknown BIO tag at index {index}: {tag!r}")
