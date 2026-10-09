"""Rule-based receipt field extraction and exact-match evaluation helpers."""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

FIELDS = ("COMPANY", "DATE", "ADDRESS", "TOTAL")
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
    r"\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}|"
    r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:RM|MYR|USD|\$|£|€)?\s*-?\d{1,3}(?:,\d{3})*(?:\.\d{2})(?!\d)",
    re.IGNORECASE,
)
ADDRESS_HINTS = ("JALAN", "JLN", "STREET", "ROAD", "RD.", "AVENUE", "AVE", "LORONG", "TAMAN", "SUITE", "NO.")
TOTAL_HINTS = ("GRAND TOTAL", "TOTAL", "AMOUNT DUE", "NET", "BALANCE")


def reconstruct_gold_fields(words: Sequence[str], tags: Sequence[str]) -> dict[str, str]:
    """Join BIO spans into reference field strings for evaluation."""
    if len(words) != len(tags):
        raise ValueError("words and tags must have equal lengths")
    spans: dict[str, list[str]] = {field: [] for field in FIELDS}
    current_field: str | None = None
    current_words: list[str] = []

    def finish() -> None:
        if current_field in spans and current_words:
            spans[current_field].append(" ".join(current_words))

    for word, tag in zip(words, tags):
        if tag == "O":
            finish()
            current_field, current_words = None, []
        elif tag.startswith("B-"):
            finish()
            current_field = tag[2:]
            current_words = [word]
        elif tag.startswith("I-") and current_field == tag[2:]:
            current_words.append(word)
        else:
            finish()
            current_field, current_words = None, []
    finish()
    return {field: " ".join(spans[field]).strip() for field in FIELDS}


def _group_lines(tokens: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for token in tokens:
        grouped.setdefault(tuple(token["line_id"]), []).append(token)
    lines = []
    for items in grouped.values():
        items = sorted(items, key=lambda item: item["pixel_bbox"][0])
        lines.append({
            "text": " ".join(item["text"] for item in items),
            "bbox": [
                min(item["bbox"][0] for item in items),
                min(item["bbox"][1] for item in items),
                max(item["bbox"][2] for item in items),
                max(item["bbox"][3] for item in items),
            ],
        })
    return sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def extract_baseline_fields(tokens: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Extract four receipt fields using simple text and position heuristics."""
    lines = _group_lines(tokens)
    result = {field: "" for field in FIELDS}
    if not lines:
        return result

    excluded = ("RECEIPT", "INVOICE", "TAX INVOICE", "CASH BILL")
    for line in lines:
        text = line["text"].strip()
        if line["bbox"][1] <= 300 and sum(c.isalpha() for c in text) >= 3:
            if not any(cue in text.upper() for cue in excluded):
                result["COMPANY"] = text
                break

    for line in lines:
        match = DATE_PATTERN.search(line["text"])
        if match:
            result["DATE"] = match.group(0).strip()
            break

    address_candidates = []
    for line in lines:
        text, upper = line["text"].strip(), line["text"].upper()
        has_hint = any(hint in upper for hint in ADDRESS_HINTS)
        has_postcode = bool(re.search(r"\b\d{5}\b", text))
        has_letters_digits = any(c.isalpha() for c in text) and any(c.isdigit() for c in text)
        if line["bbox"][1] < 650 and has_letters_digits and (has_hint or has_postcode):
            address_candidates.append((int(has_hint) + int(has_postcode), len(text), text))
    if address_candidates:
        result["ADDRESS"] = max(address_candidates)[2]

    amount_candidates = []
    for line in lines:
        for match in AMOUNT_PATTERN.finditer(line["text"]):
            cue = int(any(hint in line["text"].upper() for hint in TOTAL_HINTS))
            amount_candidates.append((cue, line["bbox"][1], match.group(0).strip()))
    if amount_candidates:
        result["TOTAL"] = sorted(amount_candidates, key=lambda value: (value[0], value[1]))[-1][2]
    return result


def normalize_for_match(value: str) -> str:
    """Lowercase and retain only ASCII letters and digits for exact comparisons."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def evaluate_field_exact_match(
    predictions: Iterable[dict[str, Any]],
    fields: Sequence[str] = FIELDS,
) -> list[dict[str, Any]]:
    """Compute field exact-match accuracy and exact-value precision/recall/F1."""
    counts = {field: {"correct": 0, "eligible": 0, "tp": 0, "fp": 0, "fn": 0}
              for field in fields}
    for row in predictions:
        for field in fields:
            pred = normalize_for_match(row["prediction"].get(field, ""))
            gold = normalize_for_match(row["gold"].get(field, ""))
            values = counts[field]
            if gold:
                values["eligible"] += 1
                if pred == gold:
                    values["correct"] += 1
                    values["tp"] += 1
                else:
                    values["fn"] += 1
                    if pred:
                        values["fp"] += 1
            elif pred:
                values["fp"] += 1

    metrics = []
    for field, values in counts.items():
        precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        recall = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact = values["correct"] / values["eligible"] if values["eligible"] else 0.0
        metrics.append({
            "field": field, "exact_match_accuracy": exact,
            "precision": precision, "recall": recall, "f1": f1,
            "correct": values["correct"], "gold_present": values["eligible"],
        })
    return metrics
