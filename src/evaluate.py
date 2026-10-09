"""Held-out evaluation metrics and BIO decoding helpers for receipt extraction."""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

FIELDS = ("COMPANY", "DATE", "ADDRESS", "TOTAL")


def bio_spans(tags: Sequence[str], document_index: int = 0) -> set[tuple[int, int, int, str]]:
    """Convert a BIO sequence to document-scoped half-open entity spans."""
    spans: set[tuple[int, int, int, str]] = set()
    active_label: str | None = None
    active_start: int | None = None

    def close(end: int) -> None:
        if active_label is not None and active_start is not None:
            spans.add((document_index, active_start, end, active_label))

    for position, tag in enumerate(tags):
        if tag == "O":
            close(position)
            active_label, active_start = None, None
        elif tag.startswith("B-"):
            close(position)
            active_label, active_start = tag[2:], position
        elif tag.startswith("I-") and active_label == tag[2:]:
            continue
        elif tag.startswith("I-"):
            # Treat malformed orphan/mismatched I-tags as a new span.
            close(position)
            active_label, active_start = tag[2:], position
        else:
            close(position)
            active_label, active_start = None, None
    close(len(tags))
    return spans


def entity_metrics(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    """Calculate micro precision, recall, and F1 over exact entity spans."""
    if len(predictions) != len(references):
        raise ValueError("Prediction and reference document counts must match.")
    predicted, gold = set(), set()
    for doc_index, (pred_tags, gold_tags) in enumerate(zip(predictions, references)):
        predicted.update(bio_spans(pred_tags, doc_index))
        gold.update(bio_spans(gold_tags, doc_index))
    tp = len(predicted & gold)
    fp, fn = len(predicted - gold), len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "entity_f1": f1,
        "true_positive_entities": tp,
        "false_positive_entities": fp,
        "false_negative_entities": fn,
    }


def fields_from_bio(words: Sequence[str], tags: Sequence[str]) -> dict[str, str]:
    """Join field BIO spans into one text string per receipt field."""
    if len(words) != len(tags):
        raise ValueError("Words and tags must have equal lengths.")
    spans: dict[str, list[str]] = {field: [] for field in FIELDS}
    active_field: str | None = None
    active_words: list[str] = []

    def finish() -> None:
        if active_field in spans and active_words:
            spans[active_field].append(" ".join(active_words))

    for word, tag in zip(words, tags):
        if tag == "O":
            finish()
            active_field, active_words = None, []
        elif tag.startswith("B-"):
            finish()
            active_field, active_words = tag[2:], [word]
        elif tag.startswith("I-") and active_field == tag[2:]:
            active_words.append(word)
        elif tag.startswith("I-"):
            finish()
            active_field, active_words = tag[2:], [word]
        else:
            finish()
            active_field, active_words = None, []
    finish()
    return {field: " ".join(spans[field]).strip() for field in FIELDS}


def normalize_field_value(value: Any) -> str:
    """Normalize case and punctuation for full-field exact-match evaluation."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def field_metrics(
    rows: Iterable[dict[str, Any]],
    prediction_key: str = "prediction",
    gold_key: str = "gold",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return aggregate and per-field exact-match, precision, recall, and F1."""
    rows = list(rows)
    counts = {field: {"correct": 0, "eligible": 0, "tp": 0, "fp": 0, "fn": 0}
              for field in FIELDS}
    exact_documents = 0
    for row in rows:
        doc_correct = True
        for field in FIELDS:
            pred = normalize_field_value(row[prediction_key].get(field, ""))
            gold = normalize_field_value(row[gold_key].get(field, ""))
            c = counts[field]
            if gold:
                c["eligible"] += 1
                if pred == gold:
                    c["correct"] += 1
                    c["tp"] += 1
                else:
                    doc_correct = False
                    c["fn"] += 1
                    if pred:
                        c["fp"] += 1
            elif pred:
                doc_correct = False
                c["fp"] += 1
        exact_documents += int(doc_correct)

    per_field = []
    total_correct = total_eligible = total_tp = total_fp = total_fn = 0
    for field, c in counts.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact = c["correct"] / c["eligible"] if c["eligible"] else 0.0
        per_field.append({
            "field": field, "exact_match_accuracy": exact,
            "precision": precision, "recall": recall, "f1": f1,
            "correct": c["correct"], "gold_present": c["eligible"],
        })
        total_correct += c["correct"]
        total_eligible += c["eligible"]
        total_tp += c["tp"]
        total_fp += c["fp"]
        total_fn += c["fn"]
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "field_exact_match_accuracy": total_correct / total_eligible if total_eligible else 0.0,
        "document_all_fields_exact_match": exact_documents / len(rows) if rows else 0.0,
        "field_value_precision": precision,
        "field_value_recall": recall,
        "field_value_f1": f1,
    }
    return summary, per_field
