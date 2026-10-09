"""PyTorch data and loop helpers for LayoutLMv3 receipt token classification."""
from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReceiptTokenDataset(Dataset):
    """On-demand image/word/box/label encoding for one SROIE split."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        processor,
        label2id: Mapping[str, int],
        max_length: int = 512,
    ):
        self.records = records
        self.processor = processor
        self.label2id = dict(label2id)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        image = Image.open(row["image_path"]).convert("RGB")
        labels = [self.label2id[label] for label in row["ner_tags"]]
        encoding = self.processor(
            images=image,
            text=row["words"],
            boxes=row["bboxes"],
            word_labels=labels,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoding.items()}


def make_dataloader(
    records: Sequence[Mapping[str, Any]],
    processor,
    label2id: Mapping[str, int],
    batch_size: int = 2,
    max_length: int = 512,
    shuffle: bool = False,
) -> DataLoader:
    """Build a simple DataLoader; workers default to zero for Colab stability."""
    dataset = ReceiptTokenDataset(records, processor, label2id, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def token_ids_to_sequences(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    id2label: Mapping[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Remove ignored positions and convert predictions/labels to BIO tag strings."""
    pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    gold_ids = label_ids.detach().cpu().tolist()
    predictions, references = [], []
    for pred_row, gold_row in zip(pred_ids, gold_ids):
        pred_tags, gold_tags = [], []
        for predicted_id, gold_id in zip(pred_row, gold_row):
            if gold_id == -100:
                continue
            pred_tags.append(id2label[int(predicted_id)])
            gold_tags.append(id2label[int(gold_id)])
        predictions.append(pred_tags)
        references.append(gold_tags)
    return predictions, references


def _bio_spans(tags: Sequence[str], document_index: int) -> set[tuple[int, int, int, str]]:
    """Convert one BIO sequence to document-scoped half-open entity spans."""
    spans: set[tuple[int, int, int, str]] = set()
    active_label: str | None = None
    active_start: int | None = None

    def close(end_index: int) -> None:
        if active_label is not None and active_start is not None:
            spans.add((document_index, active_start, end_index, active_label))

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
            close(position)
            active_label, active_start = tag[2:], position
        else:
            close(position)
            active_label, active_start = None, None
    close(len(tags))
    return spans


def exact_entity_metrics(
    predictions: Sequence[Sequence[str]], references: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Calculate micro precision, recall, and F1 over exact BIO entity spans."""
    predicted_spans, reference_spans = set(), set()
    for index, (predicted, reference) in enumerate(zip(predictions, references)):
        predicted_spans.update(_bio_spans(predicted, index))
        reference_spans.update(_bio_spans(reference, index))
    true_positive = len(predicted_spans & reference_spans)
    false_positive = len(predicted_spans - reference_spans)
    false_negative = len(reference_spans - predicted_spans)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "entity_f1": f1}


def evaluate_token_classifier(
    model,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Mapping[int, str],
) -> dict[str, float]:
    """Calculate mean validation loss and exact BIO span precision/recall/F1."""
    model.eval()
    total_loss, n_documents = 0.0, 0
    predictions, references = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size = batch["input_ids"].shape[0]
            total_loss += float(outputs.loss.detach().float().cpu()) * batch_size
            n_documents += batch_size
            pred, gold = token_ids_to_sequences(outputs.logits, batch["labels"], id2label)
            predictions.extend(pred)
            references.extend(gold)
    metrics = exact_entity_metrics(predictions, references)
    metrics["val_loss"] = total_loss / max(n_documents, 1)
    return metrics


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    accumulation_steps: int = 1,
    scaler=None,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
) -> float:
    """Run one epoch with optional AMP scaler and gradient accumulation."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss, documents = 0.0, 0
    for step, batch in enumerate(dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}
        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(dataloader) - group_start)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp and device.type == "cuda" else nullcontext()
        )
        with amp_context:
            outputs = model(**batch)
            loss = outputs.loss
        scaled_loss = loss / group_size
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        batch_size = batch["input_ids"].shape[0]
        running_loss += float(loss.detach().float().cpu()) * batch_size
        documents += batch_size
        should_update = (step + 1) % accumulation_steps == 0 or step + 1 == len(dataloader)
        if should_update:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return running_loss / max(documents, 1)
