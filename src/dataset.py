"""Reader for prepared Phase 1 JSONL records (no training dependencies)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

try:
    from .preprocessing import validate_document_record
except ImportError:  # Allows running this file from the project src directory.
    from preprocessing import validate_document_record


def read_prepared_jsonl(
    path: str | Path, label_names: set[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Yield validated prepared records from one split's JSONL file."""
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}") from exc
            try:
                validate_document_record(record, label_names=label_names)
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError(f"Invalid record at {jsonl_path}:{line_number}: {exc}") from exc
            if not Path(record["image_path"]).is_file():
                raise FileNotFoundError(
                    f"Image for record {record['document_id']!r} is missing: {record['image_path']}"
                )
            yield record
