"""FastAPI endpoints for local receipt and invoice extraction."""
from __future__ import annotations

import io
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf as fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.inference import DEFAULT_MODEL_DIR, DocumentExtractor


app = FastAPI(
    title="Intelligent Document AI",
    description="Receipt and invoice field extraction using Tesseract OCR and a fine-tuned LayoutLMv3 model.",
    version="0.1.0",
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 20
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@lru_cache(maxsize=1)
def get_extractor() -> DocumentExtractor:
    """Create one model instance and reuse it for later requests."""
    return DocumentExtractor(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def _images_from_upload(filename: str, content: bytes) -> list[Image.Image]:
    suffix = Path(filename).suffix.lower()
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")

    if suffix == ".pdf":
        try:
            pdf = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ValueError("Could not read this PDF. Check that it is a valid, unencrypted PDF.") from exc
        if pdf.page_count == 0:
            raise ValueError("The PDF has no pages.")
        if pdf.page_count > MAX_PDF_PAGES:
            raise ValueError(f"PDF has {pdf.page_count} pages; the limit is {MAX_PDF_PAGES}.")
        pages = []
        for page in pdf:
            pixmap = page.get_pixmap(dpi=150, alpha=False)
            pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
        return pages

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("Unsupported file type. Upload a JPG, JPEG, PNG, or PDF.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            source.verify()
        with Image.open(io.BytesIO(content)) as source:
            return [source.convert("RGB")]
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Could not read this image. Check that the file is a valid JPG or PNG.") from exc


def _extract_document(filename: str, content: bytes) -> dict[str, Any]:
    pages = _images_from_upload(filename, content)
    extractor = get_extractor()
    results = []
    for page_number, image in enumerate(pages, start=1):
        result = extractor.extract(image)
        results.append({"page": page_number, **result})
    page_types = {page["document_type"] for page in results}
    document_type = next(iter(page_types)) if len(page_types) == 1 else "mixed"
    return {
        "filename": Path(filename).name,
        "document_type": document_type,
        "pages": results,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Report local prerequisites without loading the model weights."""
    model_dir = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
    supported_types = ["receipt"]
    config_path = model_dir / "config.json"
    model_files_present = config_path.is_file()
    if model_files_present:
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
            labels = [str(value) for value in model_config.get("id2label", {}).values()]
            if any(label.startswith(("B-INV_", "I-INV_")) for label in labels):
                supported_types.append("invoice")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return {
        "status": "ok",
        "supported_document_types": supported_types,
        "model_directory": str(model_dir),
        "model_files_present": model_files_present,
        "model_loaded": get_extractor.cache_info().currsize > 0,
        "tesseract_available": shutil.which("tesseract") is not None,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract receipt or invoice fields from one image or a multi-page PDF."""
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        return await run_in_threadpool(_extract_document, file.filename or "upload", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if "tesseract" in str(exc).lower():
            raise HTTPException(
                status_code=503,
                detail="Tesseract OCR is unavailable. Install it and make sure `tesseract` is on PATH.",
            ) from exc
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc


@app.post("/extract-batch")
async def extract_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Extract receipt or invoice fields from up to 10 image or PDF documents."""
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one document.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Upload no more than 10 documents per batch.")

    results = []
    for upload in files:
        filename = upload.filename or "upload"
        try:
            content = await upload.read(MAX_UPLOAD_BYTES + 1)
            result = await run_in_threadpool(_extract_document, filename, content)
            results.append(result)
        except (ValueError, FileNotFoundError) as exc:
            results.append({"filename": Path(filename).name, "error": str(exc)})
        except Exception as exc:
            results.append({"filename": Path(filename).name, "error": f"Extraction failed: {exc}"})
    return {"results": results}
