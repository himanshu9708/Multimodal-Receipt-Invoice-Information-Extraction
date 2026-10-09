"""Streamlit review interface for local receipt and invoice extraction."""
from __future__ import annotations

import io
import json
import os

import pymupdf as fitz
import pandas as pd
import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Intelligent Document AI", page_icon="🧾", layout="wide")
st.title("Intelligent Document AI")
st.caption("Receipt and invoice extraction with Tesseract OCR and fine-tuned LayoutLMv3")
st.info(
    "The combined checkpoint supports receipt fields and invoice key fields. "
    "Invoice line-item grouping is not included yet. Review and correct extracted values before export."
)

api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
try:
    health_response = requests.get(f"{api_url}/health", timeout=5)
    health_response.raise_for_status()
    health = health_response.json()
    if not health.get("model_files_present"):
        st.warning(f"Model files were not found at `{health['model_directory']}`.")
    supported_types = health.get("supported_document_types", ["receipt"])
    st.caption("Loaded model supports: " + ", ".join(supported_types))
    if "invoice" not in supported_types:
        st.warning("Invoice extraction is unavailable until layoutlmv3-multidoc-best is copied into the models folder.")
    if not health.get("tesseract_available"):
        st.warning("Tesseract was not found by the API. Install it and restart the API.")
except requests.RequestException:
    st.error(f"Cannot reach the API at {api_url}. Start FastAPI in another terminal first.")


def show_preview(upload) -> None:
    """Display the first page of a PDF or the uploaded image."""
    try:
        content = upload.getvalue()
        if upload.name.lower().endswith(".pdf"):
            pdf = fitz.open(stream=content, filetype="pdf")
            if pdf.page_count:
                pixmap = pdf[0].get_pixmap(dpi=120, alpha=False)
                preview = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                st.image(preview, caption=f"{upload.name} — page 1 preview", use_container_width=True)
        else:
            st.image(content, caption=upload.name, use_container_width=True)
    except Exception as exc:
        st.warning(f"Preview unavailable for {upload.name}: {exc}")


uploads = st.file_uploader(
    "Upload receipt or invoice images or PDFs",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
    help="The model detects receipt or invoice pages. Upload up to 10 files; PDFs may contain up to 20 pages.",
)

if uploads:
    preview_columns = st.columns(min(3, len(uploads)))
    for index, upload in enumerate(uploads):
        with preview_columns[index % len(preview_columns)]:
            show_preview(upload)

if st.button("Extract fields", type="primary", disabled=not uploads):
    if len(uploads) > 10:
        st.error("Upload no more than 10 files per request.")
    else:
        request_files = [
            ("files", (upload.name, upload.getvalue(), upload.type or "application/octet-stream"))
            for upload in uploads
        ]
        try:
            with st.spinner("Running OCR and document extraction…"):
                response = requests.post(
                    f"{api_url}/extract-batch", files=request_files, timeout=600
                )
            response.raise_for_status()
            st.session_state["extraction_response"] = response.json()
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"Extraction request failed: {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not contact the local API: {exc}")

response_data = st.session_state.get("extraction_response")
if response_data:
    st.subheader("Review and edit extracted fields")
    rows = []
    for document in response_data.get("results", []):
        if document.get("error"):
            st.error(f"{document.get('filename', 'Document')}: {document['error']}")
            continue
        for page in document.get("pages", []):
            if page.get("warning"):
                st.warning(f"{document['filename']} — page {page['page']}: {page['warning']}")
            rows.append({
                "filename": document["filename"],
                "page": page["page"],
                "document_type": page.get("document_type", document.get("document_type", "unknown")),
                **page.get("fields", {}),
            })

    if rows:
        editable = st.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            disabled=["filename", "page", "document_type"],
            key="reviewed_fields",
        )
        records = editable.to_dict(orient="records")
        st.download_button(
            "Download reviewed results as JSON",
            data=json.dumps(records, ensure_ascii=False, indent=2),
            file_name="document_extractions.json",
            mime="application/json",
        )
        st.download_button(
            "Download reviewed results as CSV",
            data=editable.to_csv(index=False).encode("utf-8-sig"),
            file_name="document_extractions.csv",
            mime="text/csv",
        )

    with st.expander("Raw model output"):
        st.json(response_data)
