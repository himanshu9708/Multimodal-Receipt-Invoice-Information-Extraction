# Architecture

```text
Browser / Streamlit
        |
        | multipart JPG, PNG, or PDF upload
        v
FastAPI (/extract or /extract-batch)
        |
        +--> PyMuPDF: render PDF pages to images
        +--> Tesseract: OCR words and bounding boxes
        +--> LayoutLMv3: predict BIO tags for OCR words
        +--> rule fallback: fill blank fields from OCR text
        |
        v
Structured receipt fields + word tags + warnings
        |
        v
Streamlit review/edit table -> JSON or CSV download
```

The service prefers the combined checkpoint at models/layoutlmv3-multidoc-best (or MODEL_DIR) and falls back to models/layoutlmv3-sroie-best when the combined checkpoint is absent. The combined model supports SROIE receipts and DocILE invoice key fields; it does not group invoice line items. Invoice pages are identified from predicted invoice labels or invoice text. Receipt OCR fallback runs only for receipt pages. The API reports each field source as layoutlmv3, ocr_rule_fallback, or not_found. Users should review extracted fields before using exports.

PDF uploads are rendered one page at a time. Batch requests are limited to 10 documents; each file is limited to 25 MB and each PDF to 20 pages. The current local API uses one model instance and processes pages sequentially.
