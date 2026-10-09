# Intelligent Document AI

A receipt and invoice information extraction project built with PyTorch, LayoutLMv3, Tesseract OCR, FastAPI, and Streamlit. The trained model is based on SROIE receipt annotations and the combined checkpoint supports receipt and invoice key-field extraction. Invoice line-item grouping and bank-statement extraction are not included.

## Features

- Extracts receipt fields and invoice key fields from JPG, JPEG, PNG, and PDF uploads.
- Uses Tesseract to produce OCR words and bounding boxes, then applies the fine-tuned LayoutLMv3 model.
- Uses a rule-based OCR fallback only when the model leaves a field blank; fallback sources are labeled in the API response.
- Offers single-file and batch FastAPI endpoints, plus a Streamlit interface for review, edits, and JSON/CSV export.
- Processes PDF pages as images and returns one result per page.

Review the extracted fields before using them. OCR or model predictions can be incomplete or incorrect.

## Project structure

```text
intelligent-document-ai/
├── api/                  # FastAPI upload endpoints
├── app/                  # Streamlit review interface
├── docs/                 # Architecture and design notes
├── notebooks/            # Colab preparation, OCR, training, and evaluation
├── src/                  # OCR, inference, model, and evaluation helpers
├── tests/                # Unit and API tests
├── models/               # Local model folder (kept out of Git)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 1. Prepare the trained model

Download the complete `layoutlmv3-multidoc-best` folder from Google Drive at `My Drive/intelligent-document-ai/models/`. Place it here:

```text
models/layoutlmv3-multidoc-best/
```

The folder needs the model weights, `config.json`, and processor/tokenizer files. Set `MODEL_DIR` if the folder is stored somewhere else. If this combined folder is absent, inference falls back to `models/layoutlmv3-sroie-best` for receipt-only use. The model weights are deliberately excluded from Git because they are large; obtain the combined checkpoint from the Colab training run.

## 2. Install local dependencies

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate        # Windows PowerShell
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

Install the Tesseract executable separately. On macOS, Homebrew may require Apple's separate Command Line Tools to build on older macOS versions; alternatively, install the prebuilt Conda-Forge package with `conda install -c conda-forge tesseract`. On Ubuntu/Debian, use `sudo apt-get install tesseract-ocr tesseract-ocr-eng`. Confirm the executable is visible with `which tesseract` and `tesseract --version`.

The local requirements pin PyTorch 2.2.2, torchvision 0.17.2, NumPy below 2, and Transformers 4.57.1 for macOS compatibility, including Intel Macs. Colab training uses its own notebook environment and may use a newer Transformers/PyTorch stack.

## 3. Run the application

Open two terminals at the project root. Keep the virtual environment active in both.

Terminal 1 — API:

```bash
./.venv/bin/python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
./.venv/bin/python -m streamlit run app/streamlit_app.py
```

Open the Streamlit URL printed in Terminal 2. FastAPI health information is at `http://127.0.0.1:8000/health`; interactive API documentation is at `http://127.0.0.1:8000/docs`.

`/extract` accepts one image or PDF. `/extract-batch` accepts up to 10 documents. The current service limits each file to 25 MB and each PDF to 20 pages. The model loads on the first extraction request and runs locally; a Colab session is not needed.

## 4. Verify the setup

Check that Python can find Tesseract:

```bash
./.venv/bin/python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

With the model folder in place, verify model loading:

```bash
./.venv/bin/python -c "from src.inference import DocumentExtractor; model = DocumentExtractor(); print('Loaded:', model.model_dir, '| device:', model.device)"
```

Then upload a clear receipt image through Streamlit or use `POST /extract` in `/docs`. Check the four output fields against the receipt. A warning indicates if fallback rules filled a blank model field or if the token limit truncated text.

Run automated unit and API checks with Python's built-in `unittest` runner:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

These tests use a fake model for the endpoint/inference paths and do not measure model accuracy. Manually smoke-test the Streamlit page by uploading a receipt, editing a field, and downloading JSON and CSV.

## 5. Run with Docker

Install Docker Desktop, ensure the model folder is in place, and run from the project root:

```bash
docker compose up --build
```

Open Streamlit at `http://localhost:8501`, API health at `http://localhost:8000/health`, and API docs at `http://localhost:8000/docs`. The model directory is mounted read-only and is not baked into the image. Stop the containers with Ctrl+C, then use `docker compose down` to remove them.

## Evaluation

Run `notebooks/04_model_evaluation.ipynb` in Colab to generate held-out test metrics and error reports. Add the actual output values here after the notebook completes; do not infer or invent accuracy numbers. The evaluation excludes exact train/validation image duplicates and reports the exclusion audit. Model results are meaningful only with the evaluated sample count and the notebook's assumptions.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the request flow, processing limits, and field-source behavior.

## GitHub setup

After reviewing `.gitignore`, create an empty repository on GitHub and add its remote:

```bash
git init -b main
git add .
git status --short
git commit -m "Build receipt document extraction app"
git remote add origin <your-github-repository-url>
git push -u origin main
```

Before pushing, confirm `git status` does not list model weights, local datasets, credentials, or `.venv/`. The model weights must be transferred separately from Google Drive.

## Resume bullet drafts

Use only claims you can demonstrate and add measured metrics after Phase 4 is complete:

- Built a receipt information extraction pipeline using Tesseract OCR, LayoutLMv3 token classification, and the SROIE dataset; returns Company, Date, Address, and Total fields for image and PDF uploads.
- Developed a local FastAPI and Streamlit application for batch document inference, human review, field correction, and JSON/CSV export.
- Evaluated the held-out receipt test set against a rule-based baseline; insert the actual entity F1, field exact-match score, and sample count from the saved evaluation report.
