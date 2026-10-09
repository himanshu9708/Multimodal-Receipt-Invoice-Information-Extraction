import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from api import main


class FakeExtractor:
    def extract(self, image):
        return {
            "document_type": "receipt",
            "fields": {
                "COMPANY": "Sample Shop", "DATE": "12/31/2025",
                "ADDRESS": "1 Main St", "TOTAL": "$4.50",
            },
            "field_sources": {
                "COMPANY": "layoutlmv3", "DATE": "ocr_rule_fallback",
                "ADDRESS": "layoutlmv3", "TOTAL": "layoutlmv3",
            },
            "tokens": [], "truncated": False, "warning": None,
        }


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class ApiTests(unittest.TestCase):
    def test_health_reports_receipt_scope_and_dependencies(self):
        with patch.object(main.shutil, "which", return_value="/usr/bin/tesseract"):
            response = TestClient(main.app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["supported_document_types"], ["receipt"])
        self.assertTrue(response.json()["tesseract_available"])

    def test_extract_accepts_image_and_returns_fields(self):
        with patch.object(main, "get_extractor", return_value=FakeExtractor()):
            response = TestClient(main.app).post(
                "/extract", files={"file": ("receipt.png", png_bytes(), "image/png")}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["filename"], "receipt.png")
        self.assertEqual(response.json()["pages"][0]["fields"]["DATE"], "12/31/2025")

    def test_extract_rejects_unsupported_file_type(self):
        response = TestClient(main.app).post(
            "/extract", files={"file": ("receipt.txt", b"not an image", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
