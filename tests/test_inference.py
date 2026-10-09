import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from src.inference import ReceiptExtractor


class FakeEncoding(dict):
    def word_ids(self, batch_index=0):
        return [None, 0, 1, None]


class FakeProcessor:
    def __call__(self, **kwargs):
        return FakeEncoding(input_ids=torch.tensor([[0, 1, 2, 3]]))


class AllOutsideModel:
    def __call__(self, **kwargs):
        logits = torch.zeros((1, 4, 3))
        logits[:, :, 0] = 1.0
        return SimpleNamespace(logits=logits)


class InferenceTests(unittest.TestCase):
    def test_empty_model_field_uses_ocr_fallback(self):
        tokens = [
            {"text": "12/31/2025", "confidence": 95.0, "bbox": [100, 100, 400, 150],
             "pixel_bbox": [10, 10, 40, 15], "line_id": (1, 1, 1)},
            {"text": "OTHER", "confidence": 95.0, "bbox": [450, 100, 700, 150],
             "pixel_bbox": [45, 10, 70, 15], "line_id": (1, 1, 1)},
        ]
        extractor = ReceiptExtractor.__new__(ReceiptExtractor)
        extractor.processor = FakeProcessor()
        extractor.model = AllOutsideModel()
        extractor.device = torch.device("cpu")
        extractor.max_length = 512
        extractor.id2label = {0: "O", 1: "B-DATE", 2: "I-DATE"}

        with patch("src.inference.image_to_ocr_tokens", return_value=tokens):
            result = extractor.extract(Image.new("RGB", (100, 100), "white"))

        self.assertEqual(result["fields"]["DATE"], "12/31/2025")
        self.assertEqual(result["field_sources"]["DATE"], "ocr_rule_fallback")
        self.assertIn("Review these values", result["warning"])


if __name__ == "__main__":
    unittest.main()
