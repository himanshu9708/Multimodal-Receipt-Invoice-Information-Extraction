import unittest

from src.preprocessing import normalize_bbox, validate_document_record


class PreprocessingTests(unittest.TestCase):
    def test_normalize_bbox_scales_pixels_to_layoutlm_coordinates(self):
        self.assertEqual(normalize_bbox([10, 20, 90, 80], (100, 100)), [100, 200, 900, 800])

    def test_normalize_bbox_clamps_coordinates_to_valid_range(self):
        self.assertEqual(normalize_bbox([-5, 0, 110, 100], (100, 100)), [0, 0, 1000, 1000])

    def test_normalize_bbox_rejects_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_bbox([0, 0, 1, 1], (0, 10))

    def test_validate_document_record_accepts_valid_sample(self):
        record = {
            "document_id": "sample-1", "split": "train", "image_path": "receipt.jpg",
            "words": ["TOTAL"], "bboxes": [[0, 0, 1000, 200]], "ner_tags": ["B-TOTAL"],
        }
        validate_document_record(record, {"O", "B-TOTAL"})

    def test_validate_document_record_rejects_misaligned_arrays(self):
        record = {
            "document_id": "sample-1", "split": "train", "image_path": "receipt.jpg",
            "words": ["TOTAL"], "bboxes": [], "ner_tags": ["B-TOTAL"],
        }
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            validate_document_record(record)


if __name__ == "__main__":
    unittest.main()
