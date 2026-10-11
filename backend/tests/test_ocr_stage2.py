"""
Unit tests for BubbleDetector and FieldExtractor (Stage 2A & 2B).

These tests verify:
- Accurate bubble detection across selected, empty, multiple-marked, and ambiguous states.
- Template-based coordinate cropping and resolution handling.
- Integration with mock TrOCR and PaddleOCR engines without downloading weights.
- Complete 11-field schema extraction and confidence reporting.
"""

import unittest
from unittest.mock import MagicMock
import cv2
import numpy as np

from backend.app.services.ocr.bubble_detector import BubbleDetector
from backend.app.services.ocr.field_extractor import FieldExtractor
from backend.app.services.ocr.paddle_engine import PaddleOCREngine
from backend.app.services.ocr.trocr_engine import TrOCREngine


def create_synthetic_form_canvas(width: int = 1000, height: int = 1400) -> np.ndarray:
    """Creates a synthetic white canvas with mock text and response circles."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    return canvas


def draw_bubble(
    canvas: np.ndarray,
    center: tuple,
    radius: int = 15,
    fill_ratio: float = 0.0,
) -> None:
    """Draws a printed circle outline with optional solid or partial ink fill."""
    cx, cy = center
    # Outer circle border
    cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), 2)
    if fill_ratio > 0.0:
        fill_r = max(2, int(radius * 0.75 * np.sqrt(fill_ratio)))
        cv2.circle(canvas, (cx, cy), fill_r, (0, 0, 0), -1)


class TestBubbleDetector(unittest.TestCase):
    def setUp(self):
        self.detector = BubbleDetector(
            filled_threshold=0.25,
            review_threshold=0.12,
            margin_threshold=0.08,
        )

    def test_density_calculation_empty_vs_filled(self):
        # Empty bubble crop
        empty_crop = np.full((60, 60, 3), 255, dtype=np.uint8)
        cv2.circle(empty_crop, (30, 30), 20, (0, 0, 0), 2)
        density_empty, _ = self.detector.calculate_bubble_density(empty_crop, 20)
        self.assertLess(density_empty, 0.10)

        # Filled bubble crop
        filled_crop = empty_crop.copy()
        cv2.circle(filled_crop, (30, 30), 15, (0, 0, 0), -1)
        density_filled, _ = self.detector.calculate_bubble_density(filled_crop, 20)
        self.assertGreater(density_filled, 0.50)

    def test_single_selected_option(self):
        canvas = create_synthetic_form_canvas(1000, 1400)
        h, w = canvas.shape[:2]

        yes_center = (int(0.443 * w), int(0.426 * h))
        no_center = (int(0.564 * w), int(0.426 * h))

        draw_bubble(canvas, yes_center, radius=15, fill_ratio=0.85)  # Yes: filled
        draw_bubble(canvas, no_center, radius=15, fill_ratio=0.0)   # No: empty

        q_cfg = {
            "question_id": "q1",
            "label": "Interested in BIT?",
            "options": [
                {"label": "Yes", "center_normalized": [0.443, 0.426], "radius_normalized": 0.015},
                {"label": "No", "center_normalized": [0.564, 0.426], "radius_normalized": 0.015},
            ],
        }

        res = self.detector.process_question(canvas, q_cfg)
        self.assertEqual(res["status"], "SELECTED")
        self.assertEqual(res["selected_option"], "Yes")
        self.assertGreater(res["confidence"], 0.70)
        self.assertGreater(res["densities"]["Yes"], 0.40)
        self.assertLess(res["densities"]["No"], 0.10)

    def test_unanswered_question(self):
        canvas = create_synthetic_form_canvas(1000, 1400)
        h, w = canvas.shape[:2]

        yes_center = (int(0.443 * w), int(0.426 * h))
        no_center = (int(0.564 * w), int(0.426 * h))

        draw_bubble(canvas, yes_center, radius=15, fill_ratio=0.0)
        draw_bubble(canvas, no_center, radius=15, fill_ratio=0.0)

        q_cfg = {
            "question_id": "q1",
            "label": "Interested in BIT?",
            "options": [
                {"label": "Yes", "center_normalized": [0.443, 0.426], "radius_normalized": 0.015},
                {"label": "No", "center_normalized": [0.564, 0.426], "radius_normalized": 0.015},
            ],
        }

        res = self.detector.process_question(canvas, q_cfg)
        self.assertEqual(res["status"], "UNANSWERED")
        self.assertIsNone(res["selected_option"])
        self.assertEqual(res["confidence"], 0.0)

    def test_multiple_marks_detected(self):
        canvas = create_synthetic_form_canvas(1000, 1400)
        h, w = canvas.shape[:2]

        yes_center = (int(0.443 * w), int(0.426 * h))
        no_center = (int(0.564 * w), int(0.426 * h))

        # Both Yes and No are filled
        draw_bubble(canvas, yes_center, radius=15, fill_ratio=0.80)
        draw_bubble(canvas, no_center, radius=15, fill_ratio=0.75)

        q_cfg = {
            "question_id": "q1",
            "label": "Interested in BIT?",
            "options": [
                {"label": "Yes", "center_normalized": [0.443, 0.426], "radius_normalized": 0.015},
                {"label": "No", "center_normalized": [0.564, 0.426], "radius_normalized": 0.015},
            ],
        }

        res = self.detector.process_question(canvas, q_cfg)
        self.assertEqual(res["status"], "MULTIPLE_MARKS")
        self.assertIsNone(res["selected_option"])

    def test_ambiguous_borderline_mark(self):
        canvas = create_synthetic_form_canvas(1000, 1400)
        h, w = canvas.shape[:2]

        yes_center = (int(0.443 * w), int(0.426 * h))
        no_center = (int(0.564 * w), int(0.426 * h))

        # Faint mark inside Yes (density ~0.18 in review zone between 0.12 and 0.25)
        draw_bubble(canvas, yes_center, radius=15, fill_ratio=0.20)
        draw_bubble(canvas, no_center, radius=15, fill_ratio=0.0)

        q_cfg = {
            "question_id": "q1",
            "label": "Interested in BIT?",
            "options": [
                {"label": "Yes", "center_normalized": [0.443, 0.426], "radius_normalized": 0.015},
                {"label": "No", "center_normalized": [0.564, 0.426], "radius_normalized": 0.015},
            ],
        }

        res = self.detector.process_question(canvas, q_cfg)
        self.assertEqual(res["status"], "AMBIGUOUS")
        self.assertEqual(res["selected_option"], "Yes")

    def test_debug_overlay_generation(self):
        canvas = create_synthetic_form_canvas(1000, 1400)
        h, w = canvas.shape[:2]

        yes_center = (int(0.443 * w), int(0.426 * h))
        no_center = (int(0.564 * w), int(0.426 * h))

        draw_bubble(canvas, yes_center, radius=15, fill_ratio=0.85)
        draw_bubble(canvas, no_center, radius=15, fill_ratio=0.0)

        res = self.detector.process_questionnaire(canvas, return_debug_image=True)
        self.assertIsNotNone(res["debug_image"])
        self.assertEqual(res["debug_image"].shape, canvas.shape)


class TestFieldExtractor(unittest.TestCase):
    def setUp(self):
        self.mock_trocr = MagicMock(spec=TrOCREngine)
        self.mock_paddle = MagicMock(spec=PaddleOCREngine)
        self.mock_bubble = MagicMock(spec=BubbleDetector)

        self.custom_template = {
            "canvas": {"width": 1000, "height": 1400},
            "fields": {
                "student_name": {
                    "type": "text_region",
                    "normalized_bbox": [0.10, 0.15, 0.50, 0.20],
                },
                "school_name": {
                    "type": "text_region",
                    "normalized_bbox": [0.52, 0.15, 0.90, 0.20],
                },
                "student_mobile": {
                    "type": "character_boxes",
                    "normalized_bbox": [0.10, 0.22, 0.60, 0.26],
                    "boxes": [
                        {"index": 0, "normalized_bbox": [0.10, 0.22, 0.15, 0.26]},
                        {"index": 1, "normalized_bbox": [0.15, 0.22, 0.20, 0.26]},
                    ],
                },
            },
            "questionnaire": {
                "questions": [
                    {
                        "question_id": "q1",
                        "label": "Interested in BIT?",
                        "options": [
                            {"label": "Yes", "center_normalized": [0.44, 0.42], "radius_normalized": 0.015},
                            {"label": "No", "center_normalized": [0.56, 0.42], "radius_normalized": 0.015},
                        ],
                    }
                ]
            },
        }

        self.extractor = FieldExtractor(
            template_config=self.custom_template,
            trocr_engine=self.mock_trocr,
            paddle_engine=self.mock_paddle,
            bubble_detector=self.mock_bubble,
        )

    def test_crop_field_bounds(self):
        canvas = create_synthetic_form_canvas(1000, 1400)
        field_cfg = self.custom_template["fields"]["student_name"]
        crop, (x1, y1, x2, y2), _ = self.extractor.crop_field(canvas, field_cfg)

        self.assertIsNotNone(crop)
        self.assertEqual(x1, int(0.10 * 1000) - 2)
        self.assertEqual(y1, int(0.15 * 1400) - 2)
        self.assertEqual(x2, int(0.50 * 1000) + 2)
        self.assertEqual(y2, int(0.20 * 1400) + 2)

    def test_extract_text_field_trocr_success(self):
        canvas = create_synthetic_form_canvas()
        self.mock_trocr.predict_crop.return_value = {
            "text": "Prapanchan",
            "confidence": 0.945,
            "status": "SUCCESS",
            "latency_ms": 45.2,
        }

        res = self.extractor.extract_text_field(canvas, "student_name", preferred_engine="trocr")
        self.assertEqual(res["status"], "EXTRACTED")
        self.assertEqual(res["value"], "Prapanchan")
        self.assertEqual(res["confidence"], 0.945)
        self.assertEqual(res["source"], "trocr")
        self.assertIsNotNone(res["bbox"])

    def test_extract_text_field_engine_unavailable(self):
        canvas = create_synthetic_form_canvas()
        self.mock_trocr.predict_crop.return_value = {
            "text": None,
            "confidence": None,
            "status": "MODEL_UNAVAILABLE",
            "error": "TrOCR weights not found",
            "latency_ms": 1.0,
        }

        res = self.extractor.extract_text_field(canvas, "student_name", preferred_engine="trocr")
        self.assertEqual(res["status"], "ENGINE_UNAVAILABLE")
        self.assertIsNone(res["value"])
        self.assertIsNone(res["confidence"])
        self.assertIn("not found", res["error"])

    def test_extract_all_fields_complete_schema(self):
        canvas = create_synthetic_form_canvas()
        # Mock TrOCR return
        self.mock_trocr.predict_crop.return_value = {
            "text": "Sample Text",
            "confidence": 0.91,
            "status": "SUCCESS",
            "latency_ms": 30.0,
        }
        # Mock BubbleDetector return
        self.mock_bubble.process_questionnaire.return_value = {
            "questions": {
                "q1": {
                    "question_id": "q1",
                    "selected_option": "Yes",
                    "status": "SELECTED",
                    "confidence": 0.95,
                    "densities": {"Yes": 0.70, "No": 0.05},
                },
                "q2": {
                    "question_id": "q2",
                    "selected_option": "Excellent",
                    "status": "SELECTED",
                    "confidence": 0.92,
                    "densities": {"Excellent": 0.65},
                },
                "q3": {
                    "question_id": "q3",
                    "selected_option": "Good",
                    "status": "SELECTED",
                    "confidence": 0.88,
                    "densities": {"Good": 0.55},
                },
                "q4": {
                    "question_id": "q4",
                    "selected_option": "No",
                    "status": "SELECTED",
                    "confidence": 0.93,
                    "densities": {"Yes": 0.02, "No": 0.62},
                },
            },
            "answers": {"q1": "Yes", "q2": "Excellent", "q3": "Good", "q4": "No"},
            "all_answered": True,
            "has_ambiguous": False,
            "debug_image": None,
        }

        res = self.extractor.extract_all_fields(canvas, return_crops=True)
        fields = res["fields"]

        # Check that all 11 fields exist in the result
        expected_fields = [
            "student_name",
            "school_name",
            "date",
            "student_mobile_number",
            "parent_mobile_number",
            "email",
            "cut_off_marks",
            "interested_in_bit",
            "career_guidance_usefulness",
            "speaker_effectiveness",
            "interested_in_choice_filling_assistance",
        ]
        for ef in expected_fields:
            self.assertIn(ef, fields, f"Missing field in extracted results: {ef}")

        # Check questionnaire mappings
        self.assertEqual(fields["interested_in_bit"]["value"], "Yes")
        self.assertEqual(fields["interested_in_bit"]["source"], "bubble_detector")
        self.assertEqual(fields["career_guidance_usefulness"]["value"], "Excellent")
        self.assertEqual(fields["speaker_effectiveness"]["value"], "Good")
        self.assertEqual(fields["interested_in_choice_filling_assistance"]["value"], "No")


if __name__ == "__main__":
    unittest.main()
