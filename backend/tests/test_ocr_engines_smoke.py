"""
Isolated smoke tests for PaddleOCREngine and TrOCREngine.

These tests verify module structure, singleton patterns, input validation,
status reporting, and mocked inference pipelines without requiring heavy model
downloads or active external network connections.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.app.services.ocr.paddle_engine import PaddleOCREngine
from backend.app.services.ocr.trocr_engine import TrOCREngine


class TestPaddleOCREngineSmoke(unittest.TestCase):
    def setUp(self):
        # Reset singleton state for clean test isolation
        PaddleOCREngine._instance = None

    def test_singleton_instance(self):
        engine1 = PaddleOCREngine(lang="en")
        engine2 = PaddleOCREngine(lang="en")
        self.assertIs(engine1, engine2)
        self.assertEqual(engine1.lang, "en")

    def test_get_status_unloaded(self):
        engine = PaddleOCREngine()
        status = engine.get_status()
        self.assertEqual(status["engine"], "PaddleOCR")
        self.assertEqual(status["lang"], "en")
        self.assertFalse(status["is_loaded"])
        self.assertIn(status["status"], ["NOT_LOADED", "ERROR"])

    def test_input_validation(self):
        engine = PaddleOCREngine()
        # Test None
        res_none = engine.predict(None)
        self.assertEqual(res_none["status"], "INVALID_INPUT")
        self.assertEqual(res_none["items"], [])

        # Test empty array
        res_empty = engine.predict(np.zeros((0, 0)))
        self.assertEqual(res_empty["status"], "INVALID_INPUT")

        # Test tiny invalid shape
        res_tiny = engine.predict(np.zeros((1, 1)))
        self.assertEqual(res_tiny["status"], "INVALID_INPUT")

    def test_mocked_paddle_inference(self):
        """Verify inference output formatting with a mocked PaddleOCR engine."""
        engine = PaddleOCREngine()
        mock_ocr = MagicMock()
        # Simulated PaddleOCR output: [[ [polygon, (text, conf)] ]]
        mock_ocr.ocr.return_value = [
            [
                [[[10, 20], [100, 20], [100, 50], [10, 50]], ("STUDENT NAME", 0.985)],
                [[[10, 60], [150, 60], [150, 90], [10, 90]], ("BANNARI AMMAN", 0.942)],
            ]
        ]

        engine._ocr = mock_ocr
        engine._is_loaded = True

        dummy_img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = engine.predict(dummy_img)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["items"][0]["text"], "STUDENT NAME")
        self.assertEqual(result["items"][0]["confidence"], 0.985)
        self.assertEqual(result["items"][0]["bbox"], [10, 20, 100, 50])
        self.assertEqual(result["full_text"], "STUDENT NAME BANNARI AMMAN")

    def test_mocked_predict_crop(self):
        engine = PaddleOCREngine()
        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = [
            [
                [[[5, 5], [50, 5], [50, 25], [5, 25]], ("9876543210", 0.991)],
            ]
        ]
        engine._ocr = mock_ocr
        engine._is_loaded = True

        crop = np.full((30, 60, 3), 255, dtype=np.uint8)
        res = engine.predict_crop(crop)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["text"], "9876543210")
        self.assertEqual(res["confidence"], 0.991)


class TestTrOCREngineSmoke(unittest.TestCase):
    def setUp(self):
        # Reset singleton state for clean test isolation
        TrOCREngine._instance = None

    def test_singleton_instance(self):
        engine1 = TrOCREngine(checkpoint="microsoft/trocr-base-handwritten")
        engine2 = TrOCREngine(checkpoint="microsoft/trocr-base-handwritten")
        self.assertIs(engine1, engine2)
        self.assertEqual(engine1.checkpoint, "microsoft/trocr-base-handwritten")

    def test_get_status_unloaded(self):
        engine = TrOCREngine()
        status = engine.get_status()
        self.assertEqual(status["engine"], "TrOCR")
        self.assertEqual(status["checkpoint"], "microsoft/trocr-base-handwritten")
        self.assertFalse(status["is_loaded"])
        self.assertIn(status["status"], ["NOT_LOADED", "ERROR"])

    def test_input_validation(self):
        engine = TrOCREngine()
        # Test None
        res_none = engine.predict_crop(None)
        self.assertEqual(res_none["status"], "INVALID_INPUT")
        self.assertIsNone(res_none["text"])

        # Test empty array
        res_empty = engine.predict_crop(np.zeros((0, 0)))
        self.assertEqual(res_empty["status"], "INVALID_INPUT")

        # Test tiny dimensions (<5px)
        res_tiny = engine.predict_crop(np.zeros((3, 3)))
        self.assertEqual(res_tiny["status"], "INVALID_INPUT")

    def test_reject_full_form_image(self):
        """TrOCR must reject full form images to enforce crop-only usage."""
        engine = TrOCREngine(max_crop_height=1000, max_crop_width=1000)
        full_form_img = np.zeros((2000, 1500, 3), dtype=np.uint8)
        res = engine.predict_crop(full_form_img)
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("exceed maximum allowed field size", res["error"])

    def test_mocked_trocr_inference(self):
        """Verify inference execution and token score parsing with a mocked processor/model."""
        import torch

        engine = TrOCREngine()
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processor return
        mock_pixel_values = torch.zeros((1, 3, 384, 384))
        mock_proc_ret = MagicMock()
        mock_proc_ret.pixel_values = mock_pixel_values
        mock_processor.return_value = mock_proc_ret
        mock_processor.batch_decode.return_value = ["Prapanchan K"]

        # Mock model generate return
        mock_gen_out = MagicMock()
        mock_gen_out.sequences = torch.tensor([[0, 100, 101, 2]])  # token IDs
        # 3 decoding steps with logits for a vocab of 200
        step1_logits = torch.randn((1, 200))
        step2_logits = torch.randn((1, 200))
        step3_logits = torch.randn((1, 200))
        mock_gen_out.scores = (step1_logits, step2_logits, step3_logits)
        mock_model.generate.return_value = mock_gen_out

        engine._processor = mock_processor
        engine._model = mock_model
        engine._is_loaded = True

        crop_sample = np.full((50, 200, 3), 255, dtype=np.uint8)
        result = engine.predict_crop(crop_sample)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["text"], "Prapanchan K")
        self.assertIsNotNone(result["confidence"])
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertIn("confidence_notice", result)


if __name__ == "__main__":
    unittest.main()
