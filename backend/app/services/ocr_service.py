"""
Unified OCR Service for Form Digitization.

Provides a consolidated facade combining:
1. TrOCREngine: Handwritten text recognition on cropped text fields.
2. PaddleOCREngine: Printed and structured text detection/recognition.
3. BubbleDetector: OpenCV-based circular mask & ink density evaluation for Yes/No and 1-5 ratings.
4. FieldExtractor: Template-driven ROI cropping and engine orchestration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.app.core.config import get_form_template_config
from backend.app.services.ocr.bubble_detector import BubbleDetector
from backend.app.services.ocr.field_extractor import FieldExtractor
from backend.app.services.ocr.paddle_engine import PaddleOCREngine
from backend.app.services.ocr.trocr_engine import TrOCREngine

logger = logging.getLogger(__name__)


class OCRService:
    """
    Unified OCR Service uniting PaddleOCR, TrOCR, and BubbleDetector.
    """

    def __init__(
        self,
        template_config: Optional[Dict[str, Any]] = None,
        trocr_engine: Optional[TrOCREngine] = None,
        paddle_engine: Optional[PaddleOCREngine] = None,
        bubble_detector: Optional[BubbleDetector] = None,
    ) -> None:
        self.config = template_config or get_form_template_config()
        self.trocr = trocr_engine or TrOCREngine()
        self.paddle = paddle_engine or PaddleOCREngine()
        self.bubble_detector = bubble_detector or BubbleDetector(template_config=self.config)

        self.field_extractor = FieldExtractor(
            template_config=template_config,
            trocr_engine=self.trocr,
            paddle_engine=self.paddle,
            bubble_detector=self.bubble_detector,
        )

    def process_form_image(
        self,
        aligned_image: np.ndarray,
        form_index: Optional[int] = None,
        is_split: bool = False,
        return_crops: bool = False,
        return_debug_overlay: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes a complete aligned feedback form image through the OCR pipeline.

        Returns:
            Dict containing:
                - fields: Dict mapping field names to extracted values, source, confidence, status.
                - questionnaire: Detailed per-question results.
                - crops: Optional dict mapping field_name to image crops.
                - debug_image: Optional annotated OpenCV image.
                - status: "SUCCESS", "PARTIAL", or "UNKNOWN_TEMPLATE"
        """
        return self.field_extractor.extract_all_fields(
            aligned_image=aligned_image,
            form_index=form_index,
            is_split=is_split,
            return_crops=return_crops,
            return_debug_overlay=return_debug_overlay,
        )

    def extract_single_field(
        self,
        aligned_image: np.ndarray,
        field_name: str,
        preferred_engine: str = "trocr",
    ) -> Dict[str, Any]:
        """
        Extracts a single field from the aligned form image.
        """
        return self.field_extractor.extract_text_field(
            aligned_image=aligned_image,
            schema_field_name=field_name,
            preferred_engine=preferred_engine,
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Returns readiness and metadata for all OCR subsystems.
        """
        return {
            "ocr_pipeline": "PaddleOCR + TrOCR + OpenCV Bubble Detector",
            "trocr": self.trocr.get_status(),
            "paddleocr": self.paddle.get_status(),
            "bubble_detector": {
                "filled_threshold": self.bubble_detector.filled_threshold,
                "review_threshold": self.bubble_detector.review_threshold,
                "margin_threshold": self.bubble_detector.margin_threshold,
            },
        }

    # Backward-compatible helper methods for legacy test compatibility
    def process_numeric_field_cv_cnn(
        self,
        image: np.ndarray,
        field_name: str,
    ) -> Dict[str, Any]:
        """
        Backward-compatible method for numeric field extraction.
        Routes to template field extractor.
        """
        res = self.extract_single_field(image, field_name, preferred_engine="trocr")
        val = res.get("value")
        conf = res.get("confidence", 0.0) or 0.0

        return {
            "field_name": field_name,
            "consensus_text": val,
            "confidence": conf,
            "variant_agreement": conf,
            "bounding_box": res.get("bbox", [0, 0, 0, 0]),
            "detector_status": "TEMPLATE_CROP",
            "per_digit_results": [],
            "ocr_engine": res.get("source", "trocr"),
            "status": "VALID" if val else res.get("status", "FAILED"),
        }

    def run_multi_variant_ocr(
        self,
        variants: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Backward-compatible runner across numeric fields.
        """
        orig = variants.get("original")
        if orig is None and len(variants) > 0:
            orig = list(variants.values())[0]

        results = {}
        numeric_fields = ["student_mobile", "parent_mobile", "cutoff_marks", "date"]

        if orig is not None:
            for field_name in numeric_fields:
                res = self.process_numeric_field_cv_cnn(orig, field_name)
                results[field_name] = res

        return results
