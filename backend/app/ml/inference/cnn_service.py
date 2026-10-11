import os
import time
import numpy as np
from typing import Dict, Any, List, Optional
from digit_classifier import DigitClassifier
from backend.app.ml.digit_segmenter import DigitSegmenter

class CNNInferenceService:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "pretrained_model.pt"
        self.classifier = DigitClassifier(model_path=self.model_path)
        self.segmenter = DigitSegmenter()

    def is_ready(self) -> bool:
        return self.classifier is not None and (
            os.path.exists(self.classifier.model_path) or os.path.exists(self.classifier.secondary_path)
        )

    def get_status(self) -> Dict[str, Any]:
        ready = self.is_ready()
        return {
            "status": "READY" if ready else "NOT_TRAINED",
            "weights_exist": ready,
            "architecture": "Custom CNN (Pluggable Interface)",
            "notice": "Pluggable Custom CNN digit recognition engine active. Ready for custom model weights."
        }

    def process_numeric_field(self, field_crop: np.ndarray) -> Dict[str, Any]:
        """
        Executes pure CV digit segmentation + 15-step preprocessing + batch CNN inference.
        STRICTLY NO OCR.
        """
        start_time = time.time()
        if field_crop is None or field_crop.size == 0:
            return {
                "value": None,
                "confidence": 0.0,
                "per_digit_results": [],
                "status": "FAILED",
                "error": "Empty or invalid field crop"
            }

        # 1. Computer Vision digit segmentation + 15-step preprocessing
        digit_crops, digit_bboxes, preprocessed_canvases, batch_tensor = self.segmenter.segment_digits(field_crop)

        if len(preprocessed_canvases) == 0:
            return {
                "value": None,
                "confidence": 0.0,
                "per_digit_results": [],
                "status": "FAILED",
                "error": "Digit segmentation failed"
            }

        # 2. Batch CNN inference on (N, 28, 28, 1) tensor
        batch_result = self.classifier.predict_digit_batch(batch_tensor)
        latency = (time.time() - start_time) * 1000.0

        return {
            "value": batch_result.get("value"),
            "confidence": batch_result.get("confidence", 0.0),
            "per_digit_results": batch_result.get("per_digit_results", []),
            "digit_bboxes": digit_bboxes,
            "status": "COMPLETED",
            "latency_ms": round(latency, 2)
        }

    def predict_phone_number(self, box_crops: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs CNN inference box-by-box for student mobile number (10 boxes).
        """
        crops = [b["crop"] for b in box_crops if "crop" in b]
        if not crops:
            return {
                "value": None,
                "confidence": 0.0,
                "per_character_results": [],
                "status": "FAILED",
                "error": "No box crops provided"
            }

        _, _, _, batch_tensor = self.segmenter.segment_digits(field_crop=np.zeros((10, 10), dtype=np.uint8), is_character_boxes=True, box_crops=crops)
        res = self.classifier.predict_digit_batch(batch_tensor)
        return {
            "value": res.get("value"),
            "confidence": res.get("confidence", 0.0),
            "per_character_results": res.get("per_digit_results", []),
            "status": "COMPLETED"
        }

    def predict_cutoff_marks(self, cutoff_data: Dict[str, Any]) -> Dict[str, Any]:
        boxes = cutoff_data.get("boxes", [])
        if not boxes:
            return {"value": None, "confidence": 0.0, "per_character_results": [], "status": "FAILED"}

        crops = [b["crop"] for b in boxes if "crop" in b and b.get("type") != "decimal"]
        if crops:
            _, _, _, batch_tensor = self.segmenter.segment_digits(field_crop=np.zeros((10, 10), dtype=np.uint8), is_character_boxes=True, box_crops=crops)
            res = self.classifier.predict_digit_batch(batch_tensor)
            val = res.get("value", "")
            if len(val) >= 3:
                val = val[:3] + "." + val[3:]
            return {"value": val, "confidence": res.get("confidence", 0.0), "per_character_results": res.get("per_digit_results", []), "status": "COMPLETED"}

        return {"value": None, "confidence": 0.0, "per_character_results": [], "status": "FAILED"}
