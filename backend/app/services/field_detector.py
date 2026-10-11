import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from backend.app.services.roi_extractor import ROIExtractor

class FieldDetector:
    """
    Field Detector Module using Template/Contour localization.
    Decoupled from YOLO for custom CNN / CV model integration.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.detector_type = "Template/Contour Field Localization"
        self.roi_extractor = ROIExtractor()

    def get_status(self) -> Dict[str, Any]:
        return {
            "detector_type": self.detector_type,
            "yolo_loaded": False,
            "notice": "Pure Template/Contour field detector active."
        }

    def detect_and_crop_field(
        self,
        image: np.ndarray,
        field_name: str,
        page_num: int = 1
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Detects bounding box for requested field_name using contour/template localization.
        """
        if image is None or image.size == 0:
            return None, {
                "field": field_name,
                "field_confidence": 0.0,
                "detector_status": self.detector_type,
                "is_valid": False,
                "reason": "Input image is empty"
            }

        crop, meta = self.roi_extractor.crop_single_field(image, field_name, page_num)
        meta["field_confidence"] = 0.95 if meta.get("is_valid") else 0.0
        meta["detector_status"] = self.detector_type
        return crop, meta
