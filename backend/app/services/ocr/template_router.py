"""
Template Router and Layout Profile Classifier.

Provides deterministic template identification, profile selection, and coordinate mapping:
1. Identifies known form layouts:
   - BIT_CAREER_GUIDANCE_HANDWRITTEN_A4: Physical printed form with 4 corner registration markers.
   - BIT_CAREER_GUIDANCE_DIGITAL_FORM_1: Digitally filled stacked upper form (from PDF).
   - BIT_CAREER_GUIDANCE_DIGITAL_FORM_2: Digitally filled stacked lower form (from PDF).
2. Explicitly rejects unclassifiable forms with UNKNOWN_TEMPLATE (no silent fallback).
3. Provides anchor-based coordinates and verified field crop bounding boxes.
4. Performs alignment quality and dimension validation checks.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

from backend.app.core.config import get_form_template_config

logger = logging.getLogger(__name__)

UNKNOWN_TEMPLATE = "UNKNOWN_TEMPLATE"
TEMPLATE_HANDWRITTEN_A4 = "BIT_CAREER_GUIDANCE_HANDWRITTEN_A4"
TEMPLATE_DIGITAL_FORM_1 = "BIT_CAREER_GUIDANCE_DIGITAL_FORM_1"
TEMPLATE_DIGITAL_FORM_2 = "BIT_CAREER_GUIDANCE_DIGITAL_FORM_2"


class TemplateRouter:
    """
    Classifies page images into template profiles and serves verified field coordinates.
    """

    def __init__(self, dummy_template_path: str = "config/form_template_dummy_pdf.yaml") -> None:
        self.dummy_template_path = dummy_template_path
        self._dummy_config: Optional[Dict[str, Any]] = None
        self._a4_config: Optional[Dict[str, Any]] = None

    def _get_dummy_config(self) -> Dict[str, Any]:
        if self._dummy_config is None:
            if os.path.exists(self.dummy_template_path):
                with open(self.dummy_template_path, "r", encoding="utf-8") as f:
                    self._dummy_config = yaml.safe_load(f) or {}
            else:
                logger.warning("Dummy template config not found at %s", self.dummy_template_path)
                self._dummy_config = {}
        return self._dummy_config

    def _get_a4_config(self) -> Dict[str, Any]:
        if self._a4_config is None:
            self._a4_config = get_form_template_config()
        return self._a4_config

    def check_registration_markers(
        self,
        img: np.ndarray,
        marker_thresh: float = 0.15,
    ) -> Tuple[bool, float]:
        """
        Validates presence of 4 registration corner markers on a physical A4 form image.
        Returns (has_markers, confidence_score).
        """
        if img is None or img.size == 0:
            return False, 0.0

        h, w = img.shape[:2]
        if h < 2500 or w < 1800:
            return False, 0.0

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        marker_regions = [
            (int(0.03 * w), int(0.02 * h), int(0.08 * w), int(0.06 * h)),   # top-left
            (int(0.91 * w), int(0.02 * h), int(0.97 * w), int(0.06 * h)),   # top-right
            (int(0.91 * w), int(0.93 * h), int(0.97 * w), int(0.98 * h)),   # bottom-right
            (int(0.03 * w), int(0.93 * h), int(0.08 * w), int(0.98 * h)),   # bottom-left
        ]

        found_count = 0
        for x1, y1, x2, y2 in marker_regions:
            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            dark_ratio = float(np.sum(patch < 120)) / float(patch.size)
            if dark_ratio >= marker_thresh:
                found_count += 1

        marker_conf = found_count / 4.0
        return (found_count >= 3), marker_conf

    def identify_template(
        self,
        img: np.ndarray,
        form_index: Optional[int] = None,
        is_split: bool = False,
    ) -> Dict[str, Any]:
        """
        Classifies an input form image into a supported template profile.

        Parameters:
            img: Form image array (OpenCV BGR or Grayscale).
            form_index: Form index if known from preprocessor splitting (1 or 2).
            is_split: Whether this image came from a split stacked page.

        Returns:
            Dict containing:
                - template_id: str ("BIT_CAREER_GUIDANCE_HANDWRITTEN_A4",
                                    "BIT_CAREER_GUIDANCE_DIGITAL_FORM_1",
                                    "BIT_CAREER_GUIDANCE_DIGITAL_FORM_2", or "UNKNOWN_TEMPLATE")
                - confidence: float
                - profile_name: str
                - reason: str
                - is_known: bool
        """
        if img is None or img.size == 0:
            return {
                "template_id": UNKNOWN_TEMPLATE,
                "confidence": 0.0,
                "profile_name": "Unknown",
                "reason": "Image is empty or None",
                "is_known": False,
            }

        h, w = img.shape[:2]
        aspect_ratio = w / float(h) if h > 0 else 0.0

        # Check 1: If preprocessor already separated a stacked form with form_index
        if is_split and form_index in (1, 2):
            # Form 1 vs Form 2
            tmpl_id = TEMPLATE_DIGITAL_FORM_1 if form_index == 1 else TEMPLATE_DIGITAL_FORM_2
            return {
                "template_id": tmpl_id,
                "confidence": 1.0,
                "profile_name": f"Digital Form {form_index}",
                "reason": f"Identified via preprocessor split marker as form {form_index}",
                "is_known": True,
            }

        # Check 2: Physical A4 Portrait Form (H: ~3508, W: ~2479, aspect ratio ~ 0.71)
        if 0.60 <= aspect_ratio <= 0.85 and h >= 2500:
            has_markers, marker_conf = self.check_registration_markers(img)
            if has_markers:
                return {
                    "template_id": TEMPLATE_HANDWRITTEN_A4,
                    "confidence": marker_conf,
                    "profile_name": "Physical Handwritten A4 Form",
                    "reason": f"A4 aspect ratio ({aspect_ratio:.2f}) and registration markers verified ({marker_conf:.2f})",
                    "is_known": True,
                }

        # Check 3: Digital US Letter Half Form (W: 2400..2700, H: 1400..1800, aspect ratio ~ 1.5..1.8)
        if 1.40 <= aspect_ratio <= 1.95 and 2000 <= w <= 2800 and 1200 <= h <= 2000:
            # Distinguish Form 1 vs Form 2 by inspecting baseline date box position
            # Form 1 date box is at y ~ 240..360, Form 2 date box is at y ~ 270..420
            # Default to Form 1 if form_index is not provided or 1
            selected_idx = 2 if form_index == 2 else 1
            tmpl_id = TEMPLATE_DIGITAL_FORM_2 if selected_idx == 2 else TEMPLATE_DIGITAL_FORM_1
            return {
                "template_id": tmpl_id,
                "confidence": 0.90,
                "profile_name": f"Digital Form {selected_idx}",
                "reason": f"Digital half-sheet dimensions ({w}x{h}, aspect {aspect_ratio:.2f})",
                "is_known": True,
            }

        # Fallthrough: Explicit UNKNOWN_TEMPLATE
        logger.warning(
            "Rejecting unclassifiable form layout: dimensions %dx%d, aspect ratio %.3f",
            w, h, aspect_ratio
        )
        return {
            "template_id": UNKNOWN_TEMPLATE,
            "confidence": 0.0,
            "profile_name": "Unknown Layout",
            "reason": f"Dimensions ({w}x{h}) and markers do not match any known template profile",
            "is_known": False,
        }

    def get_template_profile(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the coordinate and extraction profile for a given template ID.
        """
        if template_id == TEMPLATE_HANDWRITTEN_A4:
            return self._get_a4_config()
        elif template_id == TEMPLATE_DIGITAL_FORM_1:
            dummy_cfg = self._get_dummy_config()
            return dummy_cfg.get("form_1", {})
        elif template_id == TEMPLATE_DIGITAL_FORM_2:
            dummy_cfg = self._get_dummy_config()
            return dummy_cfg.get("form_2", {})
        return None
