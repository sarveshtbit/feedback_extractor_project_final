"""
Template-Based Field Extractor.

This module coordinates layout-based cropping and dispatches extracted regions
to the appropriate recognition engines:
- Handwritten Text Regions (e.g., Student Name, School Name) -> TrOCREngine
- Structured / Boxed Text (e.g., Mobile Numbers, Date, Cutoff, Email) -> PaddleOCREngine
- Response Circles (Yes/No, 1-5 Ratings) -> OpenCV Bubble Analysis / BubbleDetector

Features:
- Configurable template profile routing via TemplateRouter (supports digital stacked PDF forms,
  physical A4 handwritten forms, and explicit UNKNOWN_TEMPLATE rejection).
- Clean prefix removal, digit sequence extraction, and formatting for structured fields.
- Non-fabrication: Missing, unreadable, or blank fields remain None with explicit status records.
- Comprehensive status reporting: VALID, BLANK, UNANSWERED, AMBIGUOUS, NEEDS_REVIEW,
  ENGINE_UNAVAILABLE, UNKNOWN_TEMPLATE.
"""

import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.app.core.config import get_form_template_config
from backend.app.services.ocr.bubble_detector import BubbleDetector
from backend.app.services.ocr.paddle_engine import PaddleOCREngine
from backend.app.services.ocr.trocr_engine import TrOCREngine
from backend.app.services.ocr.template_router import (
    TemplateRouter,
    UNKNOWN_TEMPLATE,
    TEMPLATE_HANDWRITTEN_A4,
    TEMPLATE_DIGITAL_FORM_1,
    TEMPLATE_DIGITAL_FORM_2,
)

logger = logging.getLogger(__name__)

ALL_SCHEMA_FIELDS = [
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

SCHEMA_TO_TEMPLATE_MAP = {
    "student_name": "student_name",
    "school_name": "school_name",
    "date": "date",
    "student_mobile_number": "student_mobile",
    "parent_mobile_number": "parent_mobile",
    "email": "email",
    "cut_off_marks": "cutoff_marks",
}

QUESTION_ID_TO_SCHEMA_MAP = {
    "q1": "interested_in_bit",
    "q2": "career_guidance_usefulness",
    "q3": "speaker_effectiveness",
    "q4": "interested_in_choice_filling_assistance",
}


def clean_text_field(raw_str: Optional[str], field_type: str) -> Optional[str]:
    """Cleans OCR artifacts and label prefixes from text fields."""
    if not raw_str:
        return None
    s = raw_str.strip()
    if field_type == "student_name":
        s = re.sub(r'^(student\s*name|student|name)\s*[:\-_]*\s*', '', s, flags=re.I).strip()
        s = re.sub(r'\b([A-Z]{2})\s+([A-Z]{4,})\b', r'\1\2', s)
    elif field_type == "school_name":
        s = re.sub(r'^(school\s*name|school|name|nam)\s*[:\-_]*\s*', '', s, flags=re.I).strip()
        s = re.sub(r'^nam\s*', '', s, flags=re.I).strip()
    elif field_type == "email":
        s = re.sub(r'^(mail\s*id|mail|email)\s*[:\-_]*\s*', '', s, flags=re.I).strip()
        if "@" not in s:
            return None
        s = re.sub(r'\s*@\s*', '@', s)
        s = re.sub(r'\s*\.\s*', '.', s)
        s = s.replace(" ", "")
        if s.endswith(".edy"):
            s = s[:-4] + ".edu"
    return s if len(s) >= 2 else None


class FieldExtractor:
    """
    Template-based field cropper and OCR dispatcher supporting multiple layout profiles.
    """

    def __init__(
        self,
        template_config: Optional[Dict[str, Any]] = None,
        trocr_engine: Optional[TrOCREngine] = None,
        paddle_engine: Optional[PaddleOCREngine] = None,
        bubble_detector: Optional[BubbleDetector] = None,
        template_router: Optional[TemplateRouter] = None,
    ) -> None:
        self.custom_template_passed = template_config is not None
        self.config = template_config or get_form_template_config()
        self.trocr = trocr_engine or TrOCREngine()
        self.paddle = paddle_engine or PaddleOCREngine()
        self.bubble_detector = bubble_detector or BubbleDetector(template_config=self.config)
        self.template_router = template_router or TemplateRouter()

    @staticmethod
    def map_normalized_bbox(
        bbox_norm: List[float],
        img_w: int,
        img_h: int,
        padding_px: int = 2,
    ) -> Tuple[int, int, int, int]:
        x1 = max(0, int(bbox_norm[0] * img_w) - padding_px)
        y1 = max(0, int(bbox_norm[1] * img_h) - padding_px)
        x2 = min(img_w, int(bbox_norm[2] * img_w) + padding_px)
        y2 = min(img_h, int(bbox_norm[3] * img_h) + padding_px)
        return x1, y1, x2, y2

    def crop_field(
        self,
        aligned_image: np.ndarray,
        field_cfg: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int], List[Dict[str, Any]]]:
        h_img, w_img = aligned_image.shape[:2]
        field_type = field_cfg.get("type", "text_region")
        box_crops: List[Dict[str, Any]] = []

        if "normalized_bbox" in field_cfg:
            x1, y1, x2, y2 = self.map_normalized_bbox(field_cfg["normalized_bbox"], w_img, h_img)
        elif "bounding_box" in field_cfg:
            bx, by, bw, bh = field_cfg["bounding_box"]
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0

        if x2 <= x1 or y2 <= y1:
            return None, (0, 0, 0, 0), []

        full_crop = aligned_image[y1:y2, x1:x2]

        if field_type == "character_boxes" and "boxes" in field_cfg:
            for b in field_cfg["boxes"]:
                b_idx = b.get("index", 0)
                b_type = b.get("type", "box")
                if "normalized_bbox" in b:
                    bx1, by1, bx2, by2 = self.map_normalized_bbox(b["normalized_bbox"], w_img, h_img)
                    b_crop = aligned_image[by1:by2, bx1:bx2]
                else:
                    bx1, by1, bx2, by2 = 0, 0, 0, 0
                    b_crop = None

                box_crops.append({
                    "index": b_idx,
                    "type": b_type,
                    "bbox": [bx1, by1, bx2, by2],
                    "crop": b_crop,
                })

        return full_crop, (x1, y1, x2, y2), box_crops

    def extract_text_field(
        self,
        aligned_image: np.ndarray,
        schema_field_name: str,
        preferred_engine: str = "trocr",
    ) -> Dict[str, Any]:
        """Extracts a single text field (backward compatible)."""
        tmpl_key = SCHEMA_TO_TEMPLATE_MAP.get(schema_field_name, schema_field_name)
        fields_cfg = self.config.get("fields", {})
        field_cfg = fields_cfg.get(tmpl_key)

        if not field_cfg:
            return {
                "field_name": schema_field_name,
                "value": None,
                "confidence": None,
                "source": "none",
                "status": "CONFIG_MISSING",
                "error": f"Field '{tmpl_key}' not found in form template configuration",
                "bbox": None,
            }

        crop, bbox, box_crops = self.crop_field(aligned_image, field_cfg)
        if crop is None or crop.size == 0:
            return {
                "field_name": schema_field_name,
                "value": None,
                "confidence": None,
                "source": "none",
                "status": "CROP_FAILED",
                "error": f"Failed to crop field '{tmpl_key}'",
                "bbox": list(bbox),
            }

        if preferred_engine == "trocr":
            res = self.trocr.predict_crop(crop)
            if res.get("status") == "SUCCESS":
                val = res.get("text")
                return {
                    "field_name": schema_field_name,
                    "value": val if val else None,
                    "confidence": res.get("confidence"),
                    "source": "trocr",
                    "status": "EXTRACTED" if val else "BLANK",
                    "bbox": list(bbox),
                    "latency_ms": res.get("latency_ms", 0.0),
                    "error": None,
                }
            else:
                return {
                    "field_name": schema_field_name,
                    "value": None,
                    "confidence": None,
                    "source": "trocr",
                    "status": "ENGINE_UNAVAILABLE" if res.get("status") == "MODEL_UNAVAILABLE" else "FAILED",
                    "error": res.get("error"),
                    "bbox": list(bbox),
                    "latency_ms": res.get("latency_ms", 0.0),
                }

        elif preferred_engine == "paddleocr":
            res = self.paddle.predict_crop(crop)
            if res.get("status") == "SUCCESS":
                val = res.get("text")
                return {
                    "field_name": schema_field_name,
                    "value": val if val else None,
                    "confidence": res.get("confidence"),
                    "source": "paddleocr",
                    "status": "EXTRACTED" if val else "BLANK",
                    "bbox": list(bbox),
                    "latency_ms": res.get("latency_ms", 0.0),
                    "error": None,
                }
            else:
                return {
                    "field_name": schema_field_name,
                    "value": None,
                    "confidence": None,
                    "source": "paddleocr",
                    "status": "ENGINE_UNAVAILABLE" if res.get("status") == "MODEL_UNAVAILABLE" else "FAILED",
                    "error": res.get("error"),
                    "bbox": list(bbox),
                    "latency_ms": res.get("latency_ms", 0.0),
                }

        return {
            "field_name": schema_field_name,
            "value": None,
            "confidence": None,
            "source": "unknown",
            "status": "INVALID_ENGINE",
            "error": f"Unknown engine '{preferred_engine}'",
            "bbox": list(bbox),
        }

    def _extract_digital_form_fields(
        self,
        form_img: np.ndarray,
        template_id: str,
        return_crops: bool = False,
        return_debug_overlay: bool = False,
    ) -> Dict[str, Any]:
        """
        Extracts fields from a digital stacked form profile using PaddleOCR and OpenCV circle analysis.
        """
        start_time = time.time()
        layout = self.template_router.get_template_profile(template_id)
        if not layout:
            return {
                "fields": {f: {"field_name": f, "value": None, "confidence": 0.0, "status": "CONFIG_MISSING"} for f in ALL_SCHEMA_FIELDS},
                "questionnaire_raw": {},
                "status": "FAILED",
                "template_id": template_id,
            }

        results: Dict[str, Dict[str, Any]] = {}
        crops_dict: Dict[str, np.ndarray] = {}
        h_img, w_img = form_img.shape[:2]

        def get_crop(y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
            y1 = max(0, min(h_img - 1, y1))
            y2 = max(y1 + 1, min(h_img, y2))
            x1 = max(0, min(w_img - 1, x1))
            x2 = max(x1 + 1, min(w_img, x2))
            return form_img[y1:y2, x1:x2]

        # 1. Text fields (student_name, school_name, email)
        text_fields_map = [
            ("student_name", "student_name", ["Student Name", "Student", "Name"]),
            ("school_name", "school_name", ["School Name", "School", "Name"]),
            ("email", "email", ["Mail id", "Mail", "Email"]),
        ]

        for schema_f, coord_key, prefixes in text_fields_map:
            if coord_key not in layout:
                continue
            y1, y2, x1, x2 = layout[coord_key]
            crop = get_crop(y1, y2, x1, x2)
            if return_crops:
                crops_dict[schema_f] = crop

            ocr_res = self.paddle.predict(crop, det=True, rec=True)
            if ocr_res.get("status") != "SUCCESS":
                results[schema_f] = {
                    "field_name": schema_f,
                    "value": None,
                    "confidence": None,
                    "source": "paddleocr",
                    "status": "ENGINE_UNAVAILABLE" if ocr_res.get("status") == "MODEL_UNAVAILABLE" else "FAILED",
                    "bbox": [x1, y1, x2, y2],
                    "error": ocr_res.get("error"),
                }
                continue

            items = ocr_res.get("items", [])
            full_text = ocr_res.get("full_text", "")
            confs = [item.get("confidence", 0.0) for item in items]
            avg_conf = float(np.mean(confs)) if confs else 0.0

            extracted = None
            for p in prefixes:
                if p.lower() in full_text.lower():
                    idx = full_text.lower().find(p.lower())
                    raw_val = full_text[idx + len(p):].strip().lstrip(": -_").strip()
                    extracted = clean_text_field(raw_val, schema_f)
                    if extracted:
                        break

            if not extracted:
                val_parts = []
                for item in items:
                    t = item.get("text", "").strip().lstrip(": -_").strip()
                    is_prefix = any(p.lower() in t.lower() for p in prefixes) and len(t) <= max(len(p) for p in prefixes) + 3
                    if not is_prefix and len(t) >= 2:
                        val_parts.append(t)
                extracted = clean_text_field(" ".join(val_parts).strip(), schema_f)

            st = "VALID" if extracted else "BLANK"
            results[schema_f] = {
                "field_name": schema_f,
                "value": extracted,
                "confidence": round(avg_conf, 4) if extracted else 0.0,
                "source": "paddleocr",
                "status": st,
                "bbox": [x1, y1, x2, y2],
                "error": None,
            }

        # 2. Date
        if "date" in layout:
            y1, y2, x1, x2 = layout["date"]
            crop = get_crop(y1, y2, x1, x2)
            if return_crops:
                crops_dict["date"] = crop

            ocr_res = self.paddle.predict(crop, det=True, rec=True)
            items = ocr_res.get("items", [])
            candidates = []
            for item in items:
                text = item.get("text", "").strip()
                conf = float(item.get("confidence", 0.0))
                if text.isdigit() and len(text) == 1 and conf >= 0.70:
                    bx = item.get("bbox", [0, 0, 0, 0])
                    cx = (bx[0] + bx[2]) / 2.0
                    candidates.append((cx, text, conf))

            candidates.sort(key=lambda d: d[0])
            digits = "".join([c[1] for c in candidates])
            avg_conf = float(np.mean([c[2] for c in candidates])) if candidates else 0.0

            if len(digits) == 8:
                formatted_date = f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
                results["date"] = {
                    "field_name": "date",
                    "value": formatted_date,
                    "confidence": round(avg_conf, 4),
                    "source": "paddleocr",
                    "status": "VALID",
                    "bbox": [x1, y1, x2, y2],
                    "error": None,
                }
            elif len(digits) > 0:
                results["date"] = {
                    "field_name": "date",
                    "value": digits,
                    "confidence": round(avg_conf, 4),
                    "source": "paddleocr",
                    "status": "NEEDS_REVIEW",
                    "bbox": [x1, y1, x2, y2],
                    "error": f"Partial date digits ({len(digits)}/8)",
                }
            else:
                results["date"] = {
                    "field_name": "date",
                    "value": None,
                    "confidence": 0.0,
                    "source": "paddleocr",
                    "status": "BLANK",
                    "bbox": [x1, y1, x2, y2],
                    "error": None,
                }

        # 3. Mobile Numbers
        for schema_f, coord_key in [("student_mobile_number", "student_mobile"), ("parent_mobile_number", "parent_mobile")]:
            if coord_key not in layout:
                continue
            y1, y2, x1, x2 = layout[coord_key]
            crop = get_crop(y1, y2, x1, x2)
            if return_crops:
                crops_dict[schema_f] = crop

            ocr_res = self.paddle.predict(crop, det=True, rec=True)
            items = ocr_res.get("items", [])
            candidates = []
            for item in items:
                text = item.get("text", "").strip()
                conf = float(item.get("confidence", 0.0))
                if text.isdigit() and conf >= 0.60:
                    bx = item.get("bbox", [0, 0, 0, 0])
                    box_cx = (bx[0] + bx[2]) / 2.0
                    if len(text) == 1:
                        candidates.append((box_cx, text, conf))
                    else:
                        for idx_ch, ch in enumerate(text):
                            candidates.append((box_cx + (idx_ch - len(text) / 2.0) * 40.0, ch, conf))

            candidates.sort(key=lambda c: c[0])
            filtered = []
            for c in candidates:
                if not filtered or abs(c[0] - filtered[-1][0]) >= 30:
                    filtered.append(c)
                elif c[2] > filtered[-1][2]:
                    filtered[-1] = c

            digits = "".join([c[1] for c in filtered])
            avg_conf = float(np.mean([c[2] for c in filtered])) if filtered else 0.0

            if len(digits) == 10:
                results[schema_f] = {
                    "field_name": schema_f,
                    "value": digits,
                    "confidence": round(avg_conf, 4),
                    "source": "paddleocr",
                    "status": "VALID",
                    "bbox": [x1, y1, x2, y2],
                    "error": None,
                }
            elif len(digits) > 0:
                results[schema_f] = {
                    "field_name": schema_f,
                    "value": digits,
                    "confidence": round(avg_conf, 4),
                    "source": "paddleocr",
                    "status": "NEEDS_REVIEW",
                    "bbox": [x1, y1, x2, y2],
                    "error": f"Partial mobile digits ({len(digits)}/10)",
                }
            else:
                results[schema_f] = {
                    "field_name": schema_f,
                    "value": None,
                    "confidence": 0.0,
                    "source": "paddleocr",
                    "status": "BLANK",
                    "bbox": [x1, y1, x2, y2],
                    "error": None,
                }

        # 4. Cutoff Marks
        if "cutoff_marks" in layout:
            y1, y2, x1, x2 = layout["cutoff_marks"]
            crop = get_crop(y1, y2, x1, x2)
            if return_crops:
                crops_dict["cut_off_marks"] = crop

            ocr_res = self.paddle.predict(crop, det=True, rec=True)
            items = ocr_res.get("items", [])
            candidates = []
            for item in items:
                text = item.get("text", "").strip()
                conf = float(item.get("confidence", 0.0))
                if text.isdigit() and len(text) == 1 and conf >= 0.65:
                    bx = item.get("bbox", [0, 0, 0, 0])
                    cx = (bx[0] + bx[2]) / 2.0
                    candidates.append((cx, text, conf))

            candidates.sort(key=lambda c: c[0])
            digits = "".join([c[1] for c in candidates])
            avg_conf = float(np.mean([c[2] for c in candidates])) if candidates else 0.0

            if len(digits) == 5:
                val = f"{digits[:3]}.{digits[3:]}"
                results["cut_off_marks"] = {
                    "field_name": "cut_off_marks",
                    "value": val,
                    "confidence": round(avg_conf, 4),
                    "source": "paddleocr",
                    "status": "VALID",
                    "bbox": [x1, y1, x2, y2],
                    "error": None,
                }
            elif len(digits) > 0:
                results["cut_off_marks"] = {
                    "field_name": "cut_off_marks",
                    "value": digits,
                    "confidence": round(avg_conf, 4),
                    "source": "paddleocr",
                    "status": "NEEDS_REVIEW",
                    "bbox": [x1, y1, x2, y2],
                    "error": f"Partial cutoff digits ({len(digits)}/5)",
                }
            else:
                results["cut_off_marks"] = {
                    "field_name": "cut_off_marks",
                    "value": None,
                    "confidence": 0.0,
                    "source": "paddleocr",
                    "status": "BLANK",
                    "bbox": [x1, y1, x2, y2],
                    "error": None,
                }

        # 5. Questionnaire Circles
        gray = cv2.cvtColor(form_img, cv2.COLOR_BGR2GRAY) if len(form_img.shape) == 3 else form_img
        _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        def fr(cx: int, cy: int, r: int = 14) -> float:
            patch = bin_img[cy-r:cy+r, cx-r:cx+r]
            if patch.size == 0:
                return 0.0
            mask = np.zeros_like(patch)
            cv2.circle(mask, (r, r), r-2, 255, -1)
            return float(np.sum((patch == 255) & (mask == 255))) / float(max(1, np.sum(mask == 255)))

        q_coord_map = {
            "interested_in_bit": ("q1_boxes", "q1"),
            "career_guidance_usefulness": ("q2_boxes", "q2"),
            "speaker_effectiveness": ("q3_boxes", "q3"),
            "interested_in_choice_filling_assistance": ("q4_boxes", "q4"),
        }

        rating_num_to_verbal = {
            "5": "Excellent",
            "4": "Good",
            "3": "Satisfactory",
            "2": "Fair",
            "1": "Poor",
        }

        questionnaire_raw = {}
        debug_overlay = form_img.copy() if return_debug_overlay else None

        for schema_f, (coord_key, q_id) in q_coord_map.items():
            opts = layout.get(coord_key, {})
            scores = {str(opt_k): fr(int(c[0]), int(c[1])) for opt_k, c in opts.items()}
            s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_k, top_v = s[0] if s else (None, 0.0)
            sec_v = s[1][1] if len(s) > 1 else 0.0

            if top_v < 0.235:
                dec, st = None, "UNANSWERED"
                conf = 0.0
            elif top_v >= 0.24 and sec_v >= 0.24 and (top_v - sec_v) < 0.05:
                dec, st = "AMBIGUOUS", "AMBIGUOUS"
                conf = 0.50
            else:
                # Map numeric rating to verbal label if applicable
                dec = rating_num_to_verbal.get(top_k, top_k)
                st = "VALID"
                margin = top_v - sec_v
                conf = min(0.99, max(0.75, round(0.75 + (margin / 0.35) * 0.24, 4)))

            results[schema_f] = {
                "field_name": schema_f,
                "value": dec if st == "VALID" else None,
                "confidence": conf,
                "source": "bubble_detector",
                "status": st,
                "densities": scores,
                "raw_fill_density": round(top_v, 4),
                "question_id": q_id,
                "error": None if st == "VALID" else f"Question status: {st}",
            }
            questionnaire_raw[q_id] = {
                "question_id": q_id,
                "selected_option": dec if st == "VALID" else None,
                "status": st,
                "confidence": conf,
                "raw_fill_density": round(top_v, 4),
                "densities": scores,
            }

            if debug_overlay is not None:
                for opt_k, (cx, cy) in opts.items():
                    color = (0, 255, 0) if opt_k == top_k and st == "VALID" else (0, 0, 255)
                    cv2.circle(debug_overlay, (int(cx), int(cy)), 16, color, 2)

        total_latency = (time.time() - start_time) * 1000.0

        return {
            "fields": results,
            "questionnaire_raw": questionnaire_raw,
            "crops": crops_dict if return_crops else None,
            "debug_image": debug_overlay,
            "status": "SUCCESS",
            "template_id": template_id,
            "total_latency_ms": round(total_latency, 2),
        }

    def _extract_a4_form_fields(
        self,
        aligned_image: np.ndarray,
        return_crops: bool = False,
        return_debug_overlay: bool = False,
    ) -> Dict[str, Any]:
        """
        Original A4 template extraction using TrOCR for handwriting and BubbleDetector for questionnaire.
        """
        start_time = time.time()
        results: Dict[str, Dict[str, Any]] = {}
        crops_dict: Dict[str, np.ndarray] = {}

        # 1. Handwritten text fields -> TrOCR
        for f_name in ["student_name", "school_name"]:
            res = self.extract_text_field(aligned_image, f_name, preferred_engine="trocr")
            results[f_name] = res
            if return_crops:
                tmpl_key = SCHEMA_TO_TEMPLATE_MAP.get(f_name, f_name)
                c, _, _ = self.crop_field(aligned_image, self.config.get("fields", {}).get(tmpl_key, {}))
                if c is not None:
                    crops_dict[f_name] = c

        # 2. Structured fields -> TrOCR / PaddleOCR
        structured_fields = ["date", "student_mobile_number", "parent_mobile_number", "email", "cut_off_marks"]
        for f_name in structured_fields:
            res = self.extract_text_field(aligned_image, f_name, preferred_engine="trocr")
            results[f_name] = res
            if return_crops:
                tmpl_key = SCHEMA_TO_TEMPLATE_MAP.get(f_name, f_name)
                c, _, _ = self.crop_field(aligned_image, self.config.get("fields", {}).get(tmpl_key, {}))
                if c is not None:
                    crops_dict[f_name] = c

        # 3. Questionnaire bubbles -> BubbleDetector
        q_results = self.bubble_detector.process_questionnaire(
            aligned_image,
            return_debug_image=return_debug_overlay,
        )

        for q_id, schema_field in QUESTION_ID_TO_SCHEMA_MAP.items():
            q_res = q_results["questions"].get(q_id, {})
            selected = q_res.get("selected_option")
            q_status = q_res.get("status", "UNANSWERED")

            results[schema_field] = {
                "field_name": schema_field,
                "value": selected,
                "confidence": q_res.get("confidence", 0.0),
                "source": "bubble_detector",
                "status": "VALID" if q_status == "SELECTED" else q_status,
                "densities": q_res.get("densities", {}),
                "question_id": q_id,
                "error": None if q_status == "SELECTED" else f"Bubble status: {q_status}",
            }

        total_latency_ms = (time.time() - start_time) * 1000.0

        return {
            "fields": results,
            "questionnaire_raw": q_results["questions"],
            "crops": crops_dict if return_crops else None,
            "debug_image": q_results.get("debug_image"),
            "status": "SUCCESS",
            "template_id": TEMPLATE_HANDWRITTEN_A4,
            "total_latency_ms": round(total_latency_ms, 2),
        }

    def extract_all_fields(
        self,
        aligned_image: np.ndarray,
        form_index: Optional[int] = None,
        is_split: bool = False,
        return_crops: bool = False,
        return_debug_overlay: bool = False,
    ) -> Dict[str, Any]:
        """
        Classifies template layout and extracts all fields accordingly.
        """
        # If user explicitly passed a custom template configuration (e.g., in unit tests), use A4 flow
        if self.custom_template_passed:
            return self._extract_a4_form_fields(
                aligned_image, return_crops=return_crops, return_debug_overlay=return_debug_overlay
            )

        # Identify layout profile
        tmpl_info = self.template_router.identify_template(
            aligned_image, form_index=form_index, is_split=is_split
        )
        template_id = tmpl_info.get("template_id", UNKNOWN_TEMPLATE)

        if template_id in (TEMPLATE_DIGITAL_FORM_1, TEMPLATE_DIGITAL_FORM_2):
            return self._extract_digital_form_fields(
                aligned_image,
                template_id=template_id,
                return_crops=return_crops,
                return_debug_overlay=return_debug_overlay,
            )

        elif template_id == TEMPLATE_HANDWRITTEN_A4:
            return self._extract_a4_form_fields(
                aligned_image, return_crops=return_crops, return_debug_overlay=return_debug_overlay
            )

        else:
            # Explicit UNKNOWN_TEMPLATE rejection
            logger.warning("Extraction aborted: %s", tmpl_info.get("reason"))
            unknown_fields = {
                f: {
                    "field_name": f,
                    "value": None,
                    "confidence": 0.0,
                    "source": "none",
                    "status": UNKNOWN_TEMPLATE,
                    "error": tmpl_info.get("reason"),
                }
                for f in ALL_SCHEMA_FIELDS
            }
            return {
                "fields": unknown_fields,
                "questionnaire_raw": {},
                "crops": None,
                "debug_image": None,
                "status": UNKNOWN_TEMPLATE,
                "template_id": UNKNOWN_TEMPLATE,
                "error": tmpl_info.get("reason"),
                "total_latency_ms": 0.0,
            }
