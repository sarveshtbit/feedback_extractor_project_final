import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from backend.app.core.config import get_form_template_config

class ROIExtractor:
    def __init__(self, template_config: Optional[Dict[str, Any]] = None):
        self.config = template_config or get_form_template_config()

    @staticmethod
    def validate_crop(
        crop: Optional[np.ndarray],
        img_w: int,
        img_h: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        min_dim: int = 5
    ) -> Tuple[bool, str]:
        """
        Validates that crop parameters stay inside image bounds, are non-empty, and meet resolution threshold.
        Returns (is_valid, reason).
        """
        if crop is None or crop.size == 0:
            return False, "Crop is None or zero-sized array"

        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            return False, f"Crop boundary [{x1}, {y1}, {x2}, {y2}] exceeds image dimensions ({img_w}x{img_h})"

        w = x2 - x1
        h = y2 - y1
        if w < min_dim or h < min_dim:
            return False, f"Crop dimensions ({w}x{h}) below minimum required resolution threshold ({min_dim}px)"

        # Check if crop is monochromatic / empty white canvas
        if len(crop.shape) == 3:
            std_dev = np.std(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        else:
            std_dev = np.std(crop)

        if std_dev < 0.5:
            return False, f"Crop has zero variance / blank background (std_dev={std_dev:.2f})"

        return True, "Valid"

    def crop_single_field(
        self,
        aligned_image: np.ndarray,
        field_name: str,
        page_num: int = 1
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Targeted on-demand single field crop generator.
        Generates and validates crop ONLY when requested for targeted inspection.
        Returns (crop_image, metadata).
        """
        h, w = aligned_image.shape[:2]
        fields_cfg = self.config.get("fields", {})

        # Map schema field name to template key if required
        tmpl_key = field_name
        if field_name not in fields_cfg:
            for k in fields_cfg.keys():
                if k in field_name or field_name in k:
                    tmpl_key = k
                    break

        cfg = fields_cfg.get(tmpl_key, {})
        if not cfg:
            return None, {
                "field": field_name,
                "page": page_num,
                "x": 0, "y": 0, "width": 0, "height": 0,
                "is_valid": False,
                "reason": f"Field '{field_name}' not defined in form template"
            }

        field_type = cfg.get("type")

        if "normalized_bbox" in cfg:
            x1_n, y1_n, x2_n, y2_n = cfg["normalized_bbox"]
            x1, y1 = int(x1_n * w), int(y1_n * h)
            x2, y2 = int(x2_n * w), int(y2_n * h)
        elif "bounding_box" in cfg:
            bx, by, bw, bh = cfg["bounding_box"]
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        elif field_type == "character_boxes" and "boxes" in cfg and cfg["boxes"]:
            boxes = cfg["boxes"]
            first_box = boxes[0].get("normalized_bbox") or [0, 0, 0, 0]
            last_box = boxes[-1].get("normalized_bbox") or [0, 0, 0, 0]
            x1, y1 = int(first_box[0] * w), int(first_box[1] * h)
            x2, y2 = int(last_box[2] * w), int(last_box[3] * h)
        else:
            x1, y1, x2, y2 = 0, 0, w, h

        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w, x2), min(h, y2)
        crop = aligned_image[y1_c:y2_c, x1_c:x2_c]

        is_valid, reason = self.validate_crop(crop, w, h, x1_c, y1_c, x2_c, y2_c)
        metadata = {
            "field": field_name,
            "page": page_num,
            "x": x1_c,
            "y": y1_c,
            "width": x2_c - x1_c,
            "height": y2_c - y1_c,
            "is_valid": is_valid,
            "reason": reason
        }

        if is_valid:
            return crop, metadata
        return None, metadata

    def crop_rois(self, aligned_image: np.ndarray) -> Dict[str, Any]:
        """
        Crops all form fields from aligned canvas image using template coordinates.
        Supports character boxes, text regions, and questionnaire checkboxes/bubbles.
        """
        h, w = aligned_image.shape[:2]
        fields_cfg = self.config.get("fields", {})
        cropped = {}

        for field_name, cfg in fields_cfg.items():
            fieldType = cfg.get("type")
            
            if fieldType == "character_boxes":
                boxes_crops = []
                for box_item in cfg.get("boxes", []):
                    idx = box_item.get("index", 0)
                    box_type = box_item.get("type", "digit")
                    
                    if "normalized_bbox" in box_item:
                        x1_n, y1_n, x2_n, y2_n = box_item["normalized_bbox"]
                        x1, y1 = int(x1_n * w), int(y1_n * h)
                        x2, y2 = int(x2_n * w), int(y2_n * h)
                    elif "box" in box_item:
                        bx, by, bw, bh = box_item["box"]
                        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                    else:
                        continue

                    crop = aligned_image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    boxes_crops.append({
                        "index": idx,
                        "type": box_type,
                        "crop": crop,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "normalized_bbox": [x1 / w, y1 / h, x2 / w, y2 / h]
                    })
                cropped[field_name] = {"boxes": boxes_crops}

            elif fieldType == "text_region":
                if "normalized_bbox" in cfg:
                    x1_n, y1_n, x2_n, y2_n = cfg["normalized_bbox"]
                    x1, y1 = int(x1_n * w), int(y1_n * h)
                    x2, y2 = int(x2_n * w), int(y2_n * h)
                elif "bounding_box" in cfg:
                    bx, by, bw, bh = cfg["bounding_box"]
                    x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                else:
                    continue

                crop = aligned_image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                cropped[field_name] = {
                    "crop": crop,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "normalized_bbox": [x1 / w, y1 / h, x2 / w, y2 / h]
                }

            elif fieldType in ("checkbox_group", "omr_group"):
                questions_out = []
                for q in cfg.get("questions", []):
                    q_id = q.get("question_id")
                    q_label = q.get("label")
                    opts_out = []
                    for opt in q.get("options", []):
                        opt_label = opt.get("label")
                        if "center_normalized" in opt:
                            cx_n, cy_n = opt["center_normalized"]
                            r_n = opt.get("radius_normalized", 0.01)
                            cx, cy = int(cx_n * w), int(cy_n * h)
                            r = int(r_n * w)
                        elif "center" in opt:
                            cx, cy = opt["center"]
                            r = opt.get("radius", 25)
                        else:
                            continue

                        x1, y1 = max(0, cx - r - 5), max(0, cy - r - 5)
                        x2, y2 = min(w, cx + r + 5), min(h, cy + r + 5)
                        bubble_crop = aligned_image[y1:y2, x1:x2]
                        opts_out.append({
                            "label": opt_label,
                            "crop": bubble_crop,
                            "center": [cx, cy],
                            "radius": r,
                            "bbox": [x1, y1, x2 - x1, y2 - y1]
                        })
                    questions_out.append({
                        "question_id": q_id,
                        "label": q_label,
                        "options": opts_out
                    })
                cropped[field_name] = {"questions": questions_out}

        return cropped
