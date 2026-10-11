"""
OpenCV-based Bubble and Response Circle Detector.

This module inspects candidate circular regions on an aligned feedback form
using OpenCV image processing techniques (inner-circle circular mask extraction,
local paper background contrast normalization, watermark baseline compensation,
and multi-threshold evaluation) to reliably distinguish marked from unmarked
answer options for Yes/No and 1-5 rating questions.

Design Principles:
1. Pure Image Processing: Strictly uses OpenCV & NumPy (NO text OCR for bubble selection).
2. Local Background Normalization: Measures surrounding annular paper intensity to be robust
   against non-uniform illumination and camera shadows.
3. Watermark Baseline Compensation: Compares inner ink density against printed letter
   watermarks (A, B, C, D) so that blank forms correctly evaluate to UNANSWERED.
4. Inner Core Mask Isolation: Masks out printed outer circle borders (0.65x radius) to avoid
   border bleed during small alignment variations.
5. Defensible Status Classification: Distinguishes SELECTED, UNANSWERED, MULTIPLE_MARKS, and AMBIGUOUS.
6. Visual Debug Support: Generates annotated diagnostic overlay images with color-coded boundaries.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.app.core.config import get_form_template_config

logger = logging.getLogger(__name__)

# Default watermark baselines for printed letter watermarks inside option bubbles
DEFAULT_WATERMARK_BASELINES: Dict[str, float] = {
    "Yes": 0.20,
    "No": 0.26,
    "Excellent": 0.20,
    "Good": 0.26,
    "Satisfactory": 0.19,
    "Fair": 0.24,
    "Poor": 0.00,
    "A": 0.20,
    "B": 0.26,
    "C": 0.19,
    "D": 0.24,
}


class BubbleDetector:
    """
    OpenCV response circle detector for feedback form questionnaires.
    """

    def __init__(
        self,
        template_config: Optional[Dict[str, Any]] = None,
        filled_threshold: float = 0.35,
        review_threshold: float = 0.22,
        margin_threshold: float = 0.08,
        inner_radius_ratio: float = 0.65,
        contrast_threshold: int = 35,
        watermark_baselines: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Parameters:
            template_config: Optional form template dict. If None, loaded from yaml config.
            filled_threshold: Minimum raw ink density ratio to consider a bubble filled.
            review_threshold: Raw ink density above which a bubble enters review/borderline state.
            margin_threshold: Required density gap between top candidate and runner-up for unambiguous selection.
            inner_radius_ratio: Fraction of outer radius to sample inside inner circle (avoids printed outer border).
            contrast_threshold: Minimum brightness difference between local paper background and inner pixel to count as ink.
            watermark_baselines: Dict mapping option labels to their printed watermark baseline density.
        """
        self.config = template_config or get_form_template_config()
        self.filled_threshold = filled_threshold
        self.review_threshold = review_threshold
        self.margin_threshold = margin_threshold
        self.inner_radius_ratio = inner_radius_ratio
        self.contrast_threshold = contrast_threshold
        self.watermark_baselines = watermark_baselines or DEFAULT_WATERMARK_BASELINES

    def calculate_bubble_density(
        self,
        crop: np.ndarray,
        radius: int,
        label: str = "",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates ink density ratio inside the inner circular core of a bubble crop,
        normalized against local paper background brightness and printed watermark baselines.

        Parameters:
            crop: BGR or Grayscale image array centered on the bubble (with padding).
            radius: Pixel radius of the bubble.
            label: Option label (e.g., 'Yes', 'No', 'Good') for watermark compensation.

        Returns:
            Tuple of (net_density, debug_meta).
        """
        if crop is None or crop.size == 0:
            return 0.0, {"error": "Empty crop"}

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]
        if h < 3 or w < 3:
            return 0.0, {"error": f"Crop dimensions ({w}x{h}) too small"}

        cx, cy = w // 2, h // 2

        # 1. Measure local paper background using annular ring around the bubble (1.3*r to 2.0*r)
        bg_mask = np.zeros((h, w), dtype=np.uint8)
        outer_bg_r = min(int(radius * 2.0), min(cx, cy, w - cx, h - cy))
        inner_bg_r = int(radius * 1.3)
        if outer_bg_r > inner_bg_r:
            cv2.circle(bg_mask, (cx, cy), outer_bg_r, 255, -1)
            cv2.circle(bg_mask, (cx, cy), inner_bg_r, 0, -1)

        bg_pixels = gray[bg_mask > 0]
        if bg_pixels.size == 0 or np.mean(bg_pixels) < 40:
            # Fallback or inside black unmapped border
            local_paper_brightness = float(np.percentile(gray, 80)) if gray.size > 0 else 255.0
            is_unmapped = bool(bg_pixels.size > 0 and np.mean(bg_pixels) < 40)
        else:
            local_paper_brightness = float(np.percentile(bg_pixels, 75))
            is_unmapped = False

        if is_unmapped:
            return 0.0, {
                "error": "Unmapped warped boundary",
                "raw_density": 0.0,
                "net_density": 0.0,
                "mean_darkness": 0.0,
            }

        # 2. Measure ink in inner core mask
        inner_r = max(2, int(radius * self.inner_radius_ratio))
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), inner_r, 255, -1)
        inner_pixels = gray[inner_mask > 0]

        total_inner_pixels = int(np.sum(inner_mask > 0))
        if total_inner_pixels == 0:
            return 0.0, {"error": "Zero mask pixels"}

        # Pixels significantly darker than local background
        ink_pixels = int(np.sum((local_paper_brightness - inner_pixels) > self.contrast_threshold))
        raw_density = float(ink_pixels / total_inner_pixels)

        # 3. Watermark compensation
        watermark = self.watermark_baselines.get(label, 0.15)
        net_density = max(0.0, raw_density - watermark)

        # Also measure mean grayscale intensity in inner region (darker = more ink)
        mean_darkness = float(max(0.0, (local_paper_brightness - np.mean(inner_pixels)) / local_paper_brightness)) if inner_pixels.size > 0 else 0.0

        meta = {
            "inner_radius": inner_r,
            "total_pixels": total_inner_pixels,
            "ink_pixels": ink_pixels,
            "raw_density": round(raw_density, 4),
            "net_density": round(net_density, 4),
            "watermark_baseline": watermark,
            "local_paper_brightness": round(local_paper_brightness, 2),
            "mean_darkness": round(mean_darkness, 4),
        }

        return round(raw_density, 4), meta

    def extract_bubble_crop(
        self,
        aligned_image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
    ) -> Optional[np.ndarray]:
        """
        Extracts a square crop centered on a bubble with safety margin.
        """
        h_img, w_img = aligned_image.shape[:2]
        cx, cy = center
        pad = int(radius * 2.2)

        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(w_img, cx + pad)
        y2 = min(h_img, cy + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = aligned_image[y1:y2, x1:x2]
        return crop

    def process_question(
        self,
        aligned_image: np.ndarray,
        question_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluates all option bubbles for a single question.

        Parameters:
            aligned_image: Warped/aligned form image array.
            question_cfg: Question configuration containing question_id, label, and options.

        Returns:
            Structured evaluation dict.
        """
        h_img, w_img = aligned_image.shape[:2]
        q_id = question_cfg.get("question_id", "unknown")
        q_label = question_cfg.get("label", "")
        options = question_cfg.get("options", [])

        densities: Dict[str, float] = {}
        net_densities: Dict[str, float] = {}
        options_details: List[Dict[str, Any]] = []

        for opt in options:
            label = opt.get("label", "")
            if "center_normalized" in opt:
                norm_cx, norm_cy = opt["center_normalized"]
                cx = int(norm_cx * w_img)
                cy = int(norm_cy * h_img)
            elif "center" in opt:
                cx, cy = opt["center"]
            else:
                cx, cy = 0, 0

            if "radius_normalized" in opt:
                radius = max(4, int(opt["radius_normalized"] * min(w_img, h_img)))
            elif "radius" in opt:
                radius = int(opt["radius"])
            else:
                radius = 28

            crop = self.extract_bubble_crop(aligned_image, (cx, cy), radius)
            raw_density, meta = self.calculate_bubble_density(crop, radius, label=label)
            net_density = meta.get("net_density", 0.0)

            densities[label] = raw_density
            net_densities[label] = net_density

            watermark = self.watermark_baselines.get(label, 0.15)
            # A bubble is filled if it meets both filled_threshold and net_density above watermark baseline
            is_marked = bool(raw_density >= self.filled_threshold and net_density >= 0.18)
            is_review = bool(not is_marked and (net_density >= 0.08 or (watermark == 0.0 and raw_density >= self.review_threshold)))

            options_details.append({
                "label": label,
                "density": raw_density,
                "net_density": net_density,
                "center": [cx, cy],
                "radius": radius,
                "is_marked": is_marked,
                "is_review": is_review,
                "meta": meta,
            })

        # Classify selection
        sorted_opts = sorted(options_details, key=lambda x: x["net_density"], reverse=True)
        top1 = sorted_opts[0] if sorted_opts else None
        top2 = sorted_opts[1] if len(sorted_opts) > 1 else None

        marked_options = [o for o in options_details if o["is_marked"]]
        review_options = [o for o in options_details if o["is_review"] and not o["is_marked"]]

        selected_option: Optional[str] = None
        status: str = "UNANSWERED"
        confidence: float = 0.0

        if len(marked_options) == 1:
            top_opt = marked_options[0]
            top_net = top_opt["net_density"]
            second_net = top2["net_density"] if (top2 and top2 != top_opt) else 0.0
            margin = top_net - second_net

            if margin >= self.margin_threshold:
                selected_option = top_opt["label"]
                status = "SELECTED"
                # Confidence is a calibrated function of net signal and margin over runner-up
                confidence = round(min(0.99, max(0.70, 0.70 + (top_net - 0.18) * 0.4 + margin * 0.4)), 4)
            else:
                selected_option = top_opt["label"]
                status = "AMBIGUOUS"
                confidence = round(top_net, 4)

        elif len(marked_options) > 1:
            status = "MULTIPLE_MARKS"
            selected_option = None
            confidence = 0.0

        elif len(review_options) > 0:
            best_rev = max(review_options, key=lambda x: x["net_density"])
            status = "AMBIGUOUS"
            selected_option = best_rev["label"]
            confidence = round(best_rev["net_density"], 4)

        else:
            status = "UNANSWERED"
            selected_option = None
            confidence = 0.0

        return {
            "question_id": q_id,
            "label": q_label,
            "selected_option": selected_option,
            "status": status,
            "confidence": confidence,
            "densities": densities,
            "net_densities": net_densities,
            "options_details": options_details,
        }

    def process_questionnaire(
        self,
        aligned_image: np.ndarray,
        return_debug_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes all questions defined in the form template questionnaire.

        Parameters:
            aligned_image: Warped form image array.
            return_debug_image: Whether to generate and return an annotated visualization.

        Returns:
            Dict containing:
                - questions: Dict mapping question_id to question result dict.
                - answers: Dict mapping question_id to selected_option value.
                - all_answered: bool
                - has_ambiguous: bool
                - debug_image: Optional annotated BGR image.
        """
        q_cfg = self.config.get("questionnaire", {})
        questions = q_cfg.get("questions", [])

        results: Dict[str, Dict[str, Any]] = {}
        answers: Dict[str, Optional[str]] = {}
        has_ambiguous = False
        all_answered = True

        for q in questions:
            q_id = q.get("question_id", "unknown")
            res = self.process_question(aligned_image, q)
            results[q_id] = res
            answers[q_id] = res["selected_option"]

            if res["status"] in ("AMBIGUOUS", "MULTIPLE_MARKS"):
                has_ambiguous = True
            if res["status"] in ("UNANSWERED", "MULTIPLE_MARKS"):
                all_answered = False

        debug_img: Optional[np.ndarray] = None
        if return_debug_image:
            debug_img = self.draw_debug_overlay(aligned_image, results)

        return {
            "questions": results,
            "answers": answers,
            "all_answered": all_answered,
            "has_ambiguous": has_ambiguous,
            "debug_image": debug_img,
        }

    def draw_debug_overlay(
        self,
        aligned_image: np.ndarray,
        question_results: Dict[str, Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draws visual diagnostics on top of the aligned image highlighting
        detected circles, fill status, and net percentage densities.
        """
        overlay = aligned_image.copy()

        for q_id, q_res in question_results.items():
            q_status = q_res.get("status", "UNANSWERED")
            selected = q_res.get("selected_option")

            for opt in q_res.get("options_details", []):
                cx, cy = opt["center"]
                r = opt["radius"]
                raw_d = opt["density"]
                net_d = opt.get("net_density", raw_d)
                label = opt["label"]
                is_selected = (label == selected and q_status == "SELECTED")

                # Color coding: Green (Selected), Red (Multiple), Amber (Ambiguous), Gray (Unanswered)
                if is_selected:
                    color = (0, 200, 0)  # Green
                    thickness = 3
                elif q_status == "MULTIPLE_MARKS" and opt["is_marked"]:
                    color = (0, 0, 220)  # Red
                    thickness = 3
                elif q_status == "AMBIGUOUS" and label == selected:
                    color = (0, 180, 255)  # Amber
                    thickness = 2
                else:
                    color = (180, 180, 180)  # Gray
                    thickness = 1

                # Draw outer circle
                cv2.circle(overlay, (cx, cy), r, color, thickness)
                # Draw inner test core
                inner_r = max(2, int(r * self.inner_radius_ratio))
                cv2.circle(overlay, (cx, cy), inner_r, color, 1)

                # Draw label and percentage text
                text = f"{label}:{int(net_d * 100)}%"
                cv2.putText(
                    overlay,
                    text,
                    (cx - r, cy - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return overlay
