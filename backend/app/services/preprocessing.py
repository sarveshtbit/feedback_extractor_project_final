import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from typing import Dict, Tuple, Any, Optional
from backend.app.core.config import get_form_template_config

class ImagePreprocessor:
    @staticmethod
    def load_and_fix_exif(file_bytes: bytes) -> np.ndarray:
        """
        Loads image bytes, applies EXIF rotation fix via PIL once, and converts to OpenCV BGR format.
        """
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr

    @staticmethod
    def deskew_image(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates skew angle via Minimum Area Bounding Box on Otsu thresholding and rotates image if angle is valid.
        Ignores angles < 0.5 deg or > 15.0 deg to prevent unnecessary rotation.
        """
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return gray, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5 or abs(angle) > 15.0:
            return gray, 0.0  # Ignore negligible or extreme angles

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    @staticmethod
    def generate_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates multi-variant images optimized for OCR and vision while explicitly safeguarding thin handwriting strokes:
        - original: High-res BGR original
        - grayscale: Standard 8-bit grayscale
        - enhanced: CLAHE contrast-enhanced grayscale
        - thresholded: Soft adaptive Gaussian thresholding (preserves thin pen strokes)
        - sharpened: Unsharp-mask sharpened image for faint lines
        - deskewed: Deskewed grayscale variant
        - high_contrast: Contrast stretched grayscale (preserves faint handwriting ink)
        - denoised: Bilateral filtered grayscale (removes paper noise while preserving stroke edges)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. CLAHE Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Soft Adaptive Thresholding (large block size to protect thin handwriting strokes)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        thresholded = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
        )

        # 3. Sharpening via Unsharp Masking
        gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
        sharpened = cv2.addWeighted(gray, 1.4, gaussian_blur, -0.4, 0)

        # 4. Deskewed variant
        deskewed_gray, _ = ImagePreprocessor.deskew_image(gray)

        # 5. High-contrast percentile normalization
        p2, p98 = np.percentile(gray, (2, 98))
        high_contrast = np.clip((gray - p2) * (255.0 / max(1.0, p98 - p2)), 0, 255).astype(np.uint8)

        return {
            "original": image,
            "grayscale": gray,
            "enhanced": enhanced,
            "thresholded": thresholded,
            "sharpened": sharpened,
            "deskewed": deskewed_gray,
            "high_contrast": high_contrast,
            "denoised": denoised
        }

    @staticmethod
    def map_bbox_norm(bbox_norm: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """
        Maps normalized [x1_n, y1_n, x2_n, y2_n] to pixel coordinates.
        Accounts for letterboxing/padding if image aspect ratio differs from canonical form.
        """
        x1_n, y1_n, x2_n, y2_n = bbox_norm
        content_aspect = 1280.0 / 904.0
        img_aspect = img_w / float(img_h) if img_h > 0 else content_aspect

        if abs(img_aspect - content_aspect) > 0.1 and img_w > 1500 and img_h > 2000:
            scale = min(img_w / 1280.0, img_h / 904.0)
            new_w = int(1280.0 * scale)
            new_h = int(904.0 * scale)
            pad_x = (img_w - new_w) // 2
            pad_y = (img_h - new_h) // 2

            x1 = pad_x + int(x1_n * new_w)
            y1 = pad_y + int(y1_n * new_h)
            x2 = pad_x + int(x2_n * new_w)
            y2 = pad_y + int(y2_n * new_h)
        else:
            x1 = int(x1_n * img_w)
            y1 = int(y1_n * img_h)
            x2 = int(x2_n * img_w)
            y2 = int(y2_n * img_h)

        return x1, y1, x2, y2

    @staticmethod
    def draw_roi_overlay(image: np.ndarray, template_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Draws colored bounding boxes and text labels for form fields onto a copy of the image.
        Used for preprocessing debug inspection (06_ROI_overlay.png).
        """
        config = template_config or get_form_template_config()
        overlay = image.copy()
        h, w = overlay.shape[:2]
        fields_cfg = config.get("fields", {})

        color_map = {
            "student_name": (0, 0, 255),        # Red
            "school_name": (0, 165, 255),      # Orange
            "date": (255, 0, 0),               # Blue
            "student_mobile": (0, 255, 0),     # Green
            "parent_mobile": (255, 255, 0),    # Cyan
            "email": (255, 0, 255),            # Magenta
            "cutoff_marks": (0, 255, 255),     # Yellow
            "questionnaire": (128, 0, 128)     # Purple
        }

        for field_name, cfg in fields_cfg.items():
            color = color_map.get(field_name, (0, 255, 0))
            fieldType = cfg.get("type")

            if fieldType == "character_boxes":
                for box_item in cfg.get("boxes", []):
                    if "normalized_bbox" in box_item:
                        x1, y1, x2, y2 = ImagePreprocessor.map_bbox_norm(box_item["normalized_bbox"], w, h)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                if "normalized_bbox" in cfg:
                    x1, y1, _, _ = ImagePreprocessor.map_bbox_norm(cfg["normalized_bbox"], w, h)
                    cv2.putText(overlay, field_name.upper(), (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif fieldType == "text_region":
                if "normalized_bbox" in cfg:
                    x1, y1, x2, y2 = ImagePreprocessor.map_bbox_norm(cfg["normalized_bbox"], w, h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(overlay, field_name.upper(), (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif fieldType in ("checkbox_group", "omr_group"):
                for q in cfg.get("questions", []):
                    q_id = q.get("question_id", "")
                    for opt in q.get("options", []):
                        if "center_normalized" in opt:
                            cx_n, cy_n = opt["center_normalized"]
                            r_n = opt.get("radius_normalized", 0.01)
                            cx, cy = int(cx_n * w), int(cy_n * h)
                            r = int(r_n * w)
                            cv2.circle(overlay, (cx, cy), r, color, 2)
                            cv2.putText(overlay, f"{q_id}:{opt.get('label')}", (cx - r, cy - r - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return overlay

    @staticmethod
    def save_debug_stages(output_dir: str, stages: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Saves preprocessing stage images for debug visualizer.
        Returns dictionary of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        stage_filenames = {
            "01_original": "01_original.png",
            "02_orientation_corrected": "02_orientation_corrected.png",
            "03_deskewed": "03_deskewed.png",
            "04_perspective_corrected": "04_perspective_corrected.png",
            "05_final_preprocessed": "05_final_preprocessed.png",
            "06_ROI_overlay": "06_ROI_overlay.png"
        }

        for stage_key, filename in stage_filenames.items():
            if stage_key in stages and stages[stage_key] is not None:
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, stages[stage_key])
                saved_paths[stage_key] = filepath

        return saved_paths
