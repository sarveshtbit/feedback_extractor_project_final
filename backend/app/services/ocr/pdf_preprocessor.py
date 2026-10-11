"""
Reusable PDF Preprocessor and Multi-Form Detection Engine.

Provides robust, production-grade PDF rendering, image deskewing, and
intelligent form boundary detection for feedback forms:
1. Multi-page & single-page PDF rendering at 300 DPI via PyMuPDF.
2. Fine-grained image deskewing using sub-degree Hough transform analysis.
3. Multi-form stacked layout detection with evidence verification (whitespace bands,
   bimodal content mass, minimum form height requirements).
4. Preservation of page, form, and document ordering.
5. Graceful handling of single-form pages, multi-page PDFs, blank pages, and malformed inputs.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pymupdf as fitz

logger = logging.getLogger(__name__)


class PDFPreprocessor:
    """
    Orchestrates PDF ingestion, page rendering, deskewing, and multi-form splitting.
    """

    @staticmethod
    def deskew_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Estimates skew angle from horizontal rule lines and rotates image back to 0.0 degrees.
        
        Parameters:
            img: OpenCV BGR or Grayscale image array.
            
        Returns:
            Tuple of (deskewed_image, detected_skew_angle_degrees).
        """
        if img is None or img.size == 0:
            return img, 0.0

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 1800,
            threshold=150,
            minLineLength=200,
            maxLineGap=10,
        )

        angles: List[float] = []
        if lines is not None:
            for l in lines:
                if len(l.shape) > 0 and l.shape[0] == 4:
                    x1, y1, x2, y2 = l
                else:
                    x1, y1, x2, y2 = l[0]
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if abs(dx) > 150:
                    angle = np.degrees(np.arctan2(dy, dx))
                    if abs(angle) < 5.0:
                        angles.append(angle)

        skew = float(np.median(angles)) if angles else 0.0
        h, w = img.shape[:2]

        if abs(skew) > 0.05:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew, 1.0)
            deskewed = cv2.warpAffine(
                img,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255) if len(img.shape) == 3 else 255,
            )
            return deskewed, skew

        return img, 0.0

    @staticmethod
    def is_blank_image(img: np.ndarray, min_ink_pixels: int = 500) -> bool:
        """
        Checks if an image is essentially blank (e.g. empty white page).
        A page is considered blank if it has fewer than min_ink_pixels dark pixels.
        """
        if img is None or img.size == 0:
            return True
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        dark_pixels = np.sum(gray < 240)
        return int(dark_pixels) < min_ink_pixels

    @classmethod
    def detect_form_splits(
        cls,
        img: np.ndarray,
        min_form_height: int = 800,
        whitespace_threshold: float = 253.0,
        min_divider_height: int = 15,
        min_partition_ink: int = 1000,
    ) -> Tuple[List[Tuple[int, int]], Optional[int]]:
        """
        Analyzes a page image to detect whether multiple feedback forms are stacked vertically.
        
        Requires evidence:
        1. Aspect ratio and height sufficient for 2 forms (H >= 2000 px).
        2. Clean horizontal whitespace divider band in middle zone (35% to 65% of H).
        3. Sufficient non-white content in both upper and lower partitions (not a blank margin).
        
        Returns:
            Tuple of (list_of_y_spans [(y1, y2), ...], split_y_or_None).
        """
        if img is None or img.size == 0:
            return [(0, 0)], None

        h, w = img.shape[:2]

        # Single form pages (e.g., standard single-form portrait A4)
        if h < 2000 or cls.is_blank_image(img):
            return [(0, h)], None

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Search middle band 35% .. 65% for contiguous whitespace divider band
        y_min = int(0.35 * h)
        y_max = int(0.65 * h)
        search_zone = gray[y_min:y_max, :]
        row_means = np.mean(search_zone, axis=1)

        is_white = (row_means >= whitespace_threshold).astype(np.uint8)
        runs = []
        start = None
        for i, v in enumerate(is_white):
            if v == 1 and start is None:
                start = i
            elif v == 0 and start is not None:
                runs.append((start + y_min, i + y_min, i - start))
                start = None
        if start is not None:
            runs.append((start + y_min, len(is_white) + y_min, len(is_white) - start))

        # Filter runs with sufficient height (>= min_divider_height)
        valid_runs = [r for r in runs if r[2] >= min_divider_height]
        if valid_runs:
            # The true divider between stacked forms is the most substantial whitespace band (largest height).
            # If multiple wide bands exist (within 70% of max height), break ties by distance to 50% height.
            max_height = max(r[2] for r in valid_runs)
            substantial_runs = [r for r in valid_runs if r[2] >= max(min_divider_height, int(0.7 * max_height))]
            mid_y = h // 2
            best_run = min(substantial_runs, key=lambda r: abs((r[0] + r[1]) // 2 - mid_y))
            split_y = (best_run[0] + best_run[1]) // 2

            top_h = split_y
            bottom_h = h - split_y

            # Validate both partitions meet minimum height
            if top_h >= min_form_height and bottom_h >= min_form_height:
                top_ink = np.sum(gray[:split_y, :] < 240)
                bottom_ink = np.sum(gray[split_y:, :] < 240)

                if top_ink > min_partition_ink and bottom_ink > min_partition_ink:
                    logger.info(
                        "Detected stacked dual-form page split at y=%d (divider run: %d..%d, length=%d px, top_ink=%d, bottom_ink=%d)",
                        split_y,
                        best_run[0],
                        best_run[1],
                        best_run[2],
                        top_ink,
                        bottom_ink,
                    )
                    return [(0, split_y), (split_y, h)], split_y

        # Insufficient evidence for splitting; preserve single form
        return [(0, h)], None

    @classmethod
    def process_rendered_page(
        cls,
        page_img: np.ndarray,
        page_num: int,
        output_dir: str,
        base_prefix: str = "page",
    ) -> List[Dict[str, Any]]:
        """
        Processes a single rendered page: deskews, checks for stacked forms, and saves output.
        
        Returns:
            List of form descriptor dicts:
            [{
                "page_number": int,
                "form_index": int,
                "image_path": str,
                "is_split": bool,
                "split_y": Optional[int],
                "skew_deg": float,
                "is_blank": bool,
                "dimensions": (w, h)
            }, ...]
        """
        os.makedirs(output_dir, exist_ok=True)
        deskewed, skew_deg = cls.deskew_image(page_img)
        is_blank = cls.is_blank_image(deskewed)

        spans, split_y = cls.detect_form_splits(deskewed)
        is_split = len(spans) > 1

        results = []
        for f_idx, (y1, y2) in enumerate(spans, start=1):
            form_slice = deskewed[y1:y2, :]
            out_filename = f"{base_prefix}_{page_num}_form_{f_idx}.png" if is_split else f"{base_prefix}_{page_num}.png"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, form_slice)

            results.append({
                "page_number": page_num,
                "form_index": f_idx,
                "image_path": out_path,
                "is_split": is_split,
                "split_y": split_y,
                "skew_deg": round(skew_deg, 3),
                "is_blank": is_blank,
                "dimensions": (form_slice.shape[1], form_slice.shape[0]),
            })

        return results

    @classmethod
    def process_pdf(
        cls,
        pdf_path: str,
        output_dir: str,
        dpi: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Renders, deskews, and separates all forms from an uploaded PDF document.
        
        Parameters:
            pdf_path: Absolute path to the PDF file.
            output_dir: Target directory to save rendered form PNG images.
            dpi: Rendering resolution (default 300 DPI).
            
        Returns:
            Ordered list of form records corresponding to every extracted form in document.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        os.makedirs(output_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF '{pdf_path}': {e}") from e

        all_forms: List[Dict[str, Any]] = []
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        page_count = len(doc)
        try:
            for page_idx in range(page_count):
                page_num = page_idx + 1
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=matrix, alpha=False)

                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 4:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                page_forms = cls.process_rendered_page(
                    page_img=img_bgr,
                    page_num=page_num,
                    output_dir=output_dir,
                )
                all_forms.extend(page_forms)
        finally:
            doc.close()

        logger.info(
            "Processed PDF '%s' (%d pages) -> %d individual form images generated",
            os.path.basename(pdf_path),
            page_count,
            len(all_forms),
        )
        return all_forms
