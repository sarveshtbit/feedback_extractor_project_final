import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class DigitSegmenter:
    """
    Computer Vision Digit Segmentation and 15-Step Preprocessing Engine.
    Strictly NO OCR.
    """
    @staticmethod
    def preprocess_digit_crop(crop: np.ndarray) -> np.ndarray:
        """
        Executes exact 15-step digit preprocessing pipeline:
        Returns normalized (28, 28) float32 numpy array with values in [0.0, 1.0].
        """
        if crop is None or crop.size == 0:
            return np.zeros((28, 28), dtype=np.float32)

        # 1. Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return np.zeros((28, 28), dtype=np.float32)

        # 2. Apply light Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. Estimate background using the border pixels
        border_pixels = np.concatenate([
            blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]
        ])
        bg_mean = float(np.mean(border_pixels))

        # 4. Automatically determine whether inversion is required (foreground digit white on black canvas)
        if bg_mean > 127:
            # Light background -> invert so digit ink becomes white foreground (255)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Dark background -> keep as is / Otsu standard
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Extract foreground & 6. Detect digit bounding box
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            # 7. Add small margin
            pad = 4
            h, w = thresh.shape
            x1, y1 = max(0, bx - pad), max(0, by - pad)
            x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
            # 8. Crop digit
            digit_fg = thresh[y1:y2, x1:x2]
        else:
            digit_fg = thresh

        dh, dw = digit_fg.shape
        if dh == 0 or dw == 0:
            return np.zeros((28, 28), dtype=np.float32)

        # 9. Preserve aspect ratio & 10. Resize to approx 20x20 inside 20x20 box
        max_dim = max(dh, dw)
        scale = 20.0 / max_dim
        new_w = max(1, int(round(dw * scale)))
        new_h = max(1, int(round(dh * scale)))

        resized_digit = cv2.resize(digit_fg, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 11. Place on a 28x28 black canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_digit

        # 12. Center using intensity-weighted center of mass
        M = cv2.moments(canvas)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            shift_x = int(round(13.5 - cx))
            shift_y = int(round(13.5 - cy))

            # Apply translation shift within bounds
            M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, M_trans, (28, 28), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 13. Normalize to [0, 1]
        normalized = canvas.astype(np.float32) / 255.0

        return normalized

    def segment_digits(
        self,
        field_crop: np.ndarray,
        is_character_boxes: bool = False,
        box_crops: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray], np.ndarray]:
        """
        Segments individual digit regions inside a numeric field crop.
        
        Returns:
        - digit_crops: List of raw digit image crops
        - digit_bboxes: List of [x, y, w, h] bounding boxes sorted left-to-right
        - preprocessed_canvases: List of (28, 28) float32 2D arrays
        - batch_tensor: (N, 28, 28, 1) float32 numpy array ready for CNN batch prediction
        """
        digit_crops = []
        digit_bboxes = []

        if is_character_boxes and box_crops:
            # Field already has pre-divided box crops (e.g. mobile number 10 boxes)
            for idx, b_crop in enumerate(box_crops):
                if b_crop is not None and b_crop.size > 0:
                    digit_crops.append(b_crop)
                    digit_bboxes.append([idx * 30, 0, b_crop.shape[1], b_crop.shape[0]])

        else:
            if field_crop is None or field_crop.size == 0:
                return [], [], [], np.zeros((0, 28, 28, 1), dtype=np.float32)

            # Perform computer vision contour/projection digit segmentation on field crop
            if len(field_crop.shape) == 3:
                gray = cv2.cvtColor(field_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = field_crop.copy()

            fh, fw = gray.shape[:2]
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Auto thresholding
            p2, p98 = np.percentile(blurred, (2, 98))
            if p98 - p2 > 10:
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

            # Morphological close to connect broken digit strokes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            raw_boxes = []
            min_h = max(6, int(fh * 0.15))
            min_w = 3
            max_w = int(fw * 0.85)
            max_h = int(fh * 0.95)
            for ctn in contours:
                bx, by, bw, bh = cv2.boundingRect(ctn)
                # Filter out full outer border box or tiny noise
                if bw > max_w and bh > max_h:
                    continue
                if bh >= min_h and bw >= min_w and bw <= max_w:
                    area = bw * bh
                    if area >= 18:
                        raw_boxes.append((bx, by, bw, bh))

            # Merge overlapping or vertically stacked bounding boxes
            raw_boxes = sorted(raw_boxes, key=lambda b: b[0])
            merged_boxes = []
            for b in raw_boxes:
                if not merged_boxes:
                    merged_boxes.append(list(b))
                else:
                    prev = merged_boxes[-1]
                    # If bounding boxes overlap horizontally significantly or are adjacent
                    if b[0] < prev[0] + prev[2] - 2 and abs(b[0] - prev[0]) < 15:
                        # Combine bounding boxes
                        x_min = min(prev[0], b[0])
                        y_min = min(prev[1], b[1])
                        x_max = max(prev[0] + prev[2], b[0] + b[2])
                        y_max = max(prev[1] + prev[3], b[1] + b[3])
                        merged_boxes[-1] = [x_min, y_min, x_max - x_min, y_max - y_min]
                    else:
                        merged_boxes.append(list(b))

            # SORT STRICTLY LEFT TO RIGHT BY X-COORDINATE
            digit_bboxes = sorted(merged_boxes, key=lambda b: b[0])

            # Crop individual digits
            for bx, by, bw, bh in digit_bboxes:
                # Add small padding margin
                pad = 2
                x1, y1 = max(0, bx - pad), max(0, by - pad)
                x2, y2 = min(fw, bx + bw + pad), min(fh, by + bh + pad)
                d_crop = field_crop[y1:y2, x1:x2]
                digit_crops.append(d_crop)

        # Apply 15-step preprocessing to every segmented digit crop
        preprocessed_canvases = []
        for d_crop in digit_crops:
            canvas_28 = self.preprocess_digit_crop(d_crop)
            preprocessed_canvases.append(canvas_28)

        if preprocessed_canvases:
            # Construct (N, 28, 28, 1) batch array
            batch_np = np.array(preprocessed_canvases, dtype=np.float32)
            batch_tensor = np.expand_dims(batch_np, axis=-1)
        else:
            batch_tensor = np.zeros((0, 28, 28, 1), dtype=np.float32)

        return digit_crops, digit_bboxes, preprocessed_canvases, batch_tensor
