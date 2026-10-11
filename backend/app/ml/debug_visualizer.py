import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

class DebugVisualizer:
    """
    Pipeline Debug Visualizer.
    Generates step-by-step visual debug artifacts saved to scratch/<job_id>/debug/
    showing:
    - Original document with YOLO / Template field bounding box
    - Cropped numeric field image
    - Segmented digit bounding boxes
    - 28x28 normalized preprocessed canvases
    - Per-digit CNN probability text
    - Final reconstructed string
    """
    def __init__(self, scratch_dir: str = "scratch"):
        self.scratch_dir = scratch_dir

    def visualize_numeric_extraction(
        self,
        job_id: str,
        field_name: str,
        original_image: np.ndarray,
        field_bbox: Optional[List[int]],
        field_crop: Optional[np.ndarray],
        digit_bboxes: List[List[int]],
        preprocessed_canvases: List[np.ndarray],
        cnn_results: Dict[str, Any]
    ) -> str:
        debug_dir = os.path.join(self.scratch_dir, str(job_id), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # 1. Annotate original image with field bbox
        annotated_doc = original_image.copy() if original_image is not None else np.zeros((300, 300, 3), dtype=np.uint8)
        if field_bbox and len(field_bbox) == 4 and annotated_doc is not None:
            x, y, w, h = field_bbox
            cv2.rectangle(annotated_doc, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_doc, f"Field: {field_name}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Annotate field crop with segmented digit bboxes
        annotated_crop = field_crop.copy() if field_crop is not None else np.zeros((100, 200, 3), dtype=np.uint8)
        if len(annotated_crop.shape) == 2:
            annotated_crop = cv2.cvtColor(annotated_crop, cv2.COLOR_GRAY2BGR)

        per_digit = cnn_results.get("per_digit_results", [])
        for idx, bbox in enumerate(digit_bboxes):
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
                cv2.rectangle(annotated_crop, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
                if idx < len(per_digit):
                    digit_char = per_digit[idx].get("digit", "?")
                    cv2.putText(annotated_crop, str(digit_char), (bx, max(10, by - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 3. Create canvas strip of 28x28 normalized images
        if preprocessed_canvases:
            canvas_imgs = []
            for c in preprocessed_canvases:
                c_u8 = (c * 255.0).astype(np.uint8)
                c_bgr = cv2.cvtColor(c_u8, cv2.COLOR_GRAY2BGR)
                # Border frame
                c_framed = cv2.copyMakeBorder(c_bgr, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[128, 128, 128])
                canvas_imgs.append(c_framed)
            strip = np.hstack(canvas_imgs)
        else:
            strip = np.zeros((32, 32, 3), dtype=np.uint8)

        # 4. Construct composite debug dashboard image
        recon_val = cnn_results.get("value", "N/A")
        conf_val = cnn_results.get("confidence", 0.0)

        summary_text = f"Field: {field_name} | Reconstructed: {recon_val} | Conf: {conf_val:.4f} | Bboxes: {len(digit_bboxes)}"
        banner = np.zeros((40, max(600, annotated_crop.shape[1] + strip.shape[1] + 20), 3), dtype=np.uint8)
        cv2.putText(banner, summary_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        output_path = os.path.join(debug_dir, f"{field_name}_debug.png")
        cv2.imwrite(output_path, annotated_crop)

        doc_path = os.path.join(debug_dir, f"{field_name}_doc_bbox.png")
        cv2.imwrite(doc_path, annotated_doc)

        strip_path = os.path.join(debug_dir, f"{field_name}_digit_canvases.png")
        cv2.imwrite(strip_path, strip)

        return debug_dir
