import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple

from backend.app.services.preprocessing import ImagePreprocessor
from backend.app.ml.inference.cnn_service import CNNInferenceService
from backend.app.core.config import get_form_template_config

def run_visual_debug_pipeline():
    print("==================================================")
    print("VISUAL DEBUGGING PIPELINE FOR ALL 5 REAL FORMS")
    print("==================================================")

    input_dir = "cnn_dataset"
    output_debug_dir = "real_digit_dataset/visual_debug"
    os.makedirs(output_debug_dir, exist_ok=True)

    gt_csv = os.path.join(input_dir, "ground_truth.csv")
    gt_excel = os.path.join(input_dir, "ground_truth.xlsx")
    if os.path.exists(gt_csv):
        df_gt = pd.read_csv(gt_csv)
    elif os.path.exists(gt_excel):
        df_gt = pd.read_excel(gt_excel)
    else:
        raise FileNotFoundError("Ground truth file missing!")

    cfg = get_form_template_config()
    student_mobile_cfg = cfg["fields"]["student_mobile"]
    boxes_cfg = student_mobile_cfg["boxes"]
    roi_bbox_norm = student_mobile_cfg["normalized_bbox"]

    cnn_service = CNNInferenceService(model_path="ml/models/pretrained_resnet18.pth")

    print(f"Loaded Template Config: student_mobile normalized_bbox = {roi_bbox_norm}")
    print(f"CNN Model Ready: {cnn_service.is_ready()}")
    print("=" * 75)

    summary_table_rows = []

    for idx, row in df_gt.iterrows():
        fname = str(row["filename"]).strip()
        gt_mobile = str(row["student_mobile"]).strip()
        form_path = os.path.join(input_dir, fname)
        form_id = os.path.splitext(fname)[0]

        print(f"\n" + "="*60)
        print(f"PROCESSING FORM: {fname} (Form ID: {form_id})")
        print(f"GROUND TRUTH MOBILE: {gt_mobile}")
        print("="*60)

        with open(form_path, "rb") as f:
            file_bytes = f.read()

        # 1. Original Image
        raw_img = ImagePreprocessor.load_and_fix_exif(file_bytes)
        h_orig, w_orig = raw_img.shape[:2]

        # Save Original Image
        cv2.imwrite(os.path.join(output_debug_dir, f"{form_id}_01_original.png"), raw_img)

        # 2. Canonical Image (Normalized 1280x904 Baseline Space)
        canonical_img = raw_img.copy()
        cv2.imwrite(os.path.join(output_debug_dir, f"{form_id}_02_aligned.png"), canonical_img)

        # 3. Detected Student Mobile Field Bounding Box (Large Rectangle)
        field_roi_img = raw_img.copy()
        rx1 = int(roi_bbox_norm[0] * w_orig)
        ry1 = int(roi_bbox_norm[1] * h_orig)
        rx2 = int(roi_bbox_norm[2] * w_orig)
        ry2 = int(roi_bbox_norm[3] * h_orig)

        cv2.rectangle(field_roi_img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
        cv2.putText(field_roi_img, "STUDENT MOBILE NUMBER FIELD", (rx1, max(30, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save Field ROI Overlay
        cv2.imwrite(os.path.join(output_debug_dir, f"{form_id}_03_field_roi.png"), field_roi_img)

        # Extract Field ROI image
        field_crop = raw_img[ry1:ry2, rx1:rx2].copy()
        fc_h, fc_w = field_crop.shape[:2]

        # 4. Student Mobile Field with All 10 Detected Digit Boxes Drawn
        field_boxes_draw = raw_img.copy()
        roi_boxes_draw = field_crop.copy()

        box_crops = []
        box_meta = []

        for b_idx, box_item in enumerate(boxes_cfg):
            c_bbox = box_item["normalized_bbox"]
            bx1 = int(c_bbox[0] * w_orig)
            by1 = int(c_bbox[1] * h_orig)
            bx2 = int(c_bbox[2] * w_orig)
            by2 = int(c_bbox[3] * h_orig)

            # Draw on full raw image
            cv2.rectangle(field_boxes_draw, (bx1, by1), (bx2, by2), (0, 200, 0), 2)
            cv2.putText(field_boxes_draw, f"P{b_idx+1}", (bx1, max(15, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Relative coords inside field_crop
            rel_bx1, rel_by1 = max(0, bx1 - rx1), max(0, by1 - ry1)
            rel_bx2, rel_by2 = min(fc_w, bx2 - rx1), min(fc_h, by2 - ry1)
            
            cv2.rectangle(roi_boxes_draw, (rel_bx1, rel_by1), (rel_bx2, rel_by2), (0, 200, 0), 2)
            cv2.putText(roi_boxes_draw, f"P{b_idx+1}", (rel_bx1, max(12, rel_by1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

            # Crop individual digit box from raw image
            digit_crop = raw_img[by1:by2, bx1:bx2]
            box_crops.append({"index": b_idx + 1, "crop": digit_crop})
            box_meta.append({
                "index": b_idx + 1,
                "bbox": (bx1, by1, bx2, by2),
                "width": bx2 - bx1,
                "height": by2 - by1,
                "rel_x": bx1 - rx1
            })

        # Save 10 digit boxes image
        cv2.imwrite(os.path.join(output_debug_dir, f"{form_id}_04_field_10_boxes.png"), field_boxes_draw)
        cv2.imwrite(os.path.join(output_debug_dir, f"{form_id}_04_field_roi_10_boxes.png"), roi_boxes_draw)

        # Run CNN inference on 10 crops
        cnn_res = cnn_service.predict_phone_number(box_crops)
        pred_val = cnn_res.get("value") or "0000000000"
        per_char = cnn_res.get("per_character_results", [])

        # 5. Contact Sheet of 10 Individual Digit Crops
        cell_w, cell_h = 140, 160
        contact_sheet = np.full((2 * cell_h, 5 * cell_w, 3), 255, dtype=np.uint8)

        print(f"\n--- DIGIT CROP INSPECTION ({fname}) ---")
        correct_count = 0

        for b_idx in range(10):
            gt_d = gt_mobile[b_idx]
            pred_d = per_char[b_idx]["digit"] if b_idx < len(per_char) else "?"
            conf = per_char[b_idx]["confidence"] if b_idx < len(per_char) else 0.0
            
            is_correct = (gt_d == pred_d)
            if is_correct: correct_count += 1
            mark = "MATCH" if is_correct else "MISMATCH"

            print(f"Digit {b_idx} (P{b_idx+1}): Actual={gt_d} | CNN={pred_d} | Conf={conf:.4f} | Status={mark}")

            # Draw cell on contact sheet
            r_idx = b_idx // 5
            c_idx = b_idx % 5
            x0 = c_idx * cell_w + 10
            y0 = r_idx * cell_h + 10

            crop_raw = box_crops[b_idx]["crop"]
            if crop_raw is not None and crop_raw.size > 0:
                resized_crop = cv2.resize(crop_raw, (80, 80))
                contact_sheet[y0 + 40:y0 + 120, x0 + 10:x0 + 90] = resized_crop

            # Header & label text
            head_txt = f"P{b_idx+1} GT:{gt_d}"
            pred_txt = f"CNN:{pred_d} ({conf:.2f})"
            color = (0, 150, 0) if is_correct else (0, 0, 200)

            cv2.putText(contact_sheet, head_txt, (x0 + 5, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.putText(contact_sheet, pred_txt, (x0 + 5, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
            cv2.rectangle(contact_sheet, (x0, y0), (x0 + cell_w - 10, y0 + cell_h - 10), color, 2)

        # Save Contact Sheet Image
        cv2.imwrite(os.path.join(output_debug_dir, f"{form_id}_05_contact_sheet.png"), contact_sheet)

        # GEOMETRY CHECKS EVALUATION
        widths = [m["width"] for m in box_meta]
        heights = [m["height"] for m in box_meta]
        xs = [m["bbox"][0] for m in box_meta]
        
        is_left_to_right = all(xs[i] < xs[i+1] for i in range(len(xs)-1))
        field_is_mobile = (ry1 >= 280 and ry2 <= 380)  # Y: 305..350 on 904 height image
        boxes_10_count = (len(box_crops) == 10)
        crops_valid = all(c["crop"] is not None and c["crop"].shape[0] >= 10 and c["crop"].shape[1] >= 10 for c in box_crops)
        exact_match = (gt_mobile == pred_val)

        print(f"\n--- GEOMETRY CHECK SUMMARY ({fname}) ---")
        print(f"  Field localized is Student Mobile: {field_is_mobile} (Y bounds: {ry1}..{ry2})")
        print(f"  Detected 10 digit boxes: {boxes_10_count}")
        print(f"  Strict left-to-right ordering: {is_left_to_right}")
        print(f"  Box width range: {min(widths)}px .. {max(widths)}px | Height range: {min(heights)}px .. {max(heights)}px")
        print(f"  Crops valid (non-empty & legible): {crops_valid}")
        print(f"  CNN Correct Digits: {correct_count} / 10")
        print(f"  CNN 10-Digit Exact Match: {exact_match}")

        summary_table_rows.append({
            "Form": fname,
            "Field correct?": "YES" if field_is_mobile else "NO",
            "10 boxes correct?": "YES" if (boxes_10_count and is_left_to_right) else "NO",
            "Crops correct?": "YES" if crops_valid else "NO",
            "CNN correct digits": f"{correct_count}/10",
            "CNN mobile exact match": "YES" if exact_match else "NO"
        })

    df_summary = pd.DataFrame(summary_table_rows)
    print("\n" + "="*75)
    print("FINAL DIAGNOSTIC SUMMARY TABLE")
    print("="*75)
    print(df_summary.to_string(index=False))
    print("="*75)

if __name__ == "__main__":
    run_visual_debug_pipeline()
