import os
import shutil
import cv2
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
from backend.app.services.alignment import FormAligner
from backend.app.services.preprocessing import ImagePreprocessor

def prepare_real_digit_dataset():
    print("==================================================")
    print("STEP 1 — READ & VALIDATE GROUND TRUTH")
    print("==================================================")
    
    input_dir = "cnn_dataset"
    output_dir = "real_digit_dataset"
    debug_dir = os.path.join(output_dir, "debug")
    
    # 1. Locate Ground Truth File (.csv or .xlsx)
    gt_csv = os.path.join(input_dir, "ground_truth.csv")
    gt_xlsx = os.path.join(input_dir, "ground_truth.xlsx")
    
    if os.path.exists(gt_csv):
        gt_path = gt_csv
        df_gt = pd.read_csv(gt_path)
    elif os.path.exists(gt_xlsx):
        gt_path = gt_xlsx
        df_gt = pd.read_excel(gt_path)
    else:
        raise FileNotFoundError("Neither ground_truth.csv nor ground_truth.xlsx found in cnn_dataset/")

    print(f"Ground truth file loaded: {gt_path}")
    
    # Validate columns
    required_cols = {"filename", "student_mobile"}
    if not required_cols.issubset(set(df_gt.columns)):
        raise ValueError(f"Ground truth file missing required columns. Expected: {required_cols}, Found: {list(df_gt.columns)}")

    # Filesystem check
    forms_in_dir = set([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gt_filenames = set(df_gt["filename"].astype(str).str.strip())
    
    missing_images = gt_filenames - forms_in_dir
    unlabeled_images = forms_in_dir - gt_filenames
    
    invalid_mobiles = []
    for idx, row in df_gt.iterrows():
        fname = str(row["filename"]).strip()
        mobile = str(row["student_mobile"]).strip()
        if len(mobile) != 10 or not mobile.isdigit():
            invalid_mobiles.append((fname, mobile))
            
    print(f"Forms found in cnn_dataset/: {len(forms_in_dir)}")
    print(f"Ground-truth rows: {len(df_gt)}")
    print(f"Matched forms: {len(gt_filenames - missing_images)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Unlabeled images: {len(unlabeled_images)}")
    print(f"Invalid mobile numbers: {len(invalid_mobiles)}")
    
    if missing_images or invalid_mobiles:
        print("Validation errors detected! Stopping execution.")
        if missing_images:
            print("Missing images:", missing_images)
        if invalid_mobiles:
            print("Invalid mobile numbers:", invalid_mobiles)
        return

    # Clean output directory real_digit_dataset/ before regenerating
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 2. Load Form Template Config & Form Aligner
    print("\n==================================================")
    print("STEP 2 — FORM ALIGNMENT & TEMPLATE COORDINATES")
    print("==================================================")
    
    template_path = "config/form_template.yaml"
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template config missing at {template_path}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    student_mobile_cfg = cfg["fields"]["student_mobile"]
    boxes_cfg = student_mobile_cfg["boxes"]
    roi_bbox_norm = student_mobile_cfg["normalized_bbox"]
    
    aligner = FormAligner(template_config=cfg)
    
    print(f"Loaded Student Mobile field template: {len(boxes_cfg)} character boxes configured.")
    print(f"Student Mobile ROI normalized bbox: {roi_bbox_norm}")

    # Prepare digit output directories (0 to 9)
    os.makedirs(debug_dir, exist_ok=True)
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    metadata_rows = []
    alignment_reports = []
    total_crops_expected = len(df_gt) * 10
    total_crops_generated = 0
    missing_crops = 0

    print("\n==================================================")
    print("STEP 3 & 4 — MULTI-FORM ALIGNMENT & CROP EXTRACTION")
    print("==================================================")

    for idx, row in df_gt.iterrows():
        fname = str(row["filename"]).strip()
        gt_mobile = str(row["student_mobile"]).strip()
        form_path = os.path.join(input_dir, fname)
        form_id = os.path.splitext(fname)[0]
        
        with open(form_path, "rb") as f:
            file_bytes = f.read()

        # A. EXIF fix & Raw BGR Image
        raw_img = ImagePreprocessor.load_and_fix_exif(file_bytes)
        h_orig, w_orig = raw_img.shape[:2]

        # B. Deskew angle check
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        deskewed_gray, skew_angle = ImagePreprocessor.deskew_image(gray)

        # C. Form Aligner (Homography marker warping or letterboxed geometry fallback)
        aligned_img, H, is_warped = aligner.align(raw_img)
        h_aligned, w_aligned = aligned_img.shape[:2]

        alignment_status = "ALIGNMENT_OK" if (aligned_img is not None and h_aligned > 0) else "ALIGNMENT_FAILED"

        # D. Save Alignment Debug Image (original vs aligned side-by-side or marked)
        align_debug_canvas = np.full((max(h_orig, 600), w_orig + 10, 3), 255, dtype=np.uint8)
        align_debug_canvas[:h_orig, :w_orig] = raw_img
        
        info_str = f"Form: {fname} | Orig: {w_orig}x{h_orig} | Angle: {skew_angle:.1f}deg | Warped: {is_warped} | Status: {alignment_status}"
        cv2.putText(align_debug_canvas, info_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 150, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{form_id}_alignment_debug.png"), align_debug_canvas)

        # E. Calculate Pixel ROI on current form image
        rx1 = int(roi_bbox_norm[0] * w_orig)
        ry1 = int(roi_bbox_norm[1] * h_orig)
        rx2 = int(roi_bbox_norm[2] * w_orig)
        ry2 = int(roi_bbox_norm[3] * h_orig)

        # Save large single-rectangle test image formX_mobile_roi_test.png
        test_roi_img = raw_img.copy()
        cv2.rectangle(test_roi_img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
        cv2.putText(test_roi_img, f"STUDENT MOBILE FIELD ({fname})", (rx1, max(15, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{form_id}_mobile_roi_test.png"), test_roi_img)

        # F. Extract Student Mobile ROI & prepare 10-box debug visualizer
        roi = raw_img[ry1:ry2, rx1:rx2].copy()
        roi_h, roi_w = roi.shape[:2]

        debug_canvas = np.full((roi_h + 45, roi_w, 3), 255, dtype=np.uint8)
        debug_canvas[45:, :] = roi

        valid_boxes_count = 0

        for b_idx, b_item in enumerate(boxes_cfg):
            c_bbox = b_item["normalized_bbox"]
            cx1, cy1 = int(c_bbox[0] * w_orig), int(c_bbox[1] * h_orig)
            cx2, cy2 = int(c_bbox[2] * w_orig), int(c_bbox[3] * h_orig)

            # Crop digit box
            crop = raw_img[cy1:cy2, cx1:cx2]
            if crop is None or crop.size == 0:
                print(f"Warning: Empty crop at {fname} box {b_idx+1}")
                missing_crops += 1
                continue

            gt_digit = gt_mobile[b_idx]
            valid_boxes_count += 1
            
            # Save crop to class directory real_digit_dataset/<digit>/
            # Filename preserves writer_id and position
            crop_name = f"{form_id}_pos_{b_idx+1:02d}_digit_{gt_digit}.png"
            rel_crop_path = os.path.join(gt_digit, crop_name)
            abs_crop_path = os.path.join(output_dir, rel_crop_path)

            cv2.imwrite(abs_crop_path, crop)
            total_crops_generated += 1

            # Append metadata entry (writer_id preserves form group to prevent data leakage)
            metadata_rows.append({
                "image_path": rel_crop_path,
                "label": int(gt_digit),
                "writer_id": form_id,
                "source": "real_handwriting",
                "original_form": fname,
                "position": b_idx + 1
            })

            # Relative coordinates inside ROI for 10-box debug overlay
            bx1, by1 = cx1 - rx1, cy1 - ry1
            bx2, by2 = cx2 - rx1, cy2 - ry1
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(roi_w, bx2), min(roi_h, by2)

            # Draw box and ground-truth text P1:9 P2:4...
            cv2.rectangle(debug_canvas, (bx1, by1 + 45), (bx2, by2 + 45), (0, 200, 0), 2)
            label_str = f"P{b_idx+1}:{gt_digit}"
            cv2.putText(debug_canvas, label_str, (bx1 + 1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)

        # Save 10-box debug visualization for form
        cv2.imwrite(os.path.join(debug_dir, f"{form_id}_student_mobile_debug.png"), debug_canvas)

        alignment_reports.append({
            "form": fname,
            "original_dim": f"{w_orig}x{h_orig}",
            "aligned_dim": f"{w_aligned}x{h_aligned}",
            "skew_angle": f"{skew_angle:.2f} deg",
            "perspective_applied": is_warped,
            "roi_px": f"[{rx1}, {ry1}, {rx2}, {ry2}]",
            "valid_boxes": valid_boxes_count,
            "status": alignment_status
        })

    # Save metadata.csv
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"Saved metadata CSV to: {metadata_csv_path}")

    print("\n==================================================")
    print("STEP 5 — ALIGNMENT QUALITY CHECK REPORT")
    print("==================================================")
    align_df = pd.DataFrame(alignment_reports)
    print(align_df.to_string(index=False))

    print("\n==================================================")
    print("STEP 6 — DATASET VALIDATION SUMMARY REPORT")
    print("==================================================")
    print(f"Forms processed: {len(df_gt)}")
    print(f"Expected digit crops: {total_crops_expected}")
    print(f"Generated digit crops: {total_crops_generated}")
    print(f"Missing crops: {missing_crops}")
    print(f"Invalid labels: 0")
    
    print("\nClass Distribution (Real Digit Crops):")
    class_counts = metadata_df["label"].value_counts().sort_index()
    for d in range(10):
        cnt = class_counts.get(d, 0)
        print(f"  Digit {d}: {cnt} crops")

if __name__ == "__main__":
    prepare_real_digit_dataset()
