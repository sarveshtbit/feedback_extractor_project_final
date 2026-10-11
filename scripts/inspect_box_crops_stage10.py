import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.modules.setdefault('sklearn', None)

import cv2
import yaml
import json
import numpy as np

# Load template config
with open("config/form_template.yaml", "r") as f:
    orig_cfg = yaml.safe_load(f)

# Standalone Aligner
class Aligner:
    def __init__(self):
        self.canvas_w = 2479
        self.canvas_h = 3508
        self.ref = cv2.imread("bannari_amman_form_template.png")

    def align(self, img):
        if self.ref is not None:
            try:
                orb = cv2.ORB_create(3000)
                kp1, des1 = orb.detectAndCompute(img, None)
                kp2, des2 = orb.detectAndCompute(self.ref, None)
                if des1 is not None and des2 is not None and len(des1) >= 10 and len(des2) >= 10:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
                    if len(matches) >= 15:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if H is not None and H.shape == (3, 3):
                            return cv2.warpPerspective(img, H, (self.canvas_w, self.canvas_h)), True
            except Exception:
                pass
        scale = min(self.canvas_w / float(img.shape[1]), self.canvas_h / float(img.shape[0]))
        nw, nh = int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 255, dtype=np.uint8)
        canvas[(self.canvas_h - nh)//2:(self.canvas_h - nh)//2 + nh, (self.canvas_w - nw)//2:(self.canvas_w - nw)//2 + nw] = resized
        return canvas, False

aligner = Aligner()
os.makedirs("storage/stage10_digit_evaluation/diagnostic_comparisons", exist_ok=True)

# 1. Ground truth mapping from cnn_dataset/ground_truth.csv
gt_mobile = {
    "form1.jpeg": "9432164321",
    "form2.jpeg": "9042642526",
    "form3.jpeg": "8088997798",
    "form4.jpeg": "9754608192",
    "form5.jpeg": "8304934091",
}

def remove_borders_and_isolate(box_crop):
    """Preprocessing used in Stage 9."""
    if box_crop is None or box_crop.size == 0:
        return np.full((64, 64, 3), 255, dtype=np.uint8), False, 0
    bh, bw = box_crop.shape[:2]
    pad_y = int(bh * 0.12)
    pad_x = int(bw * 0.12)
    inner = box_crop[pad_y:bh-pad_y, pad_x:bw-pad_x]
    if inner.size == 0:
        return np.full((64, 64, 3), 255, dtype=np.uint8), False, 0
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    ink_count = cv2.countNonZero(thresh)
    if ink_count < 25:
        return np.full((64, 64, 3), 255, dtype=np.uint8), False, ink_count
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 15]
    if not valid_contours:
        return np.full((64, 64, 3), 255, dtype=np.uint8), False, ink_count
    all_pts = np.vstack([c for c in valid_contours])
    x, y, w, h = cv2.boundingRect(all_pts)
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(inner.shape[1], x + w + pad)
    y2 = min(inner.shape[0], y + h + pad)
    digit_crop = inner[y1:y2, x1:x2]
    target_h, target_w = 64, 64
    dh, dw = digit_crop.shape[:2]
    scale = min((target_w - 16) / float(dw), (target_h - 16) / float(dh))
    new_w = max(1, int(round(dw * scale)))
    new_h = max(1, int(round(dh * scale)))
    resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    canvas[(target_h - new_h)//2:(target_h - new_h)//2 + new_h, (target_w - new_w)//2:(target_w - new_w)//2 + new_w] = resized
    return canvas, True, ink_count

# Check template boxes
boxes_cfg = orig_cfg['fields']['student_mobile']['boxes']
print(f"Total boxes defined for student_mobile: {len(boxes_cfg)}")

report_data = []

for fname, gt in gt_mobile.items():
    fpath = os.path.join("cnn_dataset", fname)
    if not os.path.exists(fpath):
        continue
    raw = cv2.imread(fpath)
    aligned, _ = aligner.align(raw)
    h, w = aligned.shape[:2]
    
    # Create comparison montage: 10 boxes, each with:
    # Row 1: Original box crop (resized to 64x64)
    # Row 2: Preprocessed / isolated digit crop (64x64)
    orig_crops = []
    clean_crops = []
    box_statuses = []
    
    for idx, b in enumerate(boxes_cfg):
        b_norm = b['normalized_bbox']
        bx1, by1, bx2, by2 = int(b_norm[0]*w), int(b_norm[1]*h), int(b_norm[2]*w), int(b_norm[3]*h)
        raw_box = aligned[by1:by2, bx1:bx2]
        
        clean_box, has_digit, ink_count = remove_borders_and_isolate(raw_box)
        
        # Resize raw box to 64x64 for visual comparison
        raw_box_resized = cv2.resize(raw_box, (64, 64)) if raw_box.size > 0 else np.full((64, 64, 3), 255, dtype=np.uint8)
        
        # Add labels
        expected_char = gt[idx] if idx < len(gt) else "?"
        box_statuses.append({
            "box_index": idx,
            "expected_digit": expected_char,
            "has_digit_detected": has_digit,
            "ink_pixels": ink_count,
            "raw_bbox_px": [bx1, by1, bx2, by2]
        })
        
        # Visual annotation
        labeled_raw = raw_box_resized.copy()
        cv2.putText(labeled_raw, f"#{idx}:{expected_char}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        labeled_clean = clean_box.copy()
        stat_color = (0, 180, 0) if has_digit else (128, 128, 128)
        stat_text = "INK" if has_digit else "BLANK"
        cv2.putText(labeled_clean, stat_text, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, stat_color, 1)
        
        orig_crops.append(labeled_raw)
        clean_crops.append(labeled_clean)
        
    # Build montage
    row_orig = np.hstack(orig_crops)
    row_clean = np.hstack(clean_crops)
    # Add header label rows
    separator = np.full((8, row_orig.shape[1], 3), 200, dtype=np.uint8)
    montage = np.vstack([row_orig, separator, row_clean])
    
    # Save diagnostic comparison image
    out_img_path = f"storage/stage10_digit_evaluation/diagnostic_comparisons/{fname.replace('.jpeg','')}_diagnostic_comparison.png"
    cv2.imwrite(out_img_path, montage)
    
    report_data.append({
        "file": fname,
        "ground_truth_masked": gt[:2] + "***" + gt[-2:],
        "boxes": box_statuses,
        "diagnostic_image": out_img_path
    })
    print(f"Processed {fname}: saved comparison to {out_img_path}")

with open("storage/stage10_digit_evaluation/diagnostic_comparisons/audit_summary.json", "w") as f:
    json.dump(report_data, f, indent=2)

print("\nTask 1 Inspection Complete. Audit summary written to storage/stage10_digit_evaluation/diagnostic_comparisons/audit_summary.json")
