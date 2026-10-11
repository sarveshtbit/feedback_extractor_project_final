import os
import sys
import time
import json
import yaml

# Add torch lib to dll directory for Windows
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

sys.modules.setdefault('sklearn', None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

def mask_pii(text: str) -> str:
    if not text or len(text.strip()) == 0:
        return "***"
    t = text.strip()
    if len(t) <= 4:
        return "***"
    return t[:2] + "***" + t[-2:]

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    dist = levenshtein_distance(pred, gt)
    return dist / len(gt)

# Emptiness / Ink detector
def detect_ink_presence(crop: np.ndarray, min_ink_ratio: float = 0.008) -> tuple[bool, float]:
    """Detects whether crop contains handwritten ink or is effectively blank."""
    if crop is None or crop.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    # Invert binary threshold to get ink pixels
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    ink_pixels = cv2.countNonZero(opened)
    total_pixels = crop.shape[0] * crop.shape[1]
    ratio = float(ink_pixels) / float(total_pixels)
    return (ratio >= min_ink_ratio), ratio

def remove_box_borders_and_isolate_digit(box_crop: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Removes box border artifacts from a single digit box crop and centers the digit stroke.
    Returns (preprocessed_crop, has_digit).
    """
    if box_crop is None or box_crop.size == 0:
        return np.full((64, 64, 3), 255, dtype=np.uint8), False
        
    bh, bw = box_crop.shape[:2]
    # 1. Inset by 12% to strip outer border grid lines
    pad_y = int(bh * 0.12)
    pad_x = int(bw * 0.12)
    inner = box_crop[pad_y:bh-pad_y, pad_x:bw-pad_x]
    if inner.size == 0:
        return np.full((64, 64, 3), 255, dtype=np.uint8), False
        
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
    
    # 2. Adaptive thresholding to segment ink strokes
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    
    # 3. Check for ink presence
    ink_count = cv2.countNonZero(thresh)
    if ink_count < 25:  # Empty box threshold
        return np.full((64, 64, 3), 255, dtype=np.uint8), False
        
    # 4. Find contours of handwritten stroke
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 15]
    if not valid_contours:
        return np.full((64, 64, 3), 255, dtype=np.uint8), False
        
    # Get bounding box of combined digit contours
    all_pts = np.vstack([c for c in valid_contours])
    x, y, w, h = cv2.boundingRect(all_pts)
    
    # Crop the digit with slight padding
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(inner.shape[1], x + w + pad)
    y2 = min(inner.shape[0], y + h + pad)
    digit_crop = inner[y1:y2, x1:x2]
    
    # Center on a standard 64x64 white canvas
    target_h, target_w = 64, 64
    dh, dw = digit_crop.shape[:2]
    scale = min((target_w - 16) / float(dw), (target_h - 16) / float(dh))
    new_w = max(1, int(round(dw * scale)))
    new_h = max(1, int(round(dh * scale)))
    resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = resized
    
    return canvas, True

class StandaloneFormAligner:
    def __init__(self):
        self.canvas_width = 2479
        self.canvas_height = 3508
        self.ref_image = cv2.imread("bannari_amman_form_template.png") if os.path.exists("bannari_amman_form_template.png") else None
        self.dst_markers = np.array([[140, 140], [2339, 140], [2339, 3368], [140, 3368]], dtype=np.float32)

    def align(self, image):
        if self.ref_image is not None:
            try:
                orb = cv2.ORB_create(3000)
                kp1, des1 = orb.detectAndCompute(image, None)
                kp2, des2 = orb.detectAndCompute(self.ref_image, None)
                if des1 is not None and des2 is not None and len(des1) >= 10 and len(des2) >= 10:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
                    if len(matches) >= 15:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if H is not None and H.shape == (3, 3):
                            return cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height)), True
            except Exception:
                pass
        scale = min(self.canvas_width / float(image.shape[1]), self.canvas_height / float(image.shape[0]))
        nw, nh = int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))
        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 255, dtype=np.uint8)
        canvas[(self.canvas_height - nh)//2:(self.canvas_height - nh)//2 + nh, (self.canvas_width - nw)//2:(self.canvas_width - nw)//2 + nw] = resized
        return canvas, False

def main():
    print("=== STAGE 9: HANDWRITTEN FIELD & BOXED DIGIT OCR BENCHMARK ===")
    
    # 1. Load TrOCR
    print("Loading TrOCR model...")
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    trocr_proc = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # 2. Load PaddleOCR
    print("Loading PaddleOCR model...")
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    
    # 3. Load configurations
    with open("config/form_template.yaml", "r") as f:
        orig_cfg = yaml.safe_load(f)
    with open("config/form_template_experimental.yaml", "r") as f:
        refined_cfg = yaml.safe_load(f)
        
    aligner = StandaloneFormAligner()
    
    # Verified ground truth catalogue
    ground_truth_db = {
        "form1.jpeg": {
            "student_name": None,
            "school_name": None,
            "date": "10/05/2023",
            "student_mobile": "9432164321",
            "parent_mobile": None,
            "email": "test@gmail.com",
            "cutoff_marks": None
        },
        "form2.jpeg": {
            "student_name": "SARAVANAN",
            "school_name": "GOVT HSS ERODE",
            "date": "12/05/2023",
            "student_mobile": "9042642526",
            "parent_mobile": None,
            "email": "saravanan@gmail.com",
            "cutoff_marks": None
        },
        "form3.jpeg": {
            "student_name": "PRIYA",
            "school_name": "BHARATHI VIDHYALAYA",
            "date": "15/05/2023",
            "student_mobile": "8088997798",
            "parent_mobile": None,
            "email": "priya@gmail.com",
            "cutoff_marks": "188.50"
        },
        "form4.jpeg": {
            "student_name": None,
            "school_name": "ST JOSEPH HSS",
            "date": "18/05/2023",
            "student_mobile": "9754608192",
            "parent_mobile": None,
            "email": "joseph@gmail.com",
            "cutoff_marks": "192.00"
        },
        "form5.jpeg": {
            "student_name": "DINESH",
            "school_name": "MODEL SCHOOL",
            "date": "20/05/2023",
            "student_mobile": "8304934091",
            "parent_mobile": None,
            "email": "dinesh@gmail.com",
            "cutoff_marks": None
        }
    }
    
    os.makedirs("storage/stage9_investigation/boxed_digits", exist_ok=True)
    
    results = {}
    
    for fname, gt_dict in ground_truth_db.items():
        fpath = f"cnn_dataset/{fname}"
        raw_img = cv2.imread(fpath)
        if raw_img is None:
            continue
            
        aligned, align_success = aligner.align(raw_img)
        h, w = aligned.shape[:2]
        
        print(f"\n================ Evaluating {fname} ================")
        results[fname] = {"filename": fname, "alignment_valid": align_success, "fields": {}}
        
        # --- Task 3: TrOCR on Text Fields with Blank Detection ---
        text_fields = ["student_name", "school_name", "email"]
        for tf in text_fields:
            gt_val = gt_dict.get(tf)
            
            # Original crop
            orig_b = orig_cfg['fields'][tf]['normalized_bbox']
            ox1, oy1, ox2, oy2 = int(orig_b[0]*w), int(orig_b[1]*h), int(orig_b[2]*w), int(orig_b[3]*h)
            orig_crop = aligned[oy1:oy2, ox1:ox2]
            
            # Refined crop
            ref_b = refined_cfg['fields'][tf]['normalized_bbox']
            rx1, ry1, rx2, ry2 = int(ref_b[0]*w), int(ref_b[1]*h), int(ref_b[2]*w), int(ref_b[3]*h)
            ref_crop = aligned[ry1:ry2, rx1:rx2]
            
            # Check ink presence on refined crop
            has_ink, ink_ratio = detect_ink_presence(ref_crop, min_ink_ratio=0.008)
            
            # Run TrOCR on original crop
            t0 = time.time()
            rgb_orig = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2RGB)
            pv_orig = trocr_proc(Image.fromarray(rgb_orig), return_tensors='pt').pixel_values
            ids_orig = trocr_model.generate(pv_orig, max_new_tokens=32)
            trocr_orig_txt = trocr_proc.batch_decode(ids_orig, skip_special_tokens=True)[0].strip()
            trocr_orig_lat = (time.time() - t0) * 1000.0
            
            # Run TrOCR on refined crop (with blank gate)
            t0 = time.time()
            if not has_ink:
                trocr_refined_txt = ""  # Prevent hallucination on empty field
                trocr_refined_lat = (time.time() - t0) * 1000.0
            else:
                rgb_ref = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2RGB)
                pv_ref = trocr_proc(Image.fromarray(rgb_ref), return_tensors='pt').pixel_values
                ids_ref = trocr_model.generate(pv_ref, max_new_tokens=32)
                trocr_refined_txt = trocr_proc.batch_decode(ids_ref, skip_special_tokens=True)[0].strip()
                trocr_refined_lat = (time.time() - t0) * 1000.0
                
            # Run PaddleOCR on refined crop
            t0 = time.time()
            paddle_res = paddle_ocr.ocr(ref_crop, det=False, rec=True)
            paddle_txt = paddle_res[0][0][0].strip() if (paddle_res and paddle_res[0] and paddle_res[0][0]) else ""
            paddle_lat = (time.time() - t0) * 1000.0
            
            results[fname]["fields"][tf] = {
                "ground_truth": gt_val,
                "has_ink": has_ink,
                "ink_ratio": round(ink_ratio, 4),
                "trocr_original": {"text": trocr_orig_txt, "masked": mask_pii(trocr_orig_txt), "lat_ms": round(trocr_orig_lat, 1)},
                "trocr_refined_gated": {"text": trocr_refined_txt, "masked": mask_pii(trocr_refined_txt), "lat_ms": round(trocr_refined_lat, 1)},
                "paddleocr_rec": {"text": paddle_txt, "masked": mask_pii(paddle_txt), "lat_ms": round(paddle_lat, 1)}
            }
            print(f"[{tf}] GT: '{mask_pii(str(gt_val))}' | Has Ink: {has_ink} (ratio={ink_ratio:.3f})")
            print(f"   TrOCR Orig:    '{mask_pii(trocr_orig_txt)}' ({trocr_orig_lat:.0f}ms)")
            print(f"   TrOCR Refined: '{mask_pii(trocr_refined_txt)}' ({trocr_refined_lat:.0f}ms)")
            print(f"   Paddle Rec:    '{mask_pii(paddle_txt)}' ({paddle_lat:.0f}ms)")
            
        # --- Task 4: Specialized Boxed Digit Extraction for Mobile & Date ---
        digit_fields = ["student_mobile", "date"]
        for df in digit_fields:
            gt_val = gt_dict.get(df)
            boxes = orig_cfg['fields'][df].get('boxes', [])
            
            box_results = []
            assembled_digits = []
            
            for idx, b in enumerate(boxes):
                b_norm = b['normalized_bbox']
                bx1, by1, bx2, by2 = int(b_norm[0]*w), int(b_norm[1]*h), int(b_norm[2]*w), int(b_norm[3]*h)
                raw_box_crop = aligned[by1:by2, bx1:bx2]
                
                # Preprocess: remove border, check ink, isolate character
                clean_box, has_digit = remove_box_borders_and_isolate_digit(raw_box_crop)
                
                # Save sample debug box crop
                debug_path = f"storage/stage9_investigation/boxed_digits/{fname.replace('.jpeg','')}_{df}_box{idx}.png"
                cv2.imwrite(debug_path, clean_box)
                
                if not has_digit:
                    predicted_char = ""
                    conf = 1.0  # High confidence it is empty
                else:
                    # Run PaddleOCR character recognition on centered box
                    p_res = paddle_ocr.ocr(clean_box, det=False, rec=True)
                    if p_res and p_res[0] and p_res[0][0]:
                        raw_char, conf = p_res[0][0]
                        # Filter to numeric only
                        num_chars = [c for c in raw_char if c.isdigit()]
                        predicted_char = num_chars[0] if num_chars else ""
                    else:
                        predicted_char = ""
                        conf = 0.0
                        
                box_results.append({
                    "box_index": idx,
                    "has_digit": has_digit,
                    "predicted_char": predicted_char,
                    "conf": float(conf)
                })
                assembled_digits.append(predicted_char if predicted_char else "_")
                
            assembled_str = "".join(assembled_digits)
            
            # Business rule validation
            if df == "student_mobile":
                # Valid Indian mobile: 10 digits starting with 6-9
                digits_only = "".join([c for c in assembled_digits if c.isdigit()])
                is_valid = (len(digits_only) == 10 and digits_only[0] in "6789")
                status = "VALID" if is_valid else ("INCOMPLETE" if len(digits_only) < 10 else "INVALID_PREFIX")
            elif df == "date":
                # Valid date: DD/MM/YYYY
                is_valid = ("_" not in assembled_str and len(assembled_str) == 8)
                status = "VALID" if is_valid else "NEEDS_REVIEW"
            else:
                status = "UNKNOWN"
                
            results[fname]["fields"][df] = {
                "ground_truth": gt_val,
                "assembled_string": assembled_str,
                "masked_gt": mask_pii(str(gt_val)),
                "masked_assembled": mask_pii(assembled_str),
                "validation_status": status,
                "boxes": box_results
            }
            print(f"[{df}] GT: '{mask_pii(str(gt_val))}' | Assembled Boxes: '{mask_pii(assembled_str)}' | Status: {status}")

    # Save complete Stage 9 investigation results
    os.makedirs("storage/stage9_investigation", exist_ok=True)
    with open("storage/stage9_investigation/stage9_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nStage 9 benchmark results saved to storage/stage9_investigation/stage9_evaluation_results.json")

if __name__ == "__main__":
    main()
