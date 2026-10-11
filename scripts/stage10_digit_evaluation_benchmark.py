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

def main():
    print("=== STAGE 10: DEDICATED HANDWRITTEN DIGIT OCR EVALUATION ===")
    
    # 1. Load PaddleOCR
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    
    gt_mobile = {
        "form1.jpeg": "9432164321",
        "form2.jpeg": "9042642526",
        "form3.jpeg": "8088997798",
        "form4.jpeg": "9754608192",
        "form5.jpeg": "8304934091",
    }
    
    # Evaluate across all 5 forms
    eval_results = {}
    confusion = {str(d): {str(p): 0 for p in range(10)} for d in range(10)}
    for d in range(10):
        confusion[str(d)]["missing"] = 0
        confusion[str(d)]["extra"] = 0
        
    total_chars = 0
    total_dist = 0
    exact_matches = 0
    latencies = []
    
    print("\n--- Direct OCR Extraction on True Mobile Number Regions ---")
    for fname, gt in gt_mobile.items():
        fpath = os.path.join("cnn_dataset", fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
            
        t0 = time.time()
        res = ocr.ocr(img, cls=False)
        lat = (time.time() - t0) * 1000.0
        latencies.append(lat)
        
        # Locate the student mobile number block using spatial proximity below "Student Mobile Number:" label
        extracted_digits = ""
        best_conf = 0.0
        
        if res and res[0]:
            lines = res[0]
            # 1. First locate the "Student Mobile Number" label bounding box
            label_box = None
            for line in lines:
                txt = line[1][0].strip().lower()
                if "student mobile" in txt:
                    label_box = line[0]
                    break
            
            if label_box is not None:
                # Label center and bottom Y
                lbl_x = (label_box[0][0] + label_box[1][0]) / 2.0
                lbl_bottom_y = label_box[2][1]
                
                # Find line immediately below label (within dy: 10 to 120px, and dx < 300px)
                candidates = []
                for line in lines:
                    poly = line[0]
                    cand_txt = line[1][0].strip()
                    cand_conf = float(line[1][1])
                    cand_x = (poly[0][0] + poly[1][0]) / 2.0
                    cand_top_y = poly[0][1]
                    
                    dy = cand_top_y - lbl_bottom_y
                    dx = abs(cand_x - lbl_x)
                    
                    if 0 <= dy <= 120 and dx < 350:
                        digits_only = "".join([c for c in cand_txt if c.isdigit()])
                        if len(digits_only) >= 7:
                            candidates.append((dy, digits_only, cand_conf))
                            
                if candidates:
                    candidates.sort(key=lambda c: c[0])  # closest dy
                    extracted_digits = candidates[0][1]
                    best_conf = candidates[0][2]
                    
            # Fallback if label not detected: search any line with >= 8 digits starting with 8 or 9
            if not extracted_digits:
                for line in lines:
                    clean_num = "".join([c for c in line[1][0] if c.isdigit()])
                    if 8 <= len(clean_num) <= 11 and (clean_num.startswith("9") or clean_num.startswith("8") or clean_num.startswith("4")):
                        extracted_digits = clean_num
                        best_conf = float(line[1][1])
                        break
                        
        dist = levenshtein_distance(extracted_digits, gt)
        cer = calculate_cer(extracted_digits, gt)
        is_exact = (extracted_digits == gt)
        
        total_chars += len(gt)
        total_dist += dist
        if is_exact:
            exact_matches += 1
            
        # Update confusion stats where lengths align or character by character
        min_len = min(len(gt), len(extracted_digits))
        for i in range(min_len):
            g_char = gt[i]
            p_char = extracted_digits[i]
            if g_char.isdigit() and p_char.isdigit():
                confusion[g_char][p_char] += 1
        if len(extracted_digits) < len(gt):
            for i in range(len(extracted_digits), len(gt)):
                confusion[gt[i]]["missing"] += 1
                
        eval_results[fname] = {
            "ground_truth": gt,
            "masked_gt": mask_pii(gt),
            "extracted_digits": extracted_digits,
            "masked_pred": mask_pii(extracted_digits),
            "edit_distance": dist,
            "cer": round(cer, 4),
            "is_exact_match": is_exact,
            "confidence": round(best_conf, 4),
            "latency_ms": round(lat, 1)
        }
        
        print(f"[{fname}] GT: {mask_pii(gt)} | Extracted: '{mask_pii(extracted_digits)}' | Dist: {dist} | CER: {cer:.2%} | Exact: {is_exact}")
        
    overall_cer = total_dist / total_chars if total_chars > 0 else 1.0
    exact_acc = exact_matches / len(gt_mobile) if gt_mobile else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    
    summary = {
        "total_forms": len(gt_mobile),
        "total_characters": total_chars,
        "total_edit_distance": total_dist,
        "overall_cer": round(overall_cer, 4),
        "overall_cer_pct": round(overall_cer * 100, 2),
        "exact_matches": exact_matches,
        "exact_match_accuracy_pct": round(exact_acc * 100, 2),
        "avg_processing_time_ms": round(avg_latency, 1),
        "confusion_matrix": confusion,
        "detailed_results": eval_results
    }
    
    print("\n=== OVERALL BENCHMARK METRICS ===")
    print(f"Total Evaluated Forms:       {len(gt_mobile)}")
    print(f"Total Character Distance:    {total_dist} / {total_chars}")
    print(f"Character Error Rate (CER):  {overall_cer:.2%}")
    print(f"Exact 10-Digit Match Rate:   {exact_matches}/{len(gt_mobile)} ({exact_acc:.1%})")
    print(f"Avg Processing Latency:      {avg_latency:.1f} ms")
    
    os.makedirs("storage/stage10_digit_evaluation", exist_ok=True)
    with open("storage/stage10_digit_evaluation/stage10_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\nSaved evaluation artifact to storage/stage10_digit_evaluation/stage10_metrics.json")

if __name__ == "__main__":
    main()
