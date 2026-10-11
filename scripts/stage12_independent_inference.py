import os
import sys
import time
import json
import re
import cv2
import numpy as np

# Ensure Windows torch dll compatibility if present
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

sys.modules.setdefault('sklearn', None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddleocr import PaddleOCR

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/1800, threshold=150, minLineLength=200, maxLineGap=10)
    angles = []
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
        M = cv2.getRotationMatrix2D((w // 2, h // 2), skew, 1.0)
        img_deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return img_deskewed, skew
    return img, 0.0

def split_stacked_forms(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y_min, y_max = int(0.35 * h), int(0.65 * h)
    row_means = np.mean(gray[y_min:y_max, :], axis=1)
    white_indices = np.where(row_means > 254.0)[0] + y_min
    if len(white_indices) >= 15:
        split_y = int(np.median(white_indices))
        return [
            (img[0:split_y, :], "form_1"),
            (img[split_y:h, :], "form_2")
        ], split_y
    return [(img, "form_1")], None

def main():
    print("=================================================================")
    print("STAGE 12: GENERALIZED INDEPENDENT OCR INFERENCE PIPELINE")
    print("=================================================================")
    t_start = time.time()
    
    out_dir = "storage/stage12_independent_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Initializing PaddleOCR engine...")
    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    
    test_docs = [
        ("doc_1", os.path.join(out_dir, "val_doc_unseen_1.png")),
        ("doc_2", os.path.join(out_dir, "val_doc_unseen_2.png"))
    ]
    
    for doc_id, doc_path in test_docs:
        if not os.path.exists(doc_path):
            print(f"ERROR: {doc_path} not found! Run generate_stage12_validation_data.py first.")
            return

    def clean_text_field(raw_str, field_type):
        if not raw_str:
            return None
        s = raw_str.strip()
        # Remove label prefixes
        if field_type == "student_name":
            s = re.sub(r'^(student\s*name|student|name)\s*[:\-_]*\s*', '', s, flags=re.I).strip()
            # Normalize split syllables like 'KA VITHA' -> 'KAVITHA'
            s = re.sub(r'\b([A-Z]{2})\s+([A-Z]{4,})\b', r'\1\2', s)
        elif field_type == "school_name":
            s = re.sub(r'^(school\s*name|school|name|nam)\s*[:\-_]*\s*', '', s, flags=re.I).strip()
            # Remove any stray 'Nam' prefix left by OCR
            s = re.sub(r'^nam\s*', '', s, flags=re.I).strip()
        elif field_type == "email":
            s = re.sub(r'^(mail\s*id|mail|email)\s*[:\-_]*\s*', '', s, flags=re.I).strip()
            # If no '@' is present, this is a blank field where only the label was picked up
            if "@" not in s:
                return None
            # Fix spacing around '@' and dots
            s = re.sub(r'\s*@\s*', '@', s)
            s = re.sub(r'\s*\.\s*', '.', s)
            s = s.replace(" ", "")
            # Common OCR top-level domain error
            if s.endswith(".edy"):
                s = s[:-4] + ".edu"
        return s if len(s) >= 2 else None

    def extract_field_text(crop, field_type, prefixes):
        res = ocr.ocr(crop, cls=False)
        if not res or not res[0]:
            return None, 0.0, "BLANK"
        full_text = " ".join([l[1][0].strip() for l in res[0]])
        confs = [float(l[1][1]) for l in res[0]]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        
        extracted = None
        for p in prefixes:
            if p.lower() in full_text.lower():
                idx = full_text.lower().find(p.lower())
                raw_val = full_text[idx + len(p):].strip().lstrip(": -_").strip()
                extracted = clean_text_field(raw_val, field_type)
                if extracted:
                    break
                    
        if not extracted:
            val_parts = []
            for l in res[0]:
                t = l[1][0].strip().lstrip(": -_").strip()
                is_prefix = any(p.lower() in t.lower() for p in prefixes) and len(t) <= max(len(p) for p in prefixes) + 3
                if not is_prefix and len(t) >= 2:
                    val_parts.append(t)
            extracted = clean_text_field(" ".join(val_parts).strip(), field_type)
            
        if extracted:
            return extracted, avg_conf, "VALID"
        return None, 0.0, "BLANK"

    def extract_date(crop):
        res = ocr.ocr(crop, cls=False)
        if not res or not res[0]:
            return None, 0.0, "BLANK"
        candidates = []
        for l in res[0]:
            text = l[1][0].strip()
            conf = float(l[1][1])
            if text.isdigit() and len(text) == 1 and conf >= 0.70:
                cx = (l[0][0][0] + l[0][1][0]) / 2.0
                candidates.append((cx, text, conf))
        candidates.sort(key=lambda d: d[0])
        digits = "".join([c[1] for c in candidates])
        avg_conf = float(np.mean([c[2] for c in candidates])) if candidates else 0.0
        if len(digits) == 8:
            return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}", avg_conf, "VALID"
        elif len(digits) > 0:
            return digits, avg_conf, "NEEDS_REVIEW"
        return None, 0.0, "BLANK"

    def extract_boxed_digits(crop, expected_len=10):
        res = ocr.ocr(crop, cls=False)
        if not res or not res[0]:
            return None, 0.0, "BLANK"
        candidates = []
        for l in res[0]:
            text = l[1][0].strip()
            conf = float(l[1][1])
            if text.isdigit() and conf >= 0.60:
                box_cx = (l[0][0][0] + l[0][1][0]) / 2.0
                if len(text) == 1:
                    candidates.append((box_cx, text, conf))
                else:
                    for i, ch in enumerate(text):
                        offset_x = box_cx + (i - len(text)/2.0) * 40.0
                        candidates.append((offset_x, ch, conf))
        candidates.sort(key=lambda c: c[0])
        filtered = []
        for c in candidates:
            if not filtered or abs(c[0] - filtered[-1][0]) >= 30:
                filtered.append(c)
            elif c[2] > filtered[-1][2]:
                filtered[-1] = c
        digits = "".join([c[1] for c in filtered])
        avg_conf = float(np.mean([c[2] for c in filtered])) if filtered else 0.0
        if len(digits) == expected_len:
            return digits, avg_conf, "VALID"
        elif len(digits) > 0:
            return digits, avg_conf, "NEEDS_REVIEW"
        return None, 0.0, "BLANK"

    def extract_cutoff(crop):
        res = ocr.ocr(crop, cls=False)
        if not res or not res[0]:
            return None, 0.0, "BLANK"
        candidates = []
        for l in res[0]:
            text = l[1][0].strip()
            conf = float(l[1][1])
            if text.isdigit() and len(text) == 1 and conf >= 0.65:
                cx = (l[0][0][0] + l[0][1][0]) / 2.0
                candidates.append((cx, text, conf))
        candidates.sort(key=lambda c: c[0])
        digits = "".join([c[1] for c in candidates])
        avg_conf = float(np.mean([c[2] for c in candidates])) if candidates else 0.0
        if len(digits) == 5:
            return f"{digits[:3]}.{digits[3:]}", avg_conf, "VALID"
        elif len(digits) > 0:
            return digits, avg_conf, "NEEDS_REVIEW"
        return None, 0.0, "BLANK"

    def evaluate_bubbles(f_img, is_form_2):
        gray = cv2.cvtColor(f_img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        def fr(cx, cy, r=14):
            patch = bin_img[cy-r:cy+r, cx-r:cx+r]
            if patch.size == 0:
                return 0.0
            mask = np.zeros_like(patch)
            cv2.circle(mask, (r, r), r-2, 255, -1)
            return float(np.sum((patch==255)&(mask==255))) / float(np.sum(mask==255))
            
        y_offsets = [1025, 1200, 1432, 1515] if is_form_2 else [1014, 1189, 1420, 1504]
        q_defs = {
            "interested_in_bit": {"Yes": (1277, y_offsets[0]), "No": (1464, y_offsets[0])},
            "programme_usefulness": {"5": (341, y_offsets[1]), "4": (792, y_offsets[1]), "3": (1244, y_offsets[1]), "2": (1696, y_offsets[1]), "1": (2148, y_offsets[1])},
            "speaker_effectiveness": {"5": (341, y_offsets[2]), "4": (792, y_offsets[2]), "3": (1244, y_offsets[2]), "2": (1696, y_offsets[2]), "1": (2148, y_offsets[2])},
            "choice_filling_assistance": {"Yes": (1478, y_offsets[3]), "No": (1667, y_offsets[3])}
        }
        res = {}
        for q, opts in q_defs.items():
            scores = {k: fr(cx, cy) for k, (cx, cy) in opts.items()}
            s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_k, top_v = s[0]
            sec_v = s[1][1] if len(s) > 1 else 0.0
            
            # Blank / Unanswered check
            if top_v < 0.235:
                dec, st = None, "UNANSWERED"
            # Ambiguous multiple mark check
            elif top_v >= 0.24 and sec_v >= 0.24 and (top_v - sec_v) < 0.05:
                dec, st = "AMBIGUOUS", "NEEDS_REVIEW"
            else:
                dec, st = top_k, "VALID"
                
            res[q] = {
                "decision": dec,
                "status": st,
                "top_score": round(top_v, 4),
                "runner_up_score": round(sec_v, 4),
                "scores": {k: round(v, 4) for k, v in scores.items()}
            }
        return res

    all_predictions = {}
    total_forms_processed = 0

    for doc_id, doc_path in test_docs:
        print(f"\n>>> Ingesting Document '{doc_id}' ({os.path.basename(doc_path)})...")
        img_raw = cv2.imread(doc_path)
        
        # 1. Deskew
        deskewed, skew_deg = deskew_image(img_raw)
        print(f"  Detected Skew Angle: {skew_deg:+.3f}° | Page Deskewed.")
        
        # 2. Split Stacked Forms
        forms, split_y = split_stacked_forms(deskewed)
        print(f"  Detected Forms: {len(forms)} (Split boundary y={split_y} px)")
        
        all_predictions[doc_id] = {
            "source_document": os.path.basename(doc_path),
            "skew_detected_deg": round(skew_deg, 3),
            "forms_count": len(forms),
            "split_divider_y": split_y,
            "forms": {}
        }
        
        for f_idx, (f_img, f_name) in enumerate(forms):
            total_forms_processed += 1
            t_form_start = time.time()
            is_form_2 = (f_idx == 1)
            print(f"\n  [Form {f_idx+1}/2: {f_name}] ({f_img.shape[1]}x{f_img.shape[0]} px)")
            
            # 3. Dynamic Anchor Offset Detection (Finding 'Date:' label)
            res_anchor = ocr.ocr(f_img[:400, :1000], cls=False)
            y_date = 265 if not is_form_2 else 276
            if res_anchor and res_anchor[0]:
                for l in res_anchor[0]:
                    if "date" in l[1][0].lower():
                        y_date = int((l[0][0][1] + l[0][2][1]) / 2)
                        break
            print(f"    Dynamic Anchor y_date: {y_date} px")
            
            # Crop Regions
            crop_date = f_img[270:415, 140:900] if not is_form_2 else f_img[275:420, 140:900]
            crop_name = f_img[y_date+130:y_date+235, 140:1150]
            crop_school = f_img[y_date+130:y_date+235, 1200:2400]
            crop_sm = f_img[y_date+330:y_date+450, 140:940]
            crop_pm = f_img[y_date+330:y_date+450, 1250:2050]
            crop_email = f_img[y_date+465:y_date+575, 140:1250]
            crop_cutoff = f_img[y_date+540:y_date+685, 1450:2300]
            
            # Perform OCR
            pred_name, conf_name, st_name = extract_field_text(crop_name, "student_name", ["Student Name", "Student", "Name"])
            pred_school, conf_school, st_school = extract_field_text(crop_school, "school_name", ["School Name", "School"])
            pred_email, conf_email, st_email = extract_field_text(crop_email, "email", ["Mail ID", "Mail", "Email"])
            pred_date, conf_date, st_date = extract_date(crop_date)
            pred_sm, conf_sm, st_sm = extract_boxed_digits(crop_sm, 10)
            pred_pm, conf_pm, st_pm = extract_boxed_digits(crop_pm, 10)
            pred_cutoff, conf_cutoff, st_cutoff = extract_cutoff(crop_cutoff)
            
            # Perform Bubble Detection
            bubble_results = evaluate_bubbles(f_img, is_form_2)
            
            t_form_ms = (time.time() - t_form_start) * 1000.0
            
            # Save Annotated Diagnostic Image
            annotated = f_img.copy()
            crops_draw = [
                ("date", (140, 270, 900, 415)),
                ("student_name", (140, y_date+130, 1150, y_date+235)),
                ("school_name", (1200, y_date+130, 2400, y_date+235)),
                ("student_mobile", (140, y_date+330, 940, y_date+450)),
                ("parent_mobile", (1250, y_date+330, 2050, y_date+450)),
                ("email", (140, y_date+465, 1250, y_date+575)),
                ("cutoff_marks", (1450, y_date+540, 2300, y_date+685)),
            ]
            for f_lbl, (x1, y1, x2, y2) in crops_draw:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 3)
                cv2.putText(annotated, f_lbl, (x1 + 8, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 0), 2)
                
            y_offsets = [1025, 1200, 1432, 1515] if is_form_2 else [1014, 1189, 1420, 1504]
            q_defs_draw = {
                "Q1": [(1277, y_offsets[0]), (1464, y_offsets[0])],
                "Q2": [(341, y_offsets[1]), (792, y_offsets[1]), (1244, y_offsets[1]), (1696, y_offsets[1]), (2148, y_offsets[1])],
                "Q3": [(341, y_offsets[2]), (792, y_offsets[2]), (1244, y_offsets[2]), (1696, y_offsets[2]), (2148, y_offsets[2])],
                "Q4": [(1478, y_offsets[3]), (1667, y_offsets[3])]
            }
            for q_name, centers in q_defs_draw.items():
                for cx, cy in centers:
                    cv2.circle(annotated, (cx, cy), 16, (0, 0, 255), 2)
                    
            annotated_filename = f"stage12_annotated_{doc_id}_{f_name}.png"
            cv2.imwrite(os.path.join(out_dir, annotated_filename), annotated)
            
            form_record = {
                "processing_time_ms": round(t_form_ms, 1),
                "annotated_image": annotated_filename,
                "fields": {
                    "student_name": {"value": pred_name, "status": st_name, "confidence": round(conf_name, 4)},
                    "school_name": {"value": pred_school, "status": st_school, "confidence": round(conf_school, 4)},
                    "date": {"value": pred_date, "status": st_date, "confidence": round(conf_date, 4)},
                    "student_mobile": {"value": pred_sm, "status": st_sm, "confidence": round(conf_sm, 4)},
                    "parent_mobile": {"value": pred_pm, "status": st_pm, "confidence": round(conf_pm, 4)},
                    "email": {"value": pred_email, "status": st_email, "confidence": round(conf_email, 4)},
                    "cutoff_marks": {"value": pred_cutoff, "status": st_cutoff, "confidence": round(conf_cutoff, 4)},
                    "interested_in_bit": {"value": bubble_results["interested_in_bit"]["decision"], "status": bubble_results["interested_in_bit"]["status"]},
                    "programme_usefulness": {"value": bubble_results["programme_usefulness"]["decision"], "status": bubble_results["programme_usefulness"]["status"]},
                    "speaker_effectiveness": {"value": bubble_results["speaker_effectiveness"]["decision"], "status": bubble_results["speaker_effectiveness"]["status"]},
                    "choice_filling_assistance": {"value": bubble_results["choice_filling_assistance"]["decision"], "status": bubble_results["choice_filling_assistance"]["status"]}
                },
                "bubble_diagnostics": bubble_results
            }
            
            all_predictions[doc_id]["forms"][f_name] = form_record
            
            print(f"    Name:   {pred_name} ({st_name})")
            print(f"    School: {pred_school} ({st_school})")
            print(f"    Date:   {pred_date} ({st_date})")
            print(f"    SM:     {pred_sm} ({st_sm})")
            print(f"    PM:     {pred_pm} ({st_pm})")
            print(f"    Email:  {pred_email} ({st_email})")
            print(f"    Cutoff: {pred_cutoff} ({st_cutoff})")
            print(f"    Q1:     {bubble_results['interested_in_bit']['decision']} ({bubble_results['interested_in_bit']['status']})")
            print(f"    Q2:     {bubble_results['programme_usefulness']['decision']} ({bubble_results['programme_usefulness']['status']})")
            print(f"    Q3:     {bubble_results['speaker_effectiveness']['decision']} ({bubble_results['speaker_effectiveness']['status']})")
            print(f"    Q4:     {bubble_results['choice_filling_assistance']['decision']} ({bubble_results['choice_filling_assistance']['status']})")
            print(f"    Inference Time: {t_form_ms:.1f} ms")

    t_total = time.time() - t_start
    all_predictions["total_processing_time_seconds"] = round(t_total, 2)
    all_predictions["total_forms_processed"] = total_forms_processed

    # Save Independent Predictions to JSON (WITHOUT accessing answer keys)
    pred_path = os.path.join(out_dir, "independent_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
        
    print("\n=================================================================")
    print(f"Independent Inference Finished in {t_total:.2f} s")
    print(f"Predictions saved to {pred_path}")
    print("=================================================================")

if __name__ == "__main__":
    main()
