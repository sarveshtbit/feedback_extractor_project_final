import os
import sys
import time
import json
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure Windows torch dll directory is loaded
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

# Ensure compatibility
sys.modules.setdefault('sklearn', None)
import cv2
import numpy as np
from PIL import Image

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

def mask_pii(text: str) -> str:
    if not text or len(text.strip()) == 0:
        return "***"
    t = text.strip()
    if len(t) <= 4:
        return "***"
    return t[:2] + "***" + t[-2:]

def run_paddle_crop(ocr, crop_bgr: np.ndarray) -> tuple[str, float, float]:
    """Runs PaddleOCR on a single crop, returning (text, conf, latency_ms)."""
    t0 = time.time()
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, 0.0
    try:
        # cls=True for angle classification
        res = ocr.ocr(crop_bgr, cls=False)
        latency = (time.time() - t0) * 1000.0
        if not res or not res[0]:
            return "", 0.0, latency
        texts = []
        confs = []
        for line in res[0]:
            _, (txt, c) = line
            texts.append(txt.strip())
            confs.append(float(c))
        full_text = " ".join(texts)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return full_text, avg_conf, latency
    except Exception as e:
        latency = (time.time() - t0) * 1000.0
        return "", 0.0, latency

def run_trocr_crop(proc, model, crop_bgr: np.ndarray) -> tuple[str, float, float]:
    """Runs TrOCR on a single crop, returning (text, conf, latency_ms)."""
    t0 = time.time()
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, 0.0
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pixel_values = proc(pil_img, return_tensors='pt').pixel_values
        gen_ids = model.generate(pixel_values, max_new_tokens=32)
        text = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        latency = (time.time() - t0) * 1000.0
        return text, 0.85 if text else 0.0, latency
    except Exception as e:
        latency = (time.time() - t0) * 1000.0
        return "", 0.0, latency

class StandaloneFormAligner:
    def __init__(self, template_config=None):
        self.canvas_width = 2479
        self.canvas_height = 3508
        self.ref_image = cv2.imread("bannari_amman_form_template.png") if os.path.exists("bannari_amman_form_template.png") else None
        self.dst_markers = np.array([
            [140, 140],
            [2339, 140],
            [2339, 3368],
            [140, 3368]
        ], dtype=np.float32)

    def is_valid_homography(self, H, src_pts, dst_pts):
        if H is None or H.shape != (3, 3):
            return False
        det = np.linalg.det(H)
        if det <= 0.001 or det > 1000.0:
            return False
        return True

    def detect_markers(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        img_h, img_w = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = (img_h * img_w) * 0.0001
        max_area = (img_h * img_w) * 0.015
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        candidates.append((cx, cy, area))
        if len(candidates) < 4:
            return None
        mid_x, mid_y = img_w / 2.0, img_h / 2.0
        quad_tl, quad_tr, quad_br, quad_bl = [], [], [], []
        for cx, cy, area in candidates:
            if cx < mid_x and cy < mid_y:
                quad_tl.append((cx, cy))
            elif cx >= mid_x and cy < mid_y:
                quad_tr.append((cx, cy))
            elif cx >= mid_x and cy >= mid_y:
                quad_br.append((cx, cy))
            elif cx < mid_x and cy >= mid_y:
                quad_bl.append((cx, cy))
        def best_corner(pts, tx, ty):
            if not pts:
                return None
            return min(pts, key=lambda p: (p[0] - tx)**2 + (p[1] - ty)**2)
        tl = best_corner(quad_tl, 0, 0)
        tr = best_corner(quad_tr, img_w, 0)
        br = best_corner(quad_br, img_w, img_h)
        bl = best_corner(quad_bl, 0, img_h)
        if None in (tl, tr, br, bl):
            return None
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def align_with_orb_features(self, image):
        if self.ref_image is None:
            return None, None
        try:
            orb = cv2.ORB_create(3000)
            kp1, des1 = orb.detectAndCompute(image, None)
            kp2, des2 = orb.detectAndCompute(self.ref_image, None)
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return None, None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) < 15:
                return None, None
            top_matches = matches[:50]
            src_pts = np.float32([kp1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None and H.shape == (3, 3):
                warped = cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height))
                return warped, H
            return None, None
        except Exception:
            return None, None

    def letterbox_to_canvas(self, image):
        if image is None or image.size == 0:
            return np.full((self.canvas_height, self.canvas_width, 3), 255, dtype=np.uint8)
        img_h, img_w = image.shape[:2]
        scale = min(self.canvas_width / float(img_w), self.canvas_height / float(img_h))
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 255, dtype=np.uint8)
        pad_x = (self.canvas_width - new_w) // 2
        pad_y = (self.canvas_height - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas

    def align(self, image):
        try:
            src_pts = self.detect_markers(image)
            if src_pts is not None:
                H, _ = cv2.findHomography(src_pts, self.dst_markers, cv2.RANSAC, 5.0)
                if self.is_valid_homography(H, src_pts, self.dst_markers):
                    warped = cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height))
                    return warped, H, True
            orb_warped, orb_H = self.align_with_orb_features(image)
            if orb_warped is not None:
                return orb_warped, orb_H, True
            scaled = self.letterbox_to_canvas(image)
            return scaled, None, False
        except Exception:
            scaled = self.letterbox_to_canvas(image)
            return scaled, None, False

def main():
    print("=== TASK 4: COMPARING PADDLEOCR VS TROCR ACROSS 4 APPROACHES ===")
    
    # 1. Load TrOCR first (initializes torch runtime cleanly)
    print("Loading TrOCR...")
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    trocr_proc = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # 2. Load PaddleOCR
    print("Loading PaddleOCR...")
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    
    # 3. Load configurations
    with open("config/form_template.yaml", "r") as f:
        orig_cfg = yaml.safe_load(f)
    with open("config/form_template_experimental.yaml", "r") as f:
        refined_cfg = yaml.safe_load(f)
        
    aligner = StandaloneFormAligner()
    
    gt_mobile = {
        "form1.jpeg": "9432164321",
        "form2.jpeg": "9042642526",
        "form3.jpeg": "8088997798",
        "form4.jpeg": "9754608192",
        "form5.jpeg": "8304934091",
    }
    
    test_files = [
        "bannari_amman_form_template.png",
        "cnn_dataset/form1.jpeg",
        "cnn_dataset/form2.jpeg",
        "cnn_dataset/form3.jpeg",
        "cnn_dataset/form4.jpeg",
        "cnn_dataset/form5.jpeg",
        "debug_original_forms/page_1_form_1.png",
        "debug_original_forms/page_1_form_2.png",
    ]
    
    fields = ["student_name", "school_name", "date", "student_mobile", "parent_mobile", "email", "cutoff_marks"]
    
    results = []
    
    # Metrics aggregators for Approach A, B, C, D
    approaches = ["approach_a_paddle_orig", "approach_b_paddle_refined", "approach_c_trocr_refined", "approach_d_paddle_boxes"]
    field_stats = {app: {f: {"total": 0, "failures": 0, "latencies": []} for f in fields} for app in approaches}
    gt_stats = {app: {"total_chars": 0, "total_dist": 0, "exact_matches": 0, "total_fields": 0} for app in approaches}

    for fpath in test_files:
        if not os.path.exists(fpath):
            continue
        fname = os.path.basename(fpath)
        raw_img = cv2.imread(fpath)
        aligned, _, _ = aligner.align(raw_img)
        h, w = aligned.shape[:2]
        
        file_record = {"file": fpath, "filename": fname, "fields": {}}
        print(f"\n================ Processing {fname} ================")
        
        for f_name in fields:
            # Coordinates
            orig_b = orig_cfg['fields'][f_name]['normalized_bbox']
            ox1, oy1, ox2, oy2 = int(orig_b[0]*w), int(orig_b[1]*h), int(orig_b[2]*w), int(orig_b[3]*h)
            orig_crop = aligned[oy1:oy2, ox1:ox2]
            
            ref_b = refined_cfg['fields'][f_name]['normalized_bbox']
            rx1, ry1, rx2, ry2 = int(ref_b[0]*w), int(ref_b[1]*h), int(ref_b[2]*w), int(ref_b[3]*h)
            ref_crop = aligned[ry1:ry2, rx1:rx2]
            
            # Approach A: PaddleOCR on original crop
            text_a, conf_a, lat_a = run_paddle_crop(paddle_ocr, orig_crop)
            
            # Approach B: PaddleOCR on refined write-in crop
            text_b, conf_b, lat_b = run_paddle_crop(paddle_ocr, ref_crop)
            
            # Approach C: TrOCR on refined write-in crop
            text_c, conf_c, lat_c = run_trocr_crop(trocr_proc, trocr_model, ref_crop)
            
            # Approach D: PaddleOCR on individual character boxes (if applicable)
            text_d = ""
            conf_d = 0.0
            lat_d = 0.0
            if "boxes" in orig_cfg['fields'][f_name]:
                box_texts = []
                box_confs = []
                t0_d = time.time()
                for b in orig_cfg['fields'][f_name]['boxes']:
                    bx1, by1, bx2, by2 = int(b['normalized_bbox'][0]*w), int(b['normalized_bbox'][1]*h), int(b['normalized_bbox'][2]*w), int(b['normalized_bbox'][3]*h)
                    # Inset 5% to remove border lines
                    bw_b, bh_b = bx2 - bx1, by2 - by1
                    pad_x, pad_y = int(bw_b * 0.06), int(bh_b * 0.06)
                    single_box = aligned[by1+pad_y : by2-pad_y, bx1+pad_x : bx2-pad_x]
                    btxt, bconf, _ = run_paddle_crop(paddle_ocr, single_box)
                    # Clean digits
                    clean_d = "".join([c for c in btxt if c.isalnum() or c == '.'])
                    box_texts.append(clean_d[0] if clean_d else "")
                    if bconf > 0:
                        box_confs.append(bconf)
                lat_d = (time.time() - t0_d) * 1000.0
                text_d = "".join(box_texts)
                conf_d = float(np.mean(box_confs)) if box_confs else 0.0
            else:
                text_d, conf_d, lat_d = text_b, conf_b, lat_b

            # Update stats
            preds = {
                "approach_a_paddle_orig": (text_a, conf_a, lat_a),
                "approach_b_paddle_refined": (text_b, conf_b, lat_b),
                "approach_c_trocr_refined": (text_c, conf_c, lat_c),
                "approach_d_paddle_boxes": (text_d, conf_d, lat_d),
            }
            
            for app, (ptxt, pconf, plat) in preds.items():
                field_stats[app][f_name]["total"] += 1
                field_stats[app][f_name]["latencies"].append(plat)
                if not ptxt or len(ptxt.strip()) == 0:
                    field_stats[app][f_name]["failures"] += 1
                    
            # Ground truth comparison for student_mobile
            if f_name == "student_mobile" and fname in gt_mobile:
                gt = gt_mobile[fname]
                for app, (ptxt, _, _) in preds.items():
                    digits_only = "".join([c for c in ptxt if c.isdigit()])
                    dist = levenshtein_distance(digits_only, gt)
                    gt_stats[app]["total_chars"] += len(gt)
                    gt_stats[app]["total_dist"] += dist
                    gt_stats[app]["total_fields"] += 1
                    if digits_only == gt:
                        gt_stats[app]["exact_matches"] += 1

            field_record = {
                "ground_truth": mask_pii(gt_mobile.get(fname)) if f_name == "student_mobile" else None,
                "approach_a_paddle_orig": {"raw": text_a, "masked": mask_pii(text_a), "conf": round(conf_a, 4), "lat_ms": round(lat_a, 1)},
                "approach_b_paddle_refined": {"raw": text_b, "masked": mask_pii(text_b), "conf": round(conf_b, 4), "lat_ms": round(lat_b, 1)},
                "approach_c_trocr_refined": {"raw": text_c, "masked": mask_pii(text_c), "conf": round(conf_c, 4), "lat_ms": round(lat_c, 1)},
                "approach_d_paddle_boxes": {"raw": text_d, "masked": mask_pii(text_d), "conf": round(conf_d, 4), "lat_ms": round(lat_d, 1)},
            }
            file_record["fields"][f_name] = field_record
            
            print(f"[{f_name}]")
            print(f"  A (Paddle Orig):    '{mask_pii(text_a)}' (conf: {conf_a:.2f}, {lat_a:.0f}ms)")
            print(f"  B (Paddle Refined): '{mask_pii(text_b)}' (conf: {conf_b:.2f}, {lat_b:.0f}ms)")
            print(f"  C (TrOCR Refined):  '{mask_pii(text_c)}' (conf: {conf_c:.2f}, {lat_c:.0f}ms)")
            print(f"  D (Paddle Boxes):   '{mask_pii(text_d)}' (conf: {conf_d:.2f}, {lat_d:.0f}ms)")

        results.append(file_record)

    # Compile aggregated report
    summary = {
        "ground_truth_mobile_evaluation": {},
        "field_level_performance": {},
        "raw_results": results
    }
    
    print("\n================ EVALUATION SUMMARY ================")
    print("--- Ground Truth Evaluation (student_mobile: 5 physical forms) ---")
    for app in approaches:
        st = gt_stats[app]
        cer = st["total_dist"] / max(1, st["total_chars"])
        exact_acc = (st["exact_matches"] / max(1, st["total_fields"])) * 100.0
        summary["ground_truth_mobile_evaluation"][app] = {
            "cer": round(cer, 4),
            "cer_pct": round(cer * 100, 2),
            "exact_matches": st["exact_matches"],
            "total_samples": st["total_fields"],
            "exact_accuracy_pct": round(exact_acc, 2)
        }
        print(f"  {app:30} | CER: {cer:.4f} ({cer*100:.1f}%) | Exact Match: {st['exact_matches']}/{st['total_fields']} ({exact_acc:.1f}%)")

    print("\n--- Field-Level Performance Breakdown ---")
    for f in fields:
        summary["field_level_performance"][f] = {}
        print(f"\n[Field: {f}]")
        for app in approaches:
            st = field_stats[app][f]
            fail_rate = (st["failures"] / max(1, st["total"])) * 100.0
            avg_lat = float(np.mean(st["latencies"])) if st["latencies"] else 0.0
            summary["field_level_performance"][f][app] = {
                "total_evaluations": st["total"],
                "unreadable_empty_count": st["failures"],
                "unreadable_rate_pct": round(fail_rate, 2),
                "avg_latency_ms": round(avg_lat, 2)
            }
            print(f"  {app:30} | Avg Latency: {avg_lat:6.1f} ms | Unreadable/Blank Rate: {fail_rate:5.1f}%")

    os.makedirs("storage/stage8_paddleocr_verification", exist_ok=True)
    with open("storage/stage8_paddleocr_verification/stage8_comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\nFull evaluation artifact saved to storage/stage8_paddleocr_verification/stage8_comparison_results.json")

if __name__ == '__main__':
    main()
