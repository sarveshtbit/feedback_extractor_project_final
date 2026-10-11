import os
import cv2
import numpy as np
import json

def analyze_bubbles_for_form(form_img_path, form_num):
    img = cv2.imread(form_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Adaptive threshold to find dark filled ink
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Question 1: interested_in_bit (Yes / No)
    # On Form 1: y=990..1050; On Form 2: y=970..1030 (relative to form start)
    # Search Yes and No regions
    # "Yes": x=1300..1400, "No": x=1450..1550
    # Let's locate the row containing "Are you interested in joining BIT"
    # In Form 1, y ~ 1010; in Form 2, y ~ 970
    
    # We can detect circles using HoughCircles in the question region:
    # Q1: y in [960, 1060]
    # Q2: y in [1120, 1240]
    # Q3: y in [1350, 1470]
    # Q4: y in [1450, 1550]
    
    results = {}
    
    # Helper to measure fill ratio inside a circle bbox
    def get_fill_ratio(cx, cy, r=16):
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        patch = binary[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        # Create circular mask
        mask = np.zeros_like(patch)
        cv2.circle(mask, (r, r), r - 2, 255, -1)
        filled_pixels = np.sum((patch == 255) & (mask == 255))
        total_circle_pixels = np.sum(mask == 255)
        return float(filled_pixels) / float(total_circle_pixels) if total_circle_pixels > 0 else 0.0

    # Let's define the candidate bubble centers based on our visual coordinates
    # For Form 1:
    if form_num == 1:
        q1_boxes = {"Yes": (1338, 1018), "No": (1456, 1018)}
        q2_boxes = {
            "5": (336, 1195),
            "4": (786, 1195),
            "3": (1240, 1195),
            "2": (1688, 1195),
            "1": (2140, 1195)
        }
        q3_boxes = {
            "5": (336, 1425),
            "4": (786, 1425),
            "3": (1240, 1425),
            "2": (1688, 1425),
            "1": (2140, 1425)
        }
        q4_boxes = {"Yes": (1538, 1505), "No": (1658, 1505)}
    else:
        # Form 2: relative coordinates from PaddleOCR detection
        q1_boxes = {"Yes": (1340, 1098), "No": (1458, 1098)}
        q2_boxes = {
            "5": (336, 1272),
            "4": (786, 1272),
            "3": (1240, 1272),
            "2": (1688, 1272),
            "1": (2140, 1272)
        }
        q3_boxes = {
            "5": (336, 1502),
            "4": (786, 1502),
            "3": (1240, 1502),
            "2": (1688, 1502),
            "1": (2140, 1502)
        }
        q4_boxes = {"Yes": (1538, 1585), "No": (1662, 1585)}

    def evaluate_group(group_dict):
        scores = {}
        for opt, (cx, cy) in group_dict.items():
            # Fine-tune center by finding local centroid in 40x40 patch
            patch_gray = gray[cy-20:cy+20, cx-20:cx+20]
            # measure density
            fr = get_fill_ratio(cx, cy, r=14)
            scores[opt] = fr
        best_opt = max(scores, key=scores.get)
        return best_opt, scores

    best_q1, s_q1 = evaluate_group(q1_boxes)
    best_q2, s_q2 = evaluate_group(q2_boxes)
    best_q3, s_q3 = evaluate_group(q3_boxes)
    best_q4, s_q4 = evaluate_group(q4_boxes)

    print(f"=== FORM {form_num} BUBBLE DETECTION ===")
    print(f"Q1 (interested_in_bit): Best='{best_q1}', Scores={s_q1}")
    print(f"Q2 (programme_usefulness): Best='{best_q2}', Scores={s_q2}")
    print(f"Q3 (speaker_effectiveness): Best='{best_q3}', Scores={s_q3}")
    print(f"Q4 (choice_filling_assistance): Best='{best_q4}', Scores={s_q4}")
    
    return {
        "interested_in_bit": {"detected": best_q1, "scores": s_q1},
        "programme_usefulness": {"detected": best_q2, "scores": s_q2},
        "speaker_effectiveness": {"detected": best_q3, "scores": s_q3},
        "choice_filling_assistance": {"detected": best_q4, "scores": s_q4}
    }

f1_res = analyze_bubbles_for_form('storage/stage11_dummy_pdf_verification/separated_form_1.png', 1)
f2_res = analyze_bubbles_for_form('storage/stage11_dummy_pdf_verification/separated_form_2.png', 2)

os.makedirs('storage/stage11_dummy_pdf_verification', exist_ok=True)
with open('storage/stage11_dummy_pdf_verification/bubble_detection_results.json', 'w') as f:
    json.dump({"form_1": f1_res, "form_2": f2_res}, f, indent=2)
print("Saved bubble_detection_results.json successfully.")
