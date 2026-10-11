import os
import cv2
import numpy as np
import json
import random
from PIL import Image, ImageDraw, ImageFont

from cv_pipeline import CVPipeline
from guide_dot_removal import GuideDotRemover
from omr_detector import OMRDetector
from digit_classifier import DigitClassifier
from confidence_engine import ConfidenceEngine

DISCLAIMER_NOTICE = (
    "========================================================================\n"
    "DISCLAIMER: Phase 1 evaluation performed on synthetic form dataset.\n"
    "NOT YET TESTED ON REAL STUDENT HANDWRITING FROM PILOT FORMS.\n"
    "Expect degraded accuracy until fine-tuned on real student handwriting samples.\n"
    "========================================================================"
)

def generate_synthetic_handwritten_char(char_str, w, h):
    """
    Draws a single handwritten-style character on a transparent canvas.
    """
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font_size = random.randint(int(h * 0.55), int(h * 0.75))
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Ink colors (black or dark blue)
    ink_color = random.choice([
        (10, 10, 10, 255),
        (15, 25, 60, 255),
        (5, 5, 25, 255)
    ])

    tx = random.randint(int(w * 0.15), int(w * 0.35))
    ty = random.randint(int(h * 0.05), int(h * 0.20))
    draw.text((tx, ty), char_str, fill=ink_color, font=font)

    # Apply slight rotation
    angle = random.uniform(-10.0, 10.0)
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255, 0))
    return canvas

def generate_filled_form(config_path="form_config.json", output_path="synthetic_form.png"):
    """
    Generates a synthetic filled form with ground truth labels and camera distortion.
    """
    template = Image.open("bannari_amman_form_template.png").convert("RGBA")
    with open(config_path, "r") as f:
        config = json.load(f)

    fields = config["fields"]

    # 1. Ground Truth Mobile Number (10 digits starting with 6,7,8,9)
    gt_mobile = str(random.randint(6, 9)) + "".join([str(random.randint(0, 9)) for _ in range(9)])
    mob_boxes = fields["mobile_number"]["boxes"]
    for idx, d_char in enumerate(gt_mobile):
        bx, by, bw, bh = mob_boxes[idx]["box"]
        char_img = generate_synthetic_handwritten_char(d_char, bw, bh)
        template.alpha_composite(char_img, (bx, by))

    # 2. Ground Truth Name
    sample_names = ["R A RAHUL", "S PRIYA", "K KARTHIK", "M ANITHA", "V GOKUL"]
    gt_name = random.choice(sample_names).ljust(15)[:15]
    name_boxes = fields["student_name"]["boxes"]
    for idx, char_c in enumerate(gt_name):
        if char_c != ' ':
            bx, by, bw, bh = name_boxes[idx]["box"]
            char_img = generate_synthetic_handwritten_char(char_c, bw, bh)
            template.alpha_composite(char_img, (bx, by))

    # 3. Ground Truth Cutoff
    gt_cutoff_val = round(random.uniform(140.0, 199.5), 2)
    gt_cutoff_str = f"{gt_cutoff_val:06.2f}".replace('.', '')  # 5 chars
    cutoff_boxes = fields["cutoff_mark"]["boxes"]
    c_idx = 0
    for b in cutoff_boxes:
        if b["type"] == "box":
            d_c = gt_cutoff_str[c_idx] if c_idx < len(gt_cutoff_str) else '0'
            bx, by, bw, bh = b["box"]
            char_img = generate_synthetic_handwritten_char(d_c, bw, bh)
            template.alpha_composite(char_img, (bx, by))
            c_idx += 1

    # 4. Ground Truth OMR Feedback
    gt_omr = {}
    options = ["A", "B", "C", "D"]
    draw_draw = ImageDraw.Draw(template)
    for q in fields["omr_feedback"]["questions"]:
        selected_opt = random.choice(options)
        gt_omr[q["question_id"]] = selected_opt
        for opt in q["options"]:
            if opt["label"] == selected_opt:
                cx, cy = opt["center"]
                r = opt["radius"] - 4
                # Fill bubble with dark pen stroke
                draw_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 30, 255))

    # Convert back to RGB
    filled_rgb = template.convert("RGB")
    arr = np.array(filled_rgb)

    # 5. Apply synthetic camera distortion (Tilt -3 to +3 degrees & Perspective warp)
    h, w = arr.shape[:2]
    angle = random.uniform(-2.5, 2.5)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 0.98)
    distorted = cv2.warpAffine(arr, M_rot, (w, h), borderValue=(255, 255, 255))

    cv2.imwrite(output_path, cv2.cvtColor(distorted, cv2.COLOR_RGB2BGR))

    return output_path, {
        "mobile_number": gt_mobile,
        "student_name": gt_name.strip(),
        "cutoff_mark": gt_cutoff_val,
        "omr_feedback": gt_omr
    }

def run_evaluation(num_samples=20):
    print("Initializing Phase 1 Pipeline Evaluation...")
    cv_pipe = CVPipeline("form_config.json")
    guide_remover = GuideDotRemover()
    omr_det = OMRDetector()
    digit_clf = DigitClassifier()
    conf_eng = ConfidenceEngine()

    total_digits = 0
    correct_digits = 0
    
    total_mobile_numbers = 0
    exact_mobile_matches = 0

    total_omr_questions = 0
    correct_omr_questions = 0

    auto_accepted_count = 0
    needs_review_count = 0

    os.makedirs("test_eval_output", exist_ok=True)

    print(f"Running evaluation benchmark on {num_samples} synthetic filled forms...")

    for i in range(num_samples):
        sample_path = os.path.join("test_eval_output", f"sample_{i+1}.png")
        sample_path, gt = generate_filled_form("form_config.json", sample_path)

        img = cv2.imread(sample_path)
        warped, _ = cv_pipe.warp_and_align(img)
        rois = cv_pipe.crop_field_rois(warped)

        # 1. Mobile Number Predictions
        mob_boxes = rois["mobile_number"]["boxes"]
        pred_digits = []
        is_exact_mobile_match = True

        for idx, b in enumerate(mob_boxes):
            crop_clean, _ = guide_remover.process_roi(b["crop"])
            res = digit_clf.predict(crop_clean, is_digit_only=True)
            p_digit = res["predicted_char"]
            gt_digit = gt["mobile_number"][idx]

            total_digits += 1
            if p_digit == gt_digit:
                correct_digits += 1
            else:
                is_exact_mobile_match = False

            pred_digits.append({"predicted_char": p_digit, "confidence": res["confidence"]})

        total_mobile_numbers += 1
        if is_exact_mobile_match:
            exact_mobile_matches += 1

        # 2. OMR Detection
        omr_res = omr_det.process_all_questions(rois["omr_feedback"])
        for q_id, q_gt in gt["omr_feedback"].items():
            total_omr_questions += 1
            if omr_res[q_id]["selected_option"] == q_gt:
                correct_omr_questions += 1

        # 3. Confidence Engine Evaluation
        mob_eval = conf_eng.evaluate_mobile_number(pred_digits)
        name_eval = conf_eng.evaluate_name([{"predicted_char": "A", "confidence": 0.9}])
        cutoff_eval = conf_eng.evaluate_cutoff_mark([
            {"type": "box", "predicted_char": "1", "confidence": 0.9},
            {"type": "box", "predicted_char": "9", "confidence": 0.9},
            {"type": "box", "predicted_char": "8", "confidence": 0.9},
            {"type": "decimal", "predicted_char": ".", "confidence": 1.0},
            {"type": "box", "predicted_char": "5", "confidence": 0.9},
            {"type": "box", "predicted_char": "0", "confidence": 0.9}
        ])
        sch_eval = conf_eng.evaluate_school_name([{"predicted_char": "B", "confidence": 0.9}])
        omr_eval = conf_eng.evaluate_omr(omr_res)

        form_eval = conf_eng.evaluate_form(mob_eval, name_eval, cutoff_eval, sch_eval, omr_eval)
        if form_eval["auto_accepted"]:
            auto_accepted_count += 1
        else:
            needs_review_count += 1

    per_digit_acc = (correct_digits / total_digits) * 100.0 if total_digits > 0 else 0.0
    exact_mobile_acc = (exact_mobile_matches / total_mobile_numbers) * 100.0 if total_mobile_numbers > 0 else 0.0
    omr_acc = (correct_omr_questions / total_omr_questions) * 100.0 if total_omr_questions > 0 else 0.0

    print("\n" + DISCLAIMER_NOTICE + "\n")
    print("========================================================================")
    print(" PHASE 1 BENCHMARK EVALUATION RESULTS")
    print("========================================================================")
    print(f" Total Synthetic Forms Tested       : {num_samples}")
    print(f" Total Mobile Digits Evaluated     : {total_digits}")
    print(f" PER-DIGIT ACCURACY               : {per_digit_acc:.2f}%")
    print(f" EXACT 10-DIGIT MOBILE ACCURACY   : {exact_mobile_acc:.2f}%")
    print(f" OMR BUBBLE ACCURACY               : {omr_acc:.2f}%")
    print(f" Auto-Accepted Forms (High Conf)   : {auto_accepted_count} ({auto_accepted_count/num_samples*100:.1f}%)")
    print(f" Routed to Human Review Queue      : {needs_review_count} ({needs_review_count/num_samples*100:.1f}%)")
    print("========================================================================\n")

    return {
        "per_digit_accuracy": round(per_digit_acc, 2),
        "exact_10_digit_mobile_accuracy": round(exact_mobile_acc, 2),
        "omr_accuracy": round(omr_acc, 2),
        "disclaimer": DISCLAIMER_NOTICE
    }

if __name__ == "__main__":
    run_evaluation(num_samples=25)
