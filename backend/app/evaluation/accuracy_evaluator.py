import os
import cv2
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from backend.app.services.gemini_service import GeminiVisionService
from backend.app.services.preprocessing import ImagePreprocessor
from backend.app.services.alignment import FormAligner

class AccuracyEvaluator:
    def __init__(self, vision_service: GeminiVisionService = None):
        self.vision_service = vision_service or GeminiVisionService()
        self.aligner = FormAligner()

    def render_form_text(self, base_img: np.ndarray, gt: Dict[str, str]) -> np.ndarray:
        img = base_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (15, 15, 120)  # Dark blue handwritten ink simulation

        cv2.putText(img, f"Name: {gt['student_name']}", (300, 750), font, 1.2, color, 3)
        cv2.putText(img, f"School: {gt['school_name']}", (300, 950), font, 1.2, color, 3)
        cv2.putText(img, f"Date: {gt['date']}", (1700, 500), font, 1.1, color, 3)
        cv2.putText(img, f"Student Mobile: {gt['student_mobile_number']}", (300, 1150), font, 1.2, color, 3)
        cv2.putText(img, f"Parent Mobile: {gt['parent_mobile_number']}", (300, 1350), font, 1.2, color, 3)
        cv2.putText(img, f"Email: {gt['email']}", (300, 1550), font, 1.1, color, 3)
        cv2.putText(img, f"Cut Off Marks: {gt['cut_off_marks']}", (300, 1750), font, 1.3, color, 3)
        cv2.putText(img, f"1. Interested in BIT: {gt['interested_in_bit']}", (300, 1950), font, 1.1, color, 3)
        cv2.putText(img, f"2. Career Guidance: {gt['career_guidance_usefulness']}", (300, 2150), font, 1.1, color, 3)
        cv2.putText(img, f"3. Speaker Effectiveness: {gt['speaker_effectiveness']}", (300, 2350), font, 1.1, color, 3)
        cv2.putText(img, f"4. Choice Filling: {gt['interested_in_choice_filling_assistance']}", (300, 2550), font, 1.1, color, 3)
        return img

    def load_test_dataset(self) -> List[Dict[str, Any]]:
        """
        Provides ground truth test dataset for handwritten form extraction evaluation.
        """
        template_path = "bannari_amman_form_template.png"
        if not os.path.exists(template_path):
            base_img = np.full((3508, 2479, 3), 255, dtype=np.uint8)
        else:
            base_img = cv2.imread(template_path)

        gt_01 = {
            "student_name": "B. Thamizhum",
            "school_name": "GHSS ERODE",
            "date": "12/03/2026",
            "student_mobile_number": "9384064560",
            "parent_mobile_number": "9443123456",
            "email": "thamizhum@gmail.com",
            "cut_off_marks": "184.56",
            "interested_in_bit": "yes",
            "career_guidance_usefulness": "excellent",
            "speaker_effectiveness": "good",
            "interested_in_choice_filling_assistance": "yes"
        }

        gt_02 = {
            "student_name": "R. KAVIN",
            "school_name": "MODEL MATRIC HSS",
            "date": "15/02/2026",
            "student_mobile_number": "9876543210",
            "parent_mobile_number": "9123456789",
            "email": "kavin.r@yahoo.com",
            "cut_off_marks": "195",
            "interested_in_bit": "yes",
            "career_guidance_usefulness": "good",
            "speaker_effectiveness": "excellent",
            "interested_in_choice_filling_assistance": "no"
        }

        dataset = [
            {
                "id": "form_sample_01",
                "image": self.render_form_text(base_img, gt_01),
                "ground_truth": gt_01
            },
            {
                "id": "form_sample_02",
                "image": self.render_form_text(base_img, gt_02),
                "ground_truth": gt_02
            }
        ]
        return dataset

    @staticmethod
    def classify_error(field_name: str, expected: Any, predicted: Any) -> Dict[str, Any]:
        """
        Classifies error types and position-level digit confusion (e.g. 3 -> 8, 5 -> 6).
        """
        if predicted is None or str(predicted).strip() == "":
            return {"error_type": "MISSING_VALUE", "details": "Prediction is empty/null"}

        if field_name in ("student_mobile_number", "parent_mobile_number"):
            exp_digits = str(expected).strip()
            pred_digits = str(predicted).strip()

            if len(pred_digits) != len(exp_digits):
                return {
                    "error_type": "DIGIT_COUNT_MISMATCH",
                    "details": f"Expected {len(exp_digits)} digits, got {len(pred_digits)}"
                }

            confusions = []
            for idx, (e_char, p_char) in enumerate(zip(exp_digits, pred_digits)):
                if e_char != p_char:
                    confusions.append({
                        "position": idx + 1,
                        "expected_char": e_char,
                        "predicted_char": p_char,
                        "pattern": f"{e_char} -> {p_char}"
                    })

            return {
                "error_type": "DIGIT_CONFUSION",
                "digit_confusions": confusions
            }

        elif field_name == "cut_off_marks":
            exp_str = str(expected).strip()
            pred_str = str(predicted).strip()
            if "." in exp_str and "." not in pred_str:
                return {"error_type": "DECIMAL_MISSING", "details": "Decimal point was dropped"}
            return {"error_type": "CUTOFF_VALUE_MISMATCH", "details": f"{exp_str} != {pred_str}"}

        return {"error_type": "FIELD_VALUE_MISMATCH", "details": f"{expected} != {predicted}"}

    def evaluate_pipeline(self, dataset: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates extraction accuracy against ground truth.
        Calculates Overall Field Accuracy, 10-Digit Mobile Exact Match, and Cutoff Exact Match.
        """
        dataset = dataset or self.load_test_dataset()

        total_fields = 0
        correct_fields = 0

        mobile_total = 0
        mobile_exact_matches = 0

        cutoff_total = 0
        cutoff_exact_matches = 0

        error_logs = []

        for sample in dataset:
            sample_id = sample["id"]
            img = sample["image"]
            gt = sample["ground_truth"]

            # Preprocess & align form image
            aligned, _, _ = self.aligner.align(img)

            # Vision extraction call
            try:
                extracted, latency = self.vision_service.extract_form_fields(aligned)
            except Exception as e:
                error_logs.append({"sample_id": sample_id, "error": str(e)})
                extracted = {}

            # Evaluate each ground truth field
            for field_name, expected_val in gt.items():
                total_fields += 1
                pred_val = str(extracted.get(field_name, "")).strip().lower() if extracted.get(field_name) is not None else ""
                exp_clean = str(expected_val).strip().lower()

                is_correct = False
                if field_name in ("student_mobile_number", "parent_mobile_number"):
                    mobile_total += 1
                    # 10-digit exact match
                    if pred_val == exp_clean and len(pred_val) == 10:
                        is_correct = True
                        mobile_exact_matches += 1
                    else:
                        error_logs.append({
                            "sample_id": sample_id,
                            "field": field_name,
                            "expected": expected_val,
                            "predicted": extracted.get(field_name),
                            "reason": "MOBILE_EXACT_MATCH_FAILURE"
                        })
                elif field_name == "cut_off_marks":
                    cutoff_total += 1
                    # Cutoff exact match including decimal point
                    if pred_val == exp_clean:
                        is_correct = True
                        cutoff_exact_matches += 1
                    else:
                        error_logs.append({
                            "sample_id": sample_id,
                            "field": field_name,
                            "expected": expected_val,
                            "predicted": extracted.get(field_name),
                            "reason": "CUTOFF_EXACT_MATCH_FAILURE"
                        })
                else:
                    if pred_val == exp_clean or (exp_clean in pred_val and len(exp_clean) > 3):
                        is_correct = True
                    else:
                        error_logs.append({
                            "sample_id": sample_id,
                            "field": field_name,
                            "expected": expected_val,
                            "predicted": extracted.get(field_name),
                            "reason": "FIELD_VALUE_MISMATCH"
                        })

                if is_correct:
                    correct_fields += 1
                else:
                    classification = AccuracyEvaluator.classify_error(field_name, expected_val, extracted.get(field_name))
                    error_logs.append({
                        "sample_id": sample_id,
                        "field": field_name,
                        "expected": expected_val,
                        "predicted": extracted.get(field_name),
                        "classification": classification
                    })

        field_accuracy = (correct_fields / max(1, total_fields)) * 100.0
        mobile_accuracy = (mobile_exact_matches / max(1, mobile_total)) * 100.0
        cutoff_accuracy = (cutoff_exact_matches / max(1, cutoff_total)) * 100.0

        return {
            "total_samples": len(dataset),
            "total_fields": total_fields,
            "correct_fields": correct_fields,
            "field_accuracy_pct": round(field_accuracy, 2),
            "mobile_exact_match_pct": round(mobile_accuracy, 2),
            "cutoff_exact_match_pct": round(cutoff_accuracy, 2),
            "error_logs": error_logs
        }

if __name__ == "__main__":
    evaluator = AccuracyEvaluator()
    results = evaluator.evaluate_pipeline()
    print("--- ACCURACY EVALUATION METRICS ---")
    print(f"Total Fields Evaluated: {results['total_fields']}")
    print(f"Overall Field Accuracy: {results['field_accuracy_pct']}%")
    print(f"Mobile Exact Match Accuracy: {results['mobile_exact_match_pct']}%")
    print(f"Cutoff Exact Match Accuracy: {results['cutoff_exact_match_pct']}%")
    print(f"Error Count: {len(results['error_logs'])}")
