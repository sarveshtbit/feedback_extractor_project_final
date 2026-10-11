import os
import cv2
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from backend.app.services.alignment import FormAligner
from backend.app.services.preprocessing import ImagePreprocessor
from backend.app.ml.inference.cnn_service import CNNInferenceService
from backend.app.decision_engine.comparator import DecisionEngine
from backend.app.core.config import get_form_template_config

def run_pipeline_test():
    print("==================================================")
    print("END-TO-END VERIFICATION PIPELINE TEST")
    print("==================================================")

    input_dir = "cnn_dataset"
    gt_path = os.path.join(input_dir, "ground_truth.xlsx")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(input_dir, "ground_truth.csv")
    
    df_gt = pd.read_excel(gt_path) if gt_path.endswith(".xlsx") else pd.read_csv(gt_path)
    
    cfg = get_form_template_config()
    student_mobile_cfg = cfg["fields"]["student_mobile"]
    boxes_cfg = student_mobile_cfg["boxes"]
    roi_bbox_norm = student_mobile_cfg["normalized_bbox"]
    
    aligner = FormAligner(template_config=cfg)
    cnn_service = CNNInferenceService(model_path="ml/models/pretrained_resnet18.pth")

    print(f"CNN Model Ready: {cnn_service.is_ready()}")
    print("-" * 75)

    results = []

    for idx, row in df_gt.iterrows():
        fname = str(row["filename"]).strip()
        gt_mobile = str(row["student_mobile"]).strip()
        form_path = os.path.join(input_dir, fname)

        with open(form_path, "rb") as f:
            file_bytes = f.read()

        # STAGE 1: FORM & FIELD LOCALIZATION (Mechanism 1)
        raw_img = ImagePreprocessor.load_and_fix_exif(file_bytes)
        aligned_img, H, is_warped = aligner.align(raw_img)
        h_orig, w_orig = raw_img.shape[:2]

        rx1, ry1, rx2, ry2 = ImagePreprocessor.map_bbox_norm(roi_bbox_norm, w_orig, h_orig)
        roi = raw_img[ry1:ry2, rx1:rx2]

        box_crops = []
        valid_boxes = True

        for b_idx, box_item in enumerate(boxes_cfg):
            bx1, by1, bx2, by2 = ImagePreprocessor.map_bbox_norm(box_item["normalized_bbox"], w_orig, h_orig)
            crop = raw_img[by1:by2, bx1:bx2]
            if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                valid_boxes = False
                break
            box_crops.append({
                "index": b_idx + 1,
                "crop": crop
            })

        if len(box_crops) != 10 or not valid_boxes:
            loc_status = "LOCALIZATION_FAILED"
            cnn_res = {
                "value": None,
                "confidence": 0.0,
                "per_character_results": [],
                "status": "LOCALIZATION_FAILED",
                "notice": "Field localization failed to extract 10 valid boxes."
            }
        else:
            loc_status = "LOCALIZATION_SUCCESS"
            # STAGE 2: CNN RECOGNITION (Mechanism 2)
            cnn_res = cnn_service.predict_phone_number(box_crops)

        # STAGE 3: QWEN + CNN DECISION ENGINE COMPARISON (Mechanism 3)
        qwen_simulated = gt_mobile
        decision = DecisionEngine.evaluate_phone_number(qwen_simulated, cnn_res)

        results.append({
            "filename": fname,
            "ground_truth": gt_mobile,
            "localization": loc_status,
            "cnn_prediction": cnn_res.get("value"),
            "cnn_confidence": cnn_res.get("confidence"),
            "decision": decision.get("decision"),
            "final_value": decision.get("final_value"),
            "confidence": decision.get("confidence")
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline_test()
