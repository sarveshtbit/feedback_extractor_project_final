import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.modules.setdefault('sklearn', None)

import cv2
import yaml
import json
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from backend.app.services.alignment import FormAligner

def mask_pii(text: str) -> str:
    if not text:
        return "***"
    t = text.strip()
    if len(t) <= 4:
        return "***"
    return t[:2] + "***" + t[-2:]

def main():
    print("Loading TrOCR for isolated box test...")
    proc = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    aligner = FormAligner()
    with open('config/form_template.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    gt_mobile = {
        "form1.jpeg": "9432164321",
        "form2.jpeg": "9042642526",
        "form3.jpeg": "8088997798",
        "form4.jpeg": "9754608192",
        "form5.jpeg": "8304934091",
    }
    
    boxes_cfg = cfg['fields']['student_mobile']['boxes']
    
    results = {}
    
    for fname, gt in gt_mobile.items():
        fpath = f"cnn_dataset/{fname}"
        raw_img = cv2.imread(fpath)
        if raw_img is None:
            continue
        aligned, _, _ = aligner.align(raw_img)
        h, w = aligned.shape[:2]
        
        extracted_digits = []
        
        for b in boxes_cfg:
            b_norm = b['normalized_bbox']
            bx1 = int(b_norm[0] * w)
            by1 = int(b_norm[1] * h)
            bx2 = int(b_norm[2] * w)
            by2 = int(b_norm[3] * h)
            
            # Inset 8% to remove outer box border
            bw = bx2 - bx1
            bh = by2 - by1
            pad_x = int(bw * 0.08)
            pad_y = int(bh * 0.08)
            
            box_crop = aligned[by1 + pad_y : by2 - pad_y, bx1 + pad_x : bx2 - pad_x]
            
            # Run TrOCR on isolated box
            rgb = cv2.cvtColor(box_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pixel_values = proc(pil_img, return_tensors='pt').pixel_values
            gen_ids = model.generate(pixel_values, max_new_tokens=4)
            char_pred = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            # extract first digit or char
            extracted_digits.append(char_pred if char_pred else "")
            
        full_pred = "".join([c[0] if c else " " for c in extracted_digits])
        results[fname] = {
            "ground_truth": gt,
            "raw_box_preds": extracted_digits,
            "assembled_string": full_pred,
            "masked_gt": mask_pii(gt),
            "masked_pred": mask_pii(full_pred),
        }
        print(f"[{fname}] GT: {mask_pii(gt)} | Assembled Box Pred: '{mask_pii(full_pred)}' | Raw Boxes: {extracted_digits}")
        
    os.makedirs('storage/stage7_investigation', exist_ok=True)
    with open('storage/stage7_investigation/isolated_box_trocr_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
