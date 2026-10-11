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
    if not text or len(text.strip()) == 0:
        return "***"
    t = text.strip()
    if len(t) <= 4:
        return "***"
    return t[:2] + "***" + t[-2:]

def run_trocr(proc, model, img_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(rgb)
    pixel_values = proc(pil_img, return_tensors='pt').pixel_values
    gen_ids = model.generate(pixel_values, max_new_tokens=32)
    return proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

def main():
    print("Loading TrOCR model...")
    proc = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    aligner = FormAligner()

    print("\n=== COMPARING LABEL-INCLUDED VS LABEL-EXCLUDED WRITE-IN REGIONS ===")
    
    for i in range(1, 6):
        fname = f"form{i}.jpeg"
        fpath = f"cnn_dataset/{fname}"
        raw = cv2.imread(fpath)
        aligned, _, _ = aligner.align(raw)
        
        # 1. Student Name
        # Full crop (includes '1. Name of the Student :')
        name_full = aligned[641:747, 202:1239]
        # Writein only (skips label from x=0 to 710)
        name_writein = aligned[641:747, 720:1300]
        
        # 2. Email
        # Full crop (includes '3. E-mail :')
        email_full = aligned[891:996, 202:1183]
        # Writein only
        email_writein = aligned[891:996, 600:1300]
        
        # 3. Date
        # Full crop
        date_full = aligned[501:639, 202:1183]
        # Box cells only (x from 600 to 1180)
        date_boxes_only = aligned[501:639, 600:1183]
        
        pred_name_full = run_trocr(proc, model, name_full)
        pred_name_writein = run_trocr(proc, model, name_writein)
        
        pred_email_full = run_trocr(proc, model, email_full)
        pred_email_writein = run_trocr(proc, model, email_writein)
        
        pred_date_full = run_trocr(proc, model, date_full)
        pred_date_boxes = run_trocr(proc, model, date_boxes_only)
        
        print(f"\n--- {fname} ---")
        print(f"  Name (Full with Label):    '{mask_pii(pred_name_full)}'")
        print(f"  Name (Write-in Only):       '{mask_pii(pred_name_writein)}'")
        print(f"  Email (Full with Label):   '{mask_pii(pred_email_full)}'")
        print(f"  Email (Write-in Only):      '{mask_pii(pred_email_writein)}'")
        print(f"  Date (Full with Label):    '{mask_pii(pred_date_full)}'")
        print(f"  Date (Boxes Only):         '{mask_pii(pred_date_boxes)}'")

if __name__ == '__main__':
    main()
