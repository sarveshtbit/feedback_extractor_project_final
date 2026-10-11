import os
import sys
import glob

# Set python path and disable sklearn import
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

def enhance_contrast(crop: np.ndarray) -> np.ndarray:
    """Applies CLAHE on L-channel in LAB color space."""
    if len(crop.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(crop)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def tight_crop_handwriting(crop: np.ndarray, pad: int = 5) -> np.ndarray:
    """Finds bounding box of dark ink in crop and tightens it."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    # Otsu or adaptive threshold to find ink
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find non-zero points
    pts = cv2.findNonZero(thresh)
    if pts is None or len(pts) < 20:
        return crop
    x, y, w, h = cv2.boundingRect(pts)
    # Pad
    h_c, w_c = crop.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_c, x + w + pad)
    y2 = min(h_c, y + h + pad)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return crop
    return crop[y1:y2, x1:x2]

def remove_grid_lines(crop: np.ndarray) -> np.ndarray:
    """Morphological line detection and inpainting / background filling."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    # Binarize
    bw = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, crop.shape[0] // 3)))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, crop.shape[1] // 5), 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

    grid_mask = cv2.add(v_lines, h_lines)
    # Dilate grid mask slightly
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    grid_mask = cv2.dilate(grid_mask, dilate_kernel, iterations=1)

    # Inpaint grid lines using fast marching
    res = cv2.inpaint(crop, grid_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return res

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
    with open('config/form_template.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Load ground truth if exists
    gt_map = {}
    if os.path.exists('cnn_dataset/ground_truth.csv'):
        import csv
        with open('cnn_dataset/ground_truth.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize keys
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
                fn = clean_row.get('filename')
                if fn:
                    gt_map[fn] = clean_row

    form_files = [
        'bannari_amman_form_template.png',
        'cnn_dataset/form1.jpeg',
        'cnn_dataset/form2.jpeg',
        'cnn_dataset/form3.jpeg',
        'cnn_dataset/form4.jpeg',
        'cnn_dataset/form5.jpeg',
        'debug_original_forms/page_1_form_1.png',
        'debug_original_forms/page_1_form_2.png',
    ]

    out_dir = 'storage/stage7_investigation'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/crops', exist_ok=True)

    fields_to_test = ['student_name', 'school_name', 'date', 'student_mobile', 'parent_mobile', 'email', 'cutoff_marks']

    results = []

    for fpath in form_files:
        if not os.path.exists(fpath):
            continue
        fname = os.path.basename(fpath)
        raw_img = cv2.imread(fpath)
        if raw_img is None:
            continue
        aligned, _, _ = aligner.align(raw_img)
        h, w = aligned.shape[:2]

        form_res = {"file": fpath, "filename": fname, "fields": {}}

        print(f"\n================ Processing {fname} ================")

        for f_name in fields_to_test:
            f_cfg = cfg['fields'].get(f_name)
            if not f_cfg:
                continue
            bbox = f_cfg['normalized_bbox']
            x1 = max(0, int(bbox[0] * w))
            y1 = max(0, int(bbox[1] * h))
            x2 = min(w, int(bbox[2] * w))
            y2 = min(h, int(bbox[3] * h))

            orig_crop = aligned[y1:y2, x1:x2]
            if orig_crop.size == 0:
                continue

            base_crop_name = f"{fname.replace('.', '_')}_{f_name}"
            cv2.imwrite(f"{out_dir}/crops/{base_crop_name}_orig.png", orig_crop)

            # Generate variants
            tight_crop = tight_crop_handwriting(orig_crop)
            contrast_crop = enhance_contrast(orig_crop)
            grid_removed_crop = remove_grid_lines(orig_crop)
            contrast_grid_crop = enhance_contrast(grid_removed_crop)

            cv2.imwrite(f"{out_dir}/crops/{base_crop_name}_tight.png", tight_crop)
            cv2.imwrite(f"{out_dir}/crops/{base_crop_name}_contrast.png", contrast_crop)
            cv2.imwrite(f"{out_dir}/crops/{base_crop_name}_grid_removed.png", grid_removed_crop)

            # Run inference on all variants
            pred_orig = run_trocr(proc, model, orig_crop)
            pred_tight = run_trocr(proc, model, tight_crop)
            pred_contrast = run_trocr(proc, model, contrast_crop)
            pred_grid_rem = run_trocr(proc, model, grid_removed_crop)
            pred_contrast_grid = run_trocr(proc, model, contrast_grid_crop)

            # Check ground truth if available
            gt_val = None
            if fname in gt_map and f_name in gt_map[fname]:
                gt_val = gt_map[fname][f_name]

            field_entry = {
                "bbox": [x1, y1, x2, y2],
                "ground_truth": mask_pii(gt_val) if gt_val else None,
                "variants": {
                    "original": {
                        "raw_text": pred_orig,
                        "masked_text": mask_pii(pred_orig),
                    },
                    "tight_crop": {
                        "raw_text": pred_tight,
                        "masked_text": mask_pii(pred_tight),
                    },
                    "contrast_enhanced": {
                        "raw_text": pred_contrast,
                        "masked_text": mask_pii(pred_contrast),
                    },
                    "grid_removed": {
                        "raw_text": pred_grid_rem,
                        "masked_text": mask_pii(pred_grid_rem),
                    },
                    "contrast_plus_grid_removed": {
                        "raw_text": pred_contrast_grid,
                        "masked_text": mask_pii(pred_contrast_grid),
                    }
                }
            }

            form_res["fields"][f_name] = field_entry
            print(f"[{f_name}]")
            print(f"  Orig:          '{mask_pii(pred_orig)}'")
            print(f"  Tight:         '{mask_pii(pred_tight)}'")
            print(f"  Contrast:      '{mask_pii(pred_contrast)}'")
            print(f"  Grid-Removed:  '{mask_pii(pred_grid_rem)}'")
            print(f"  Contrast+Grid: '{mask_pii(pred_contrast_grid)}'")
            if gt_val:
                print(f"  Ground Truth:  '{mask_pii(gt_val)}'")

        results.append(form_res)

    with open(f"{out_dir}/stage7_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved all results to {out_dir}/stage7_experiment_results.json")

if __name__ == '__main__':
    main()
