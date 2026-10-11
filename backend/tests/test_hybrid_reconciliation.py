import pytest
import numpy as np
import cv2
import os
from backend.app.decision_engine.field_reconciler import FieldReconciler
from backend.app.services.scratch_manager import ScratchManager
from backend.app.services.ocr_service import OCRService
from backend.app.services.preprocessing import ImagePreprocessor

def test_field_reconciler_corroborated_mobile():
    qwen_val = "9432164321"
    ocr_res = {
        "consensus_text": "9432164321",
        "ocr_confidence": 0.92,
        "variant_agreement": 1.0,
        "status": "VALID"
    }
    rec = FieldReconciler.reconcile_field("student_mobile_number", qwen_val, ocr_res)
    assert rec["status"] == "CORROBORATED"
    assert rec["final_value"] == "9432164321"
    assert rec["final_confidence"] == 0.98
    assert rec["source"] == "ocr+vlm"

def test_field_reconciler_ocr_override_mobile():
    # Qwen hallucinated 9 digits or invalid string, OCR has high consensus 10-digit number
    qwen_val = "943216432"  # Only 9 digits (invalid)
    ocr_res = {
        "consensus_text": "9432164321",  # Valid 10 digits
        "ocr_confidence": 0.88,
        "variant_agreement": 0.75,
        "status": "VALID"
    }
    rec = FieldReconciler.reconcile_field("student_mobile_number", qwen_val, ocr_res)
    assert rec["status"] == "OCR_OVERRIDE"
    assert rec["final_value"] == "9432164321"
    assert rec["final_confidence"] >= 0.90

def test_field_reconciler_vlm_selected_when_ocr_empty():
    qwen_val = "KAVIN K"
    ocr_res = {
        "consensus_text": "",
        "ocr_confidence": 0.0,
        "variant_agreement": 0.0,
        "status": "EMPTY"
    }
    rec = FieldReconciler.reconcile_field("student_name", qwen_val, ocr_res)
    assert rec["status"] == "VLM_SELECTED"
    assert rec["final_value"] == "KAVIN K"
    assert rec["source"] == "qwen_vlm"

def test_field_reconciler_non_hallucination_blank():
    rec = FieldReconciler.reconcile_field("student_name", None, None)
    assert rec["status"] == "BLANK"
    assert rec["final_value"] is None
    assert rec["final_confidence"] == 0.0
    assert "blank" in rec["reasoning"].lower()

def test_field_reconciler_cutoff_corroboration():
    qwen_val = "184.56"
    ocr_res = {
        "consensus_text": "184.56",
        "ocr_confidence": 0.95,
        "variant_agreement": 0.75,
        "status": "VALID"
    }
    rec = FieldReconciler.reconcile_field("cut_off_marks", qwen_val, ocr_res)
    assert rec["status"] == "CORROBORATED"
    assert rec["final_value"] == "184.56"
    assert rec["final_confidence"] == 0.98

def test_scratch_manager_job_isolation(tmp_path):
    job_id = "test_job_123"
    job_dir = ScratchManager.get_job_dir(job_id, "crops")
    assert os.path.exists(job_dir)
    assert "crops" in job_dir

    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
    saved_path = ScratchManager.save_image(job_id, "crops", "test_crop.png", dummy_img)
    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "test_crop.png"

def test_multi_variant_preprocessing():
    img = np.zeros((100, 100, 3), dtype=np.uint8) + 128
    variants = ImagePreprocessor.generate_variants(img)
    assert "original" in variants
    assert "grayscale" in variants
    assert "enhanced" in variants
    assert "thresholded" in variants
    assert "sharpened" in variants
    assert "deskewed" in variants
    assert "high_contrast" in variants
    assert "denoised" in variants

def test_field_reconciler_ocr_typo_vlm_wins():
    # VLM="Computer Science", OCR="Computer Scienc" -> VLM selected due to complete string
    qwen_val = "Computer Science"
    ocr_res = {
        "consensus_text": "Computer Scienc",
        "ocr_confidence": 0.72,
        "variant_agreement": 0.50,
        "status": "VALID"
    }
    rec = FieldReconciler.reconcile_field("school_name", qwen_val, ocr_res)
    assert rec["final_value"] == "Computer Science"
    assert rec["source"] in ("qwen_vlm", "ocr+vlm")

def test_field_reconciler_date_ocr_override():
    qwen_val = "2024"  # Incomplete / uncertain
    ocr_res = {
        "consensus_text": "15/09/2021",
        "ocr_confidence": 0.95,
        "variant_agreement": 1.0,
        "status": "VALID"
    }
    rec = FieldReconciler.reconcile_field("date", qwen_val, ocr_res)
    assert rec["status"] == "OCR_OVERRIDE"
    assert rec["final_value"] == "15/09/2021"

def test_crop_validation_and_on_demand_extraction():
    from backend.app.services.roi_extractor import ROIExtractor
    extractor = ROIExtractor()
    canvas = np.zeros((1000, 800, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (300, 300), (255, 255, 255), -1)

    crop, meta = extractor.crop_single_field(canvas, "student_mobile_number", page_num=1)
    assert meta["field"] == "student_mobile_number"
    assert meta["page"] == 1
    assert "is_valid" in meta

def test_map_bbox_norm_resolution_invariance():
    # Test coordinate mapping across resolutions with matching aspect ratio allowing 1px rounding tolerance
    x1_2k, y1_2k, x2_2k, y2_2k = ImagePreprocessor.map_bbox_norm((0.1, 0.2, 0.5, 0.6), 2560, 1808)
    x1_1k, y1_1k, x2_1k, y2_1k = ImagePreprocessor.map_bbox_norm((0.1, 0.2, 0.5, 0.6), 1280, 904)
    assert abs(x1_2k - 2 * x1_1k) <= 1
    assert abs(y1_2k - 2 * y1_1k) <= 1

