"""
Background Task Orchestrator for Form Processing.

Coordinates the end-to-end feedback digitization pipeline:
1. Job isolation and scratch directory management.
2. Form deskewing and homography alignment.
3. Unified OCR + Bubble Detection pipeline execution (TrOCR, PaddleOCR, OpenCV).
4. Business rule validation & field evidence reconciliation.
5. Database persistence of structured fields, decisions, and audit provenance.
"""

import os
import cv2
import json
import datetime
import numpy as np
from sqlalchemy.orm import Session

from backend.app.workers.celery_app import celery_app
from backend.app.db.session import SessionLocal
from backend.app.models.domain import (
    Document, Page, ExtractionJob, FieldExtraction, GeminiResult, CNNResult, VerificationDecision
)
from backend.app.services.scratch_manager import ScratchManager
from backend.app.services.preprocessing import ImagePreprocessor
from backend.app.services.alignment import FormAligner
from backend.app.services.roi_extractor import ROIExtractor
from backend.app.services.ocr_service import OCRService
from backend.app.decision_engine.field_reconciler import FieldReconciler
from backend.app.validation.validator import FieldValidator

ALL_SCHEMA_FIELDS = [
    "student_name",
    "school_name",
    "date",
    "student_mobile_number",
    "parent_mobile_number",
    "email",
    "cut_off_marks",
    "interested_in_bit",
    "career_guidance_usefulness",
    "speaker_effectiveness",
    "interested_in_choice_filling_assistance"
]


def process_document_pipeline_sync(job_id: str, document_id: int):
    """
    Synchronous OCR Pipeline Runner:
    1. Prepares job-isolated scratch paths under scratch/<job_id>/
    2. Orientation & form alignment (4-marker homography warp)
    3. Modular OCR + TrOCR + OpenCV Bubble Detection extraction
    4. Business rule validation & field reconciliation
    5. Database persistence of field extractions & verification decisions
    """
    db: Session = SessionLocal()
    try:
        job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not job or not doc:
            return

        print(f"[OCR PIPELINE STARTED] job_id={job_id} | document_id={doc.id} | file={doc.filename}")

        # Reset any top-level stale scratch
        ScratchManager.reset_stale_scratch()

        job.status = "RUNNING"
        job.stage = "PREPROCESSING"
        job.progress = 10.0
        db.commit()

        aligner = FormAligner()
        ocr_service = OCRService()
        roi_extractor = ROIExtractor()

        pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()
        success_count = 0
        failed_count = 0

        for idx, page in enumerate(pages):
            try:
                # Query existing records to preserve manual verifications
                existing_vds = {
                    vd.field_name: vd
                    for vd in db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).all()
                }
                existing_fes = {
                    fe.field_name: fe
                    for fe in db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).all()
                }

                # Clear only non-human records for page before reprocessing
                for fn, vd in list(existing_vds.items()):
                    if vd.decision != "HUMAN_VERIFIED" and not vd.reviewer_id:
                        db.delete(vd)
                        existing_vds.pop(fn, None)

                for fn, fe in list(existing_fes.items()):
                    if fe.source != "human" and fe.status != "VERIFIED":
                        db.delete(fe)
                        existing_fes.pop(fn, None)

                db.query(GeminiResult).filter(GeminiResult.page_id == page.id).delete()
                db.commit()

                job.stage = f"PROCESSING_PAGE_{page.page_number}"
                job.progress = 10.0 + (80.0 * (idx / max(1, len(pages))))
                db.commit()

                print(f"[PAGE PROCESSING] page_id={page.id} | page_number={page.page_number} | original_file={page.original_image_path}")

                # 1. Load image & apply EXIF orientation fix
                with open(page.original_image_path, "rb") as f:
                    file_bytes = f.read()
                img_exif_fixed = ImagePreprocessor.load_and_fix_exif(file_bytes)
                ScratchManager.save_image(job_id, "original", f"page_{page.id}_original.png", img_exif_fixed)

                # 2. Deskew
                gray_exif = cv2.cvtColor(img_exif_fixed, cv2.COLOR_BGR2GRAY)
                deskewed_gray, skew_angle = ImagePreprocessor.deskew_image(gray_exif)
                if abs(skew_angle) > 0.5:
                    (h_e, w_e) = img_exif_fixed.shape[:2]
                    M = cv2.getRotationMatrix2D((w_e // 2, h_e // 2), skew_angle, 1.0)
                    img_deskewed = cv2.warpAffine(img_exif_fixed, M, (w_e, h_e), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                else:
                    img_deskewed = img_exif_fixed.copy()

                # 3. Template-Aware Alignment
                form_idx = None
                if "form_2" in page.original_image_path:
                    form_idx = 2
                elif "form_1" in page.original_image_path:
                    form_idx = 1

                from backend.app.services.ocr.template_router import TemplateRouter, TEMPLATE_HANDWRITTEN_A4, UNKNOWN_TEMPLATE
                router = TemplateRouter()
                tmpl_info = router.identify_template(
                    img_deskewed,
                    form_index=form_idx,
                    is_split=("form_" in page.original_image_path)
                )

                if tmpl_info["template_id"] == TEMPLATE_HANDWRITTEN_A4:
                    aligned, H, is_warped = aligner.align(img_deskewed)
                else:
                    # Digital forms or un-warped layouts use calibrated deskewed image directly
                    aligned = img_deskewed
                    H = None
                    is_warped = False

                # Save aligned image
                aligned_path = page.original_image_path.replace(".png", "_aligned.png").replace(".jpg", "_aligned.png")
                cv2.imwrite(aligned_path, aligned)
                page.aligned_image_path = aligned_path

                # Save ROI overlay debug stage to debug/ if A4 template
                if tmpl_info["template_id"] == TEMPLATE_HANDWRITTEN_A4:
                    roi_overlay = ImagePreprocessor.draw_roi_overlay(aligned)
                    ScratchManager.save_image(job_id, "debug", f"page_{page.id}_roi_overlay.png", roi_overlay)

                # 4. OCR Extraction Pipeline (TrOCR + PaddleOCR + OpenCV Bubble Detector)
                job.stage = "OCR_EXTRACTION"
                db.commit()

                print(f"[OCR EXTRACTION] Executing modular OCR pipeline for page_id={page.id} (template={tmpl_info['template_id']})...")
                ocr_output = ocr_service.process_form_image(
                    aligned_image=aligned,
                    form_index=form_idx,
                    is_split=("form_" in page.original_image_path),
                    return_crops=True,
                    return_debug_overlay=True,
                )

                extracted_fields = ocr_output.get("fields", {})

                # Save debug bubble overlay if generated
                debug_bubble_img = ocr_output.get("debug_image")
                if debug_bubble_img is not None:
                    ScratchManager.save_image(job_id, "debug", f"page_{page.id}_bubble_overlay.png", debug_bubble_img)

                # Save field crops to scratch
                crops = ocr_output.get("crops") or {}
                for f_name, c_img in crops.items():
                    if c_img is not None and c_img.size > 0:
                        ScratchManager.save_image(job_id, "crops", f"page_{page.id}_{f_name}.png", c_img)

                # 5. Field-Level Validation & Evidence Reconciliation
                job.stage = "FIELD_RECONCILIATION"
                db.commit()

                reconciled_results = {}
                for schema_field in ALL_SCHEMA_FIELDS:
                    ocr_res = extracted_fields.get(schema_field, {})

                    # If human manually verified this field earlier, DO NOT overwrite!
                    if schema_field in existing_vds and existing_vds[schema_field].decision == "HUMAN_VERIFIED":
                        hum_vd = existing_vds[schema_field]
                        raw_val = ocr_res.get("value")
                        hum_vd.cnn_value = str(raw_val) if raw_val is not None else None
                        reconciled_results[schema_field] = {
                            "final_value": hum_vd.final_value,
                            "status": "HUMAN_VERIFIED",
                            "source": "human",
                            "final_confidence": 1.0,
                            "reasoning": "Preserved manual verification.",
                        }
                        continue

                    # Reconcile field evidence with business validation rules
                    rec = FieldReconciler.reconcile_ocr_field(schema_field, ocr_res)
                    reconciled_results[schema_field] = rec

                    # Store / update in FieldExtraction DB table
                    raw_val = ocr_res.get("value")
                    if schema_field in existing_fes:
                        fe = existing_fes[schema_field]
                        fe.raw_value = str(raw_val) if raw_val is not None else ""
                        fe.normalized_value = rec["final_value"]
                        fe.source = rec["source"]
                        fe.confidence = rec["final_confidence"]
                        fe.status = rec["status"]
                    else:
                        db.add(FieldExtraction(
                            page_id=page.id,
                            field_name=schema_field,
                            raw_value=str(raw_val) if raw_val is not None else "",
                            normalized_value=rec["final_value"],
                            source=rec["source"],
                            confidence=rec["final_confidence"],
                            status=rec["status"],
                        ))

                    # Store / update in VerificationDecision DB table
                    if schema_field in existing_vds:
                        vd = existing_vds[schema_field]
                        vd.gemini_value = str(rec.get("secondary_value")) if rec.get("secondary_value") is not None else None
                        vd.cnn_value = str(raw_val) if raw_val is not None else None
                        vd.final_value = rec["final_value"]
                        vd.decision = rec["status"]
                        vd.confidence = rec["final_confidence"]
                        vd.reason = rec["reasoning"]
                    else:
                        db.add(VerificationDecision(
                            page_id=page.id,
                            field_name=schema_field,
                            gemini_value=str(rec.get("secondary_value")) if rec.get("secondary_value") is not None else None,
                            cnn_value=str(raw_val) if raw_val is not None else None,
                            final_value=rec["final_value"],
                            decision=rec["status"],
                            confidence=rec["final_confidence"],
                            reason=rec["reasoning"],
                        ))

                page.status = "COMPLETED"
                success_count += 1
                print(f"[PAGE COMPLETED] page_id={page.id} | Reconciled fields: {len(reconciled_results)}")

            except Exception as e:
                db.rollback()
                page.status = "FAILED"
                page.error_message = str(e)
                failed_count += 1
                print(f"[PAGE EXTRACTION FAILED] page_id={page.id} | error={e}")

            db.commit()

        job.progress = 100.0
        job.stage = "COMPLETED"
        if failed_count == 0:
            job.status = "COMPLETED"
            doc.status = "COMPLETED"
        elif success_count > 0:
            job.status = "PARTIAL_FAILURE"
            doc.status = "PARTIAL_FAILURE"
        else:
            job.status = "FAILED"
            doc.status = "FAILED"

        doc.completed_at = datetime.datetime.utcnow()
        db.commit()

        # Clean temporary heavy variants while keeping debug logs
        ScratchManager.clean_job_scratch(job_id, keep_debug=True)
        print(f"[JOB COMPLETED] job_id={job_id} | status={job.status}")

    finally:
        db.close()


@celery_app.task(name="tasks.process_document")
def process_document_task(job_id: str, document_id: int):
    process_document_pipeline_sync(job_id, document_id)
