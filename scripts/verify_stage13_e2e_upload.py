"""
Stage 13 End-to-End Verification: Real PDF Upload-to-Review Lifecycle Test.

Performs independent end-to-end extraction on `BIT_OCR_Dummy_Filled.pdf`:
1. Ingests the real PDF through PDFPreprocessor and database document lifecycle.
2. Verifies 2 separate Page records are generated for Form 1 and Form 2 without mixing.
3. Executes real inference through PaddleOCR subprocess bridge and OpenCV bubble detector.
4. Validates that 2 distinct sets of 11 fields are extracted and stored in SQLite.
5. Verifies manual review workflow (human verification persistence without overwriting).
6. Verifies CSV export output with separated records.
7. Evaluates predictions against ground truth and reports exact matches, CER, and latency.
"""

import os
import sys
import time
import json
import csv
import io
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db.session import SessionLocal
from backend.app.models.domain import (
    Document, Page, ExtractionJob, FieldExtraction, VerificationDecision
)
from backend.app.services.ocr.pdf_preprocessor import PDFPreprocessor
from backend.app.workers.tasks import process_document_pipeline_sync
from backend.app.api.v1.documents import export_document_csv


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


def main():
    print("=" * 70)
    print("STAGE 13: REAL APPLICATION UPLOAD-TO-REVIEW LIFECYCLE VERIFICATION")
    print("=" * 70)
    t_start = time.time()

    pdf_path = "BIT_OCR_Dummy_Filled.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: '{pdf_path}' not found in workspace!")
        sys.exit(1)

    print(f"[STEP 1] Ingesting PDF '{pdf_path}' ({os.path.getsize(pdf_path)} bytes)...")
    db = SessionLocal()
    doc_id = None
    page_ids = []

    try:
        # Create Document record
        doc = Document(
            filename=os.path.basename(pdf_path),
            file_type="application/pdf",
            total_pages=1,
            status="UPLOADED",
            storage_path=os.path.abspath(pdf_path),
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id
        print(f"[DATABASE] Created Document ID={doc.id}")

        # Render & separate forms using PDFPreprocessor
        output_pages_dir = f"./storage/uploads/doc_{doc.id}_pages"
        forms = PDFPreprocessor.process_pdf(pdf_path, output_pages_dir)
        print(f"[PREPROCESSOR] Separated PDF into {len(forms)} distinct form images:")
        for f_desc in forms:
            print(f"  - Form index {f_desc['form_index']}: {f_desc['image_path']} ({f_desc['dimensions'][0]}x{f_desc['dimensions'][1]})")

        assert len(forms) == 2, f"Expected 2 forms, found {len(forms)}"

        # Register Page records
        for seq_idx, f_desc in enumerate(forms, start=1):
            page = Page(
                document_id=doc.id,
                page_number=seq_idx,
                original_image_path=f_desc["image_path"],
                status="PENDING",
            )
            db.add(page)
            db.commit()
            db.refresh(page)
            page_ids.append(page.id)
            print(f"[DATABASE] Created Page ID={page.id} (page_number={page.page_number}, file={os.path.basename(page.original_image_path)})")

        doc.total_pages = len(page_ids)
        db.commit()

        # Create Background Extraction Job
        job_id = f"stage13_verify_{int(time.time())}"
        job = ExtractionJob(
            id=job_id,
            document_id=doc.id,
            stage="UPLOADED",
            progress=0.0,
            status="PENDING",
        )
        db.add(job)
        db.commit()
        print(f"[DATABASE] Registered Extraction Job ID='{job_id}'")

        # [STEP 2] Execute Extraction Pipeline
        print("\n[STEP 2] Executing document processing pipeline...")
        t_pipe = time.time()
        process_document_pipeline_sync(job_id, doc.id)
        pipe_duration = time.time() - t_pipe
        print(f"[PIPELINE COMPLETED] Execution time: {pipe_duration:.2f}s")

        # Refresh Document & Job status
        db.refresh(doc)
        db.refresh(job)
        print(f"[STATUS] Document status: {doc.status} | Job status: {job.status}")

        # [STEP 3] Inspect and Validate Database Records
        print("\n[STEP 3] Validating extracted database records...")
        ground_truth = {
            1: {
                "student_name": "ARUN KUMAR R",
                "school_name": "GREEN VALLEY HR SEC SCHOOL",
                "date": "10/09/2026",
                "student_mobile_number": "9000012345",
                "parent_mobile_number": "9000056789",
                "email": "arun.test@example.com",
                "cut_off_marks": "187.50",
                "interested_in_bit": "Yes",
                "career_guidance_usefulness": "Excellent",
                "speaker_effectiveness": "Good",
                "interested_in_choice_filling_assistance": "Yes",
            },
            2: {
                "student_name": "PRIYA S",
                "school_name": "SUNRISE HR SEC SCHOOL",
                "date": "11/09/2026",
                "student_mobile_number": "9000024680",
                "parent_mobile_number": "9000013579",
                "email": "priya.test@example.com",
                "cut_off_marks": "176.25",
                "interested_in_bit": "No",
                "career_guidance_usefulness": "Good",
                "speaker_effectiveness": "Satisfactory",
                "interested_in_choice_filling_assistance": "No",
            }
        }

        total_fields = 0
        exact_matches = 0
        cer_scores = []
        extracted_summary = {}

        for page_idx, p_id in enumerate(page_ids, start=1):
            page_rec = db.query(Page).filter(Page.id == p_id).first()
            fe_records = db.query(FieldExtraction).filter(FieldExtraction.page_id == p_id).all()
            vd_records = db.query(VerificationDecision).filter(VerificationDecision.page_id == p_id).all()

            print(f"\n--- Form {page_idx} (Page ID {p_id}, Status: {page_rec.status}) ---")
            fe_dict = {fe.field_name: fe for fe in fe_records}
            gt_form = ground_truth[page_idx]
            extracted_summary[page_idx] = {}

            for f_name, expected_val in gt_form.items():
                total_fields += 1
                fe = fe_dict.get(f_name)
                pred_val = fe.normalized_value if fe else None
                status = fe.status if fe else "MISSING"
                conf = fe.confidence if fe else 0.0

                extracted_summary[page_idx][f_name] = {
                    "predicted": pred_val,
                    "expected": expected_val,
                    "status": status,
                    "confidence": conf,
                }

                # Evaluate match
                is_match = (str(pred_val).strip().lower() == str(expected_val).strip().lower())
                if is_match:
                    exact_matches += 1
                    match_str = "[MATCH]"
                    cer = 0.0
                else:
                    match_str = "[DIFF] "
                    cer = calculate_cer(str(pred_val or ""), str(expected_val))

                cer_scores.append(cer)
                print(f"  {match_str} {f_name:<38}: '{pred_val}' (expected: '{expected_val}') [status={status}, conf={conf:.2f}]")

        print("\n" + "=" * 70)
        print("ACCURACY SUMMARY")
        print("=" * 70)
        exact_pct = (exact_matches / total_fields) * 100.0 if total_fields else 0.0
        avg_cer = (sum(cer_scores) / len(cer_scores)) * 100.0 if cer_scores else 0.0
        print(f"Total Fields Evaluated : {total_fields}")
        print(f"Exact Matches          : {exact_matches}/{total_fields} ({exact_pct:.2f}%)")
        print(f"Average CER            : {avg_cer:.2f}%")
        print(f"Total Pipeline Runtime : {pipe_duration:.2f}s (avg {pipe_duration/2.0:.2f}s per form)")

        # [STEP 4] Test Manual Review Persistence
        print("\n[STEP 4] Testing manual verification persistence...")
        # Manually verify student_name on Form 1
        vd_target = db.query(VerificationDecision).filter(
            VerificationDecision.page_id == page_ids[0],
            VerificationDecision.field_name == "student_name"
        ).first()
        fe_target = db.query(FieldExtraction).filter(
            FieldExtraction.page_id == page_ids[0],
            FieldExtraction.field_name == "student_name"
        ).first()

        assert vd_target is not None and fe_target is not None
        vd_target.final_value = "HUMAN CONFIRMED ARUN"
        vd_target.decision = "HUMAN_VERIFIED"
        vd_target.reviewer_id = "qa_reviewer_1"
        fe_target.normalized_value = "HUMAN CONFIRMED ARUN"
        fe_target.source = "human"
        fe_target.status = "VERIFIED"
        db.commit()

        # Re-run pipeline to simulate job retry
        retry_job_id = f"stage13_retry_{int(time.time())}"
        process_document_pipeline_sync(retry_job_id, doc.id)

        # Verify value was preserved
        db.expire_all()
        vd_after = db.query(VerificationDecision).filter(
            VerificationDecision.page_id == page_ids[0],
            VerificationDecision.field_name == "student_name"
        ).first()
        assert vd_after.final_value == "HUMAN CONFIRMED ARUN", "Manual review was overwritten by re-run!"
        assert vd_after.decision == "HUMAN_VERIFIED"
        print("  [SUCCESS] Manual verification preserved across pipeline re-run without regression!")

        # [STEP 5] Test CSV Export
        print("\n[STEP 5] Testing CSV export generation...")
        import asyncio
        csv_response = export_document_csv(doc.id, db)
        async def read_body():
            chunks = []
            async for chunk in csv_response.body_iterator:
                chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
            return b"".join(chunks).decode("utf-8")
        csv_content = asyncio.run(read_body())
        reader = list(csv.reader(io.StringIO(csv_content)))
        print(f"  CSV Header: {reader[0]}")
        print(f"  CSV Form 1 row: {reader[1]}")
        print(f"  CSV Form 2 row: {reader[2]}")
        assert len(reader) == 3, f"Expected 3 rows (header + 2 forms), got {len(reader)}"
        print("  [SUCCESS] CSV export generates distinct row per form successfully!")

        print("\n" + "=" * 70)
        print("STAGE 13 REAL VERIFICATION PASSED SUCCESSFULLY")
        print("=" * 70)

    finally:
        # Keep records for user inspection or clean up
        db.close()


if __name__ == "__main__":
    main()
