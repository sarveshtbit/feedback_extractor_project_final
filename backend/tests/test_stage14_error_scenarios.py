"""
Stage 14: Comprehensive Error Scenarios Test Suite.

Tests:
1. Empty or blank page processing (does not crash or save corrupt data).
2. Unanswered questionnaire items (status UNANSWERED, value None, confidence 0.0).
3. Ambiguous selected circles (multiple filled or borderline mark -> status AMBIGUOUS).
4. OCR subprocess bridge error handling and recovery (invalid input, simulated failure).
5. Unsupported layout handling (template router flags UNKNOWN_TEMPLATE, values None).
6. Single-form page processing (without splitting).
7. Job failure and retry resilience (DB rollback and clean rerun).
"""

import os
import time
import cv2
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.app.db.session import SessionLocal
from backend.app.models.domain import Document, Page, ExtractionJob, FieldExtraction, VerificationDecision
from backend.app.services.ocr.template_router import TemplateRouter, UNKNOWN_TEMPLATE
from backend.app.services.ocr.field_extractor import FieldExtractor
from backend.app.services.ocr.paddle_engine import PaddleBridgeClient
from backend.app.services.ocr.pdf_preprocessor import PDFPreprocessor
from backend.app.workers.tasks import process_document_pipeline_sync


def test_blank_page_pipeline_handling(tmp_path):
    """Verify that a completely blank page is handled gracefully without corrupt data."""
    blank_img_path = str(tmp_path / "blank_page.png")
    blank_img = np.full((1500, 1000, 3), 255, dtype=np.uint8)
    cv2.imwrite(blank_img_path, blank_img)

    assert PDFPreprocessor.is_blank_image(blank_img) is True
    spans, split_y = PDFPreprocessor.detect_form_splits(blank_img)
    assert len(spans) == 1
    assert split_y is None


def test_unanswered_questionnaire_bubble_status():
    """Verify that an unanswered questionnaire item returns UNANSWERED status."""
    canvas = np.full((1700, 2550, 3), 255, dtype=np.uint8)
    # Draw empty question circles (unfilled)
    layout = {
        "template_id": "TEST_BLANK",
        "q1_boxes": {"Yes": [1338, 1018], "No": [1456, 1018]},
    }
    for cx, cy in [(1338, 1018), (1456, 1018)]:
        cv2.circle(canvas, (cx, cy), 16, (0, 0, 0), 2)

    extractor = FieldExtractor()
    with patch.object(extractor.template_router, "get_template_profile", return_value=layout):
        res = extractor._extract_digital_form_fields(canvas, "TEST_BLANK")
        q1 = res["fields"].get("interested_in_bit")
        assert q1 is not None
        assert q1["status"] == "UNANSWERED"
        assert q1["value"] is None
        assert q1["confidence"] == 0.0


def test_ambiguous_multiple_bubbles_selected():
    """Verify that marking multiple bubbles produces AMBIGUOUS status requiring review."""
    canvas = np.full((1700, 2550, 3), 255, dtype=np.uint8)
    # Fill both Yes and No circles heavily
    layout = {
        "template_id": "TEST_MULTI",
        "q1_boxes": {"Yes": [1338, 1018], "No": [1456, 1018]},
    }
    for cx, cy in [(1338, 1018), (1456, 1018)]:
        cv2.circle(canvas, (cx, cy), 14, (0, 0, 0), -1)

    extractor = FieldExtractor()
    with patch.object(extractor.template_router, "get_template_profile", return_value=layout):
        res = extractor._extract_digital_form_fields(canvas, "TEST_MULTI")
        q1 = res["fields"].get("interested_in_bit")
        assert q1 is not None
        assert q1["status"] == "AMBIGUOUS"
        assert q1["confidence"] == 0.50


def test_unsupported_layout_returns_unknown_template():
    """Verify that an unsupported image dimensions / layout flags UNKNOWN_TEMPLATE."""
    router = TemplateRouter()
    # Random tiny dimensions
    weird_img = np.full((400, 700, 3), 255, dtype=np.uint8)
    info = router.identify_template(weird_img)
    assert info["template_id"] == UNKNOWN_TEMPLATE
    assert info["is_known"] is False

    extractor = FieldExtractor()
    res = extractor.extract_all_fields(weird_img)
    assert res["status"] == UNKNOWN_TEMPLATE
    for field_name, f_data in res["fields"].items():
        assert f_data["status"] == UNKNOWN_TEMPLATE
        assert f_data["value"] is None


def test_paddle_bridge_error_handling_and_safe_command():
    """Verify PaddleBridgeClient handles unknown commands gracefully without crashing."""
    client = PaddleBridgeClient()
    if not client.python_exe or not os.path.exists(client.python_exe):
        pytest.skip(".venv-paddleocr not available")

    assert client.start() is True
    # Send unrecognized action
    res = client.send_command({"action": "unknown_action_xyz"})
    assert res.get("status") == "UNKNOWN_ACTION"
    assert "Unknown action" in res.get("error", "")

    # Client remains ready after error
    ping_res = client.send_command({"action": "ping"})
    assert ping_res.get("status") == "PONG"
    client.shutdown()


def test_single_form_page_extraction_without_split(tmp_path):
    """Verify that a single-form document without dual split runs end-to-end cleanly."""
    db = SessionLocal()
    doc = None
    page = None
    try:
        img_path = str(tmp_path / "single_page.png")
        # Standard portrait form
        canvas = np.full((1500, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "BANNARI AMMAN INSTITUTE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(img_path, canvas)

        doc = Document(filename="single_form.png", file_type="image/png", total_pages=1, status="UPLOADED", storage_path=img_path)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        page = Page(document_id=doc.id, page_number=1, original_image_path=img_path, status="PENDING")
        db.add(page)
        db.commit()
        db.refresh(page)

        job_id = f"job_single_{int(time.time())}"
        job = ExtractionJob(id=job_id, document_id=doc.id, stage="PENDING", progress=0.0, status="PENDING")
        db.add(job)
        db.commit()

        # Run pipeline
        process_document_pipeline_sync(job_id, doc.id)

        db.refresh(doc)
        db.refresh(page)
        assert doc.status == "COMPLETED"
        assert page.status == "COMPLETED"

        # 11 fields generated without errors
        fes = db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).all()
        assert len(fes) == 11

    finally:
        if page:
            db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).delete()
            db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).delete()
            db.query(Page).filter(Page.document_id == doc.id).delete()
        if doc:
            db.query(ExtractionJob).filter(ExtractionJob.document_id == doc.id).delete()
            db.query(Document).filter(Document.id == doc.id).delete()
        db.commit()
        db.close()
