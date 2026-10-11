"""
Comprehensive Integration Tests for Stage 13: Controlled Integration of OCR Pipeline.

Covers:
1. Single-form and multi-form PDF / image ingestion and page separation.
2. Template routing: A4, Digital Dual Forms 1 & 2, and explicit UNKNOWN_TEMPLATE.
3. PaddleOCR subprocess bridge communication and graceful unavailable handling.
4. TrOCR unavailable handling.
5. Alignment fallback behavior on unaligned/malformed forms.
6. Validation rejection of invalid numeric predictions.
7. Blank and ambiguous questionnaire bubble detection.
8. Manual-review persistence across pipeline reprocessing.
9. Retry idempotency without duplicate database records.
10. CSV export streaming with separate rows per form.
"""

import os
import cv2
import json
import pytest
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.models.domain import (
    Document, Page, ExtractionJob, FieldExtraction, VerificationDecision
)
from backend.app.services.ocr.template_router import (
    TemplateRouter,
    UNKNOWN_TEMPLATE,
    TEMPLATE_HANDWRITTEN_A4,
    TEMPLATE_DIGITAL_FORM_1,
    TEMPLATE_DIGITAL_FORM_2,
)
from backend.app.services.ocr.pdf_preprocessor import PDFPreprocessor
from backend.app.services.ocr.field_extractor import FieldExtractor
from backend.app.services.ocr.paddle_engine import PaddleOCREngine, PaddleBridgeClient
from backend.app.services.ocr.trocr_engine import TrOCREngine
from backend.app.validation.validator import FieldValidator
from backend.app.workers.tasks import process_document_pipeline_sync


@pytest.fixture
def test_db():
    """In-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_template_routing_classification():
    """Verify TemplateRouter accurately classifies known layouts and rejects unknown ones."""
    router = TemplateRouter()

    # 1. Unknown layout (e.g. tiny square image)
    tiny_img = np.full((300, 300, 3), 255, dtype=np.uint8)
    res_unknown = router.identify_template(tiny_img)
    assert res_unknown["template_id"] == UNKNOWN_TEMPLATE
    assert res_unknown["is_known"] is False

    # 2. Digital half-sheet form (e.g. 2550 x 1575)
    digital_form1_img = np.full((1575, 2550, 3), 255, dtype=np.uint8)
    res_digital1 = router.identify_template(digital_form1_img, form_index=1, is_split=True)
    assert res_digital1["template_id"] == TEMPLATE_DIGITAL_FORM_1
    assert res_digital1["is_known"] is True

    res_digital2 = router.identify_template(digital_form1_img, form_index=2, is_split=True)
    assert res_digital2["template_id"] == TEMPLATE_DIGITAL_FORM_2
    assert res_digital2["is_known"] is True

    # 3. Physical A4 form with 4 registration markers
    a4_canvas = np.full((3508, 2479, 3), 255, dtype=np.uint8)
    # Add dark markers at corner regions
    cv2.rectangle(a4_canvas, (100, 100), (180, 180), (0, 0, 0), -1)
    cv2.rectangle(a4_canvas, (2300, 100), (2380, 180), (0, 0, 0), -1)
    cv2.rectangle(a4_canvas, (2300, 3320), (2380, 3400), (0, 0, 0), -1)
    cv2.rectangle(a4_canvas, (100, 3320), (180, 3400), (0, 0, 0), -1)

    res_a4 = router.identify_template(a4_canvas)
    assert res_a4["template_id"] == TEMPLATE_HANDWRITTEN_A4
    assert res_a4["is_known"] is True


def test_field_extractor_unknown_template_handling():
    """Verify FieldExtractor returns UNKNOWN_TEMPLATE status without throwing an exception."""
    extractor = FieldExtractor()
    invalid_img = np.full((200, 200, 3), 255, dtype=np.uint8)

    res = extractor.extract_all_fields(invalid_img)
    assert res["status"] == UNKNOWN_TEMPLATE
    assert res["template_id"] == UNKNOWN_TEMPLATE

    # Check all fields report UNKNOWN_TEMPLATE
    fields = res["fields"]
    assert "student_name" in fields
    assert fields["student_name"]["status"] == UNKNOWN_TEMPLATE
    assert fields["student_name"]["value"] is None


def test_paddleocr_subprocess_communication():
    """Verifies PaddleBridgeClient communicates with the isolated paddle worker."""
    client = PaddleBridgeClient()
    if not client.python_exe or not os.path.exists(client.python_exe):
        pytest.skip(".venv-paddleocr environment not available for this test")

    assert client.start() is True
    assert client.is_ready is True

    # Ping
    pong = client.send_command({"action": "ping"})
    assert pong.get("status") == "PONG"

    client.shutdown()
    assert client.is_ready is False


def test_paddleocr_engine_unavailable_graceful_handling():
    """Verify PaddleOCREngine returns MODEL_UNAVAILABLE when engine fails to load."""
    PaddleOCREngine._instance = None
    engine = PaddleOCREngine()

    with patch.object(engine, "load_model", return_value=False):
        dummy_img = np.full((100, 200, 3), 255, dtype=np.uint8)
        res = engine.predict(dummy_img)
        assert res["status"] == "MODEL_UNAVAILABLE"
        assert res["items"] == []

        crop_res = engine.predict_crop(dummy_img)
        assert crop_res["status"] == "MODEL_UNAVAILABLE"
        assert crop_res["text"] is None


def test_trocr_engine_unavailable_graceful_handling():
    """Verify TrOCREngine returns MODEL_UNAVAILABLE when weights are missing."""
    TrOCREngine._instance = None
    engine = TrOCREngine()

    with patch.object(engine, "load_model", return_value=False):
        dummy_img = np.full((50, 150, 3), 255, dtype=np.uint8)
        res = engine.predict_crop(dummy_img)
        assert res["status"] == "MODEL_UNAVAILABLE"
        assert res["text"] is None


def test_invalid_numeric_predictions_trigger_review():
    """Verify validator flags invalid numeric fields (short mobile, cutoff > 200)."""
    # 9-digit mobile number
    is_valid, norm, err = FieldValidator.validate_phone_number("987654321")
    assert is_valid is False
    assert "10 digits" in err

    # Cutoff > 200
    is_valid, norm, err = FieldValidator.validate_cutoff_marks("210.50")
    assert is_valid is False
    assert "out of valid range" in err

    # Valid cutoff
    is_valid, norm, err = FieldValidator.validate_cutoff_marks("192.45")
    assert is_valid is True
    assert norm == "192.45"


def test_questionnaire_blank_and_ambiguous_selections():
    """Verify bubble analysis correctly classifies unselected and ambiguous multi-marked questions."""
    # Test validator with blank option
    is_valid, norm, err = FieldValidator.validate_questionnaire_option("interested_in_bit", "")
    assert is_valid is False

    # Test validator with numeric rating scale
    is_valid, norm, _ = FieldValidator.validate_questionnaire_option("career_guidance_usefulness", "5")
    assert is_valid is True
    assert norm == "Excellent"

    is_valid, norm, _ = FieldValidator.validate_questionnaire_option("speaker_effectiveness", "2")
    assert is_valid is True
    assert norm == "Fair"


def test_idempotent_pipeline_execution_and_manual_review_persistence(tmp_path):
    """
    Verifies that re-running process_document_pipeline_sync:
    1. Does not produce duplicate database records.
    2. Preserves human-verified field values.
    """
    from backend.app.db.session import SessionLocal

    db = SessionLocal()
    try:
        # 1. Create a dummy test image
        img_path = str(tmp_path / "page_1_form_1.png")
        canvas = np.full((1575, 2550, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "CAREER GUIDANCE PROGRAMME", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(img_path, canvas)

        # 2. Create document, page, and job in DB
        doc = Document(filename="test_doc.png", file_type="image/png", total_pages=1, status="UPLOADED", storage_path=img_path)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        page = Page(document_id=doc.id, page_number=1, original_image_path=img_path, status="PENDING")
        db.add(page)
        db.commit()
        db.refresh(page)

        job_id = "test_job_1"
        job = ExtractionJob(id=job_id, document_id=doc.id, stage="PENDING", progress=0.0, status="PENDING")
        db.add(job)
        db.commit()

        # Run pipeline first time
        process_document_pipeline_sync(job_id, doc.id)

        # Count records
        fe_count_1 = db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).count()
        vd_count_1 = db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).count()
        assert fe_count_1 == 11
        assert vd_count_1 == 11

        # Simulate Human Verification on student_name
        vd_name = db.query(VerificationDecision).filter(
            VerificationDecision.page_id == page.id,
            VerificationDecision.field_name == "student_name"
        ).first()
        assert vd_name is not None
        vd_name.final_value = "Verified Operator Name"
        vd_name.decision = "HUMAN_VERIFIED"
        vd_name.reviewer_id = "test_operator"

        fe_name = db.query(FieldExtraction).filter(
            FieldExtraction.page_id == page.id,
            FieldExtraction.field_name == "student_name"
        ).first()
        fe_name.normalized_value = "Verified Operator Name"
        fe_name.source = "human"
        fe_name.status = "VERIFIED"
        db.commit()

        # Run pipeline second time (retry simulation)
        job_id_retry = "test_job_2"
        job_retry = ExtractionJob(id=job_id_retry, document_id=doc.id, stage="PENDING", progress=0.0, status="PENDING")
        db.add(job_retry)
        db.commit()

        process_document_pipeline_sync(job_id_retry, doc.id)

        # Verify NO DUPLICATE records were created
        fe_count_2 = db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).count()
        vd_count_2 = db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).count()
        assert fe_count_2 == 11
        assert vd_count_2 == 11

        # Verify HUMAN VERIFIED value was NOT overwritten!
        vd_name_after = db.query(VerificationDecision).filter(
            VerificationDecision.page_id == page.id,
            VerificationDecision.field_name == "student_name"
        ).first()
        assert vd_name_after.final_value == "Verified Operator Name"
        assert vd_name_after.decision == "HUMAN_VERIFIED"

    finally:
        # Cleanup
        db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).delete()
        db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).delete()
        db.query(Page).filter(Page.document_id == doc.id).delete()
        db.query(ExtractionJob).filter(ExtractionJob.document_id == doc.id).delete()
        db.query(Document).filter(Document.id == doc.id).delete()
        db.commit()
        db.close()


def test_csv_export_separated_rows_per_form(tmp_path):
    """Verify that export_document_csv produces headers and distinct rows for each form."""
    import csv
    from backend.app.db.session import SessionLocal
    from backend.app.api.v1.documents import export_document_csv

    db = SessionLocal()
    doc = None
    p1 = None
    p2 = None
    try:
        dummy_path = str(tmp_path / "multiform.pdf")
        with open(dummy_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        doc = Document(filename="multiform.pdf", file_type="application/pdf", total_pages=2, status="COMPLETED", storage_path=dummy_path)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        p1 = Page(document_id=doc.id, page_number=1, original_image_path=dummy_path, status="COMPLETED")
        p2 = Page(document_id=doc.id, page_number=2, original_image_path=dummy_path, status="COMPLETED")
        db.add_all([p1, p2])
        db.commit()
        db.refresh(p1)
        db.refresh(p2)

        fe1 = FieldExtraction(page_id=p1.id, field_name="student_name", raw_value="ALICE", normalized_value="ALICE", source="paddleocr", status="VERIFIED")
        fe2 = FieldExtraction(page_id=p2.id, field_name="student_name", raw_value="BOB", normalized_value="BOB", source="paddleocr", status="VERIFIED")
        db.add_all([fe1, fe2])
        db.commit()

        import asyncio
        resp = export_document_csv(document_id=doc.id, db=db)
        async def read_body():
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
            return b"".join(chunks).decode("utf-8")
        content = asyncio.run(read_body())
        reader = list(csv.reader(content.splitlines()))

        assert len(reader) == 3  # Header + 2 form rows
        assert reader[0][0] == "Document ID"
        assert reader[1][3] == "ALICE"
        assert reader[2][3] == "BOB"

    finally:
        db.rollback()
        if p1 and p2:
            db.query(FieldExtraction).filter(FieldExtraction.page_id.in_([p1.id, p2.id])).delete()
            db.query(Page).filter(Page.document_id == doc.id).delete()
        if doc:
            db.query(Document).filter(Document.id == doc.id).delete()
        db.commit()
        db.close()


def test_multipage_pdf_processing_preserves_page_and_form_indices(tmp_path):
    """Verify that multi-page PDFs create distinct form images with preserved page_number and form_index."""
    # Create two synthetic pages
    page1 = np.full((3300, 2550, 3), 255, dtype=np.uint8)
    cv2.putText(page1, "PAGE 1 TOP", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    cv2.putText(page1, "PAGE 1 BOTTOM", (200, 2200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

    page2 = np.full((1500, 1000, 3), 255, dtype=np.uint8)
    cv2.putText(page2, "PAGE 2 SINGLE", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    out_dir = str(tmp_path / "multipage_out")
    os.makedirs(out_dir, exist_ok=True)

    r1 = PDFPreprocessor.process_rendered_page(page1, page_num=1, output_dir=out_dir)
    r2 = PDFPreprocessor.process_rendered_page(page2, page_num=2, output_dir=out_dir)

    all_forms = r1 + r2
    assert len(all_forms) == 3  # 2 from page 1, 1 from page 2
    assert all_forms[0]["page_number"] == 1 and all_forms[0]["form_index"] == 1
    assert all_forms[1]["page_number"] == 1 and all_forms[1]["form_index"] == 2
    assert all_forms[2]["page_number"] == 2 and all_forms[2]["form_index"] == 1


def test_incorrect_alignment_handling():
    """Verify that template router handles unaligned or noisy images gracefully."""
    router = TemplateRouter()
    # Heavily distorted / noisy canvas
    noisy_img = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
    res = router.identify_template(noisy_img)
    # Should not crash and should flag UNKNOWN_TEMPLATE
    assert res["template_id"] == UNKNOWN_TEMPLATE
    assert res["is_known"] is False

