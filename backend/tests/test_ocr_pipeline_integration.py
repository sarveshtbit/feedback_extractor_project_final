"""
End-to-end integration tests for the OCR pipeline integration.

Tests:
1. OCRService orchestration and field extraction.
2. Background processing task lifecycle and DB persistence.
3. API response contracts for /pages/{id}/results, /pages/{id}/verify, and /models/cnn/status.
4. CSV export contract with extracted OCR and bubble fields.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.session import SessionLocal
from backend.app.models.domain import Document, Page, ExtractionJob, FieldExtraction, VerificationDecision
from backend.app.services.ocr_service import OCRService
from backend.app.workers.tasks import process_document_pipeline_sync


class TestOCRPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_ocr_service_status_contract(self):
        ocr_service = OCRService()
        status = ocr_service.get_status()
        self.assertIn("ocr_pipeline", status)
        self.assertIn("trocr", status)
        self.assertIn("paddleocr", status)
        self.assertIn("bubble_detector", status)

    def test_models_status_api_endpoint(self):
        response = self.client.get("/api/v1/models/cnn/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertIn("status", data)
        self.assertIn("architecture", data)
        self.assertIn("weights_exist", data)

    def test_ocr_models_status_api_endpoint(self):
        response = self.client.get("/api/v1/models/ocr/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertIn("ocr_pipeline", data)
        self.assertIn("trocr", data)

    def test_end_to_end_ocr_task_and_persistence(self):
        """Simulates full background OCR pipeline execution on a mock document."""
        # 1. Create a dummy test image and save to disk
        test_dir = "./storage/test_uploads"
        os.makedirs(test_dir, exist_ok=True)
        test_img_path = os.path.join(test_dir, "test_ocr_form.png")

        canvas = np.full((3508, 2479, 3), 255, dtype=np.uint8)
        # Draw registration markers (4 corners)
        cv2.rectangle(canvas, (100, 100), (180, 180), (0, 0, 0), -1)
        cv2.rectangle(canvas, (2299, 100), (2379, 180), (0, 0, 0), -1)
        cv2.rectangle(canvas, (2299, 3328), (2379, 3408), (0, 0, 0), -1)
        cv2.rectangle(canvas, (100, 3328), (180, 3408), (0, 0, 0), -1)

        cv2.imwrite(test_img_path, canvas)

        # 2. Insert Document & Page into DB
        doc = Document(
            filename="test_ocr_form.png",
            file_type="image/png",
            total_pages=1,
            status="UPLOADED",
            storage_path=test_img_path,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        page = Page(
            document_id=doc.id,
            page_number=1,
            original_image_path=test_img_path,
            status="PENDING",
        )
        self.db.add(page)
        self.db.commit()
        self.db.refresh(page)

        job_id = f"test-job-{doc.id}"
        job = ExtractionJob(
            id=job_id,
            document_id=doc.id,
            stage="UPLOADED",
            progress=0.0,
            status="PENDING",
        )
        self.db.add(job)
        self.db.commit()

        # 3. Execute background pipeline task
        process_document_pipeline_sync(job_id, doc.id)

        # 4. Verify DB records in clean session state
        self.db.expire_all()
        updated_doc = self.db.query(Document).filter(Document.id == doc.id).first()
        self.assertEqual(updated_doc.status, "COMPLETED")

        updated_page = self.db.query(Page).filter(Page.id == page.id).first()
        self.assertEqual(updated_page.status, "COMPLETED")

        extractions = self.db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).all()
        self.assertGreaterEqual(len(extractions), 11)

        decisions = self.db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).all()
        self.assertGreaterEqual(len(decisions), 11)

        # 5. Verify API results retrieval
        res_page = self.client.get(f"/api/v1/pages/{page.id}/results")
        self.assertEqual(res_page.status_code, 200)
        page_data = res_page.json()["data"]
        self.assertIn("fields", page_data)
        self.assertIn("student_name", page_data["fields"])
        self.assertIn("student_mobile_number", page_data["fields"])

        # 6. Verify manual verification endpoint
        verify_res = self.client.post(
            f"/api/v1/pages/{page.id}/verify",
            json={
                "field_name": "student_name",
                "corrected_value": "Verified Student Name",
                "operator": "test_reviewer",
                "reason": "Corrected during QA",
            },
        )
        self.assertEqual(verify_res.status_code, 200)

        # 7. Verify CSV export
        csv_res = self.client.get(f"/api/v1/documents/{doc.id}/export/csv")
        self.assertEqual(csv_res.status_code, 200)
        self.assertIn("text/csv", csv_res.headers["content-type"])
        self.assertIn("Student Name", csv_res.text)


if __name__ == "__main__":
    unittest.main()
