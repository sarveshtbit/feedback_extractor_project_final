import os
import uuid
import datetime
import io
import csv
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

import cv2
from backend.app.db.session import get_db
from backend.app.models.domain import Document, Page, ExtractionJob
from backend.app.schemas.pydantic_models import APIResponse
from backend.app.services.pdf_service import PDFProcessor
from backend.app.services.ocr.pdf_preprocessor import PDFPreprocessor
from backend.app.workers.tasks import process_document_pipeline_sync

router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_STORAGE_DIR = "./storage/uploads"
os.makedirs(UPLOAD_STORAGE_DIR, exist_ok=True)

@router.post("/upload", response_model=APIResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a single image (PNG/JPEG) or multi-page PDF. Creates Document, Page, and Job records.
    Dispatches pipeline execution as a background task to prevent proxy timeouts / ECONNRESET.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_uuid = str(uuid.uuid4())[:8]
    sanitized_filename = f"{timestamp}_{file_uuid}_{file.filename}"
    file_path = os.path.join(UPLOAD_STORAGE_DIR, sanitized_filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    file_ext = os.path.splitext(file.filename)[1].lower()
    is_pdf = file_ext == ".pdf"
    
    total_pages = 1
    if is_pdf:
        try:
            total_pages = PDFProcessor.get_pdf_page_count(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")

    doc = Document(
        filename=file.filename,
        file_type="application/pdf" if is_pdf else "image/png",
        total_pages=total_pages,
        status="UPLOADED",
        storage_path=file_path
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Render PDF pages or register image page using PDFPreprocessor
    pages_to_process = []
    page_dir = os.path.join(UPLOAD_STORAGE_DIR, f"doc_{doc.id}_pages")
    if is_pdf:
        forms = PDFPreprocessor.process_pdf(file_path, page_dir)
    else:
        img = cv2.imread(file_path)
        forms = PDFPreprocessor.process_rendered_page(img, page_num=1, output_dir=page_dir)

    for seq_idx, f_desc in enumerate(forms, start=1):
        page = Page(
            document_id=doc.id,
            page_number=seq_idx,
            original_image_path=f_desc["image_path"],
            status="PENDING"
        )
        db.add(page)
        pages_to_process.append(page)

    doc.total_pages = len(pages_to_process)
    db.commit()

    # Create Background Extraction Job
    job_id = str(uuid.uuid4())
    job = ExtractionJob(
        id=job_id,
        document_id=doc.id,
        stage="UPLOADED",
        progress=0.0,
        status="PENDING"
    )
    db.add(job)
    db.commit()

    # Dispatch pipeline task asynchronously in background
    background_tasks.add_task(process_document_pipeline_sync, job_id, doc.id)

    return APIResponse(data={
        "document_id": doc.id,
        "job_id": job_id,
        "filename": doc.filename,
        "total_pages": doc.total_pages,
        "status": doc.status
    })

@router.post("/batch", response_model=APIResponse)
async def upload_batch_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Handles batch upload of multiple images/PDFs.
    """
    results = []
    for file in files:
        res = await upload_document(background_tasks, file, db)
        results.append(res.data)
    return APIResponse(data={"batch_count": len(results), "documents": results})

@router.get("", response_model=APIResponse)
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    out = []
    for d in docs:
        out.append({
            "document_id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "total_pages": d.total_pages,
            "status": d.status,
            "created_at": d.created_at.isoformat()
        })
    return APIResponse(data=out)

@router.get("/{document_id}", response_model=APIResponse)
def get_document_details(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()
    page_list = []
    for p in pages:
        page_list.append({
            "page_id": p.id,
            "page_number": p.page_number,
            "status": p.status,
            "original_image_path": p.original_image_path,
            "aligned_image_path": p.aligned_image_path
        })

    return APIResponse(data={
        "document_id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "total_pages": doc.total_pages,
        "status": doc.status,
        "created_at": doc.created_at.isoformat(),
        "pages": page_list
    })

@router.get("/{document_id}/export/csv")
def export_document_csv(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = db.query(Page).filter(Page.document_id == doc.id).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Document ID", "Page ID", "Page Number", "Student Name", "School Name", "Date",
        "Student Mobile (Final)", "Student Mobile (Gemini)", "Student Mobile (CNN)",
        "Cut Off Marks (Final)", "Cut Off Marks (Gemini)", "Cut Off Marks (CNN)",
        "Email", "Parent Mobile", "Interested BIT", "Career Guidance", "Speaker Guidance", "Choice Filling"
    ])

    for p in pages:
        fe_dict = {}
        for fe in p.field_extractions:
            fe_dict[fe.field_name] = fe.normalized_value or fe.raw_value

        dec_dict = {}
        for vd in p.verification_decisions:
            dec_dict[vd.field_name] = vd

        mob_dec = dec_dict.get("student_mobile_number")
        cutoff_dec = dec_dict.get("cut_off_marks")

        writer.writerow([
            doc.id, p.id, p.page_number,
            fe_dict.get("student_name", ""),
            fe_dict.get("school_name", ""),
            fe_dict.get("date", ""),
            mob_dec.final_value if mob_dec else fe_dict.get("student_mobile_number", ""),
            mob_dec.gemini_value if mob_dec else "",
            mob_dec.cnn_value if mob_dec else "",
            cutoff_dec.final_value if cutoff_dec else fe_dict.get("cut_off_marks", ""),
            cutoff_dec.gemini_value if cutoff_dec else "",
            cutoff_dec.cnn_value if cutoff_dec else "",
            fe_dict.get("email", ""),
            fe_dict.get("parent_mobile_number", ""),
            fe_dict.get("interested_in_bit", ""),
            fe_dict.get("career_guidance_usefulness", ""),
            fe_dict.get("speaker_effectiveness", ""),
            fe_dict.get("interested_in_choice_filling_assistance", "")
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=document_{document_id}_extracted.csv"}
    )
