from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.app.db.session import get_db
from backend.app.models.domain import Document, Page, VerificationDecision, FieldExtraction
from backend.app.schemas.pydantic_models import APIResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("", response_model=APIResponse)
def get_system_metrics(db: Session = Depends(get_db)):
    total_docs = db.query(Document).count()
    total_pages = db.query(Page).count()
    completed_pages = db.query(Page).filter(Page.status == "COMPLETED").count()
    failed_pages = db.query(Page).filter(Page.status == "FAILED").count()

    total_decisions = db.query(VerificationDecision).count()
    agreements = db.query(VerificationDecision).filter(VerificationDecision.decision == "AGREEMENT").count()
    conflicts = db.query(VerificationDecision).filter(VerificationDecision.decision == "MANUAL_REVIEW").count()
    cnn_selected = db.query(VerificationDecision).filter(VerificationDecision.decision == "CNN_SELECTED").count()
    gemini_selected = db.query(VerificationDecision).filter(VerificationDecision.decision == "GEMINI_SELECTED").count()
    cnn_unavailable = db.query(VerificationDecision).filter(VerificationDecision.decision == "CNN_UNAVAILABLE").count()

    agreement_rate = float(agreements / max(1, total_decisions))
    conflict_rate = float(conflicts / max(1, total_decisions))

    return APIResponse(data={
        "total_documents": total_docs,
        "total_pages": total_pages,
        "completed_pages": completed_pages,
        "failed_pages": failed_pages,
        "total_verification_decisions": total_decisions,
        "agreements": agreements,
        "conflicts": conflicts,
        "cnn_selected": cnn_selected,
        "gemini_selected": gemini_selected,
        "cnn_unavailable": cnn_unavailable,
        "agreement_rate": round(agreement_rate, 4),
        "conflict_rate": round(conflict_rate, 4)
    })
