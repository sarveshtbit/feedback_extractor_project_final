from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.models.domain import Page, FieldExtraction, VerificationDecision, AuditLog
from backend.app.schemas.pydantic_models import APIResponse, ManualVerificationRequest

router = APIRouter(prefix="/pages", tags=["pages"])

@router.get("/{page_id}/results", response_model=APIResponse)
def get_page_results(page_id: int, db: Session = Depends(get_db)):
    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    fields = db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).all()
    field_map = {}
    for f in fields:
        field_map[f.field_name] = {
            "raw_value": f.raw_value,
            "normalized_value": f.normalized_value,
            "source": f.source,
            "confidence": f.confidence,
            "status": f.status
        }

    decisions = db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).all()
    dec_list = []
    for d in decisions:
        dec_list.append({
            "field_name": d.field_name,
            "gemini_value": d.gemini_value,
            "cnn_value": d.cnn_value,
            "final_value": d.final_value,
            "decision": d.decision,
            "confidence": d.confidence,
            "reason": d.reason
        })

    return APIResponse(data={
        "page_id": page.id,
        "document_id": page.document_id,
        "page_number": page.page_number,
        "status": page.status,
        "original_image_path": page.original_image_path,
        "aligned_image_path": page.aligned_image_path,
        "fields": field_map,
        "verification_decisions": dec_list
    })

@router.get("/{page_id}/comparison", response_model=APIResponse)
def get_page_comparison(page_id: int, db: Session = Depends(get_db)):
    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    decisions = db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).all()
    comp_list = []
    for d in decisions:
        comp_list.append({
            "field_name": d.field_name,
            "gemini": d.gemini_value,
            "cnn": d.cnn_value,
            "final": d.final_value,
            "confidence": d.confidence,
            "decision": d.decision,
            "reason": d.reason
        })
    return APIResponse(data={"page_id": page.id, "comparisons": comp_list})

@router.post("/{page_id}/verify", response_model=APIResponse)
def verify_page_field(
    page_id: int, req: ManualVerificationRequest, db: Session = Depends(get_db)
):
    """
    Submits manual reviewer correction for a field. Updates FieldExtraction and records AuditLog.
    """
    page = db.query(Page).filter(Page.id == page_id).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    fe = db.query(FieldExtraction).filter(
        FieldExtraction.page_id == page.id, FieldExtraction.field_name == req.field_name
    ).first()

    if not fe:
        fe = FieldExtraction(
            page_id=page.id,
            field_name=req.field_name,
            raw_value=req.corrected_value,
            normalized_value=req.corrected_value,
            source="human",
            confidence=1.0,
            status="VERIFIED"
        )
        db.add(fe)
    else:
        fe.normalized_value = req.corrected_value
        fe.source = "human"
        fe.confidence = 1.0
        fe.status = "VERIFIED"

    vd = db.query(VerificationDecision).filter(
        VerificationDecision.page_id == page.id, VerificationDecision.field_name == req.field_name
    ).first()
    if vd:
        vd.final_value = req.corrected_value
        vd.decision = "HUMAN_VERIFIED"
        vd.reviewer_id = req.operator
        vd.reason = f"Manually verified by {req.operator}"

    audit = AuditLog(
        entity_type="field_extraction",
        entity_id=str(page.id),
        action="MANUAL_VERIFICATION",
        operator=req.operator,
        details_json={
            "field_name": req.field_name,
            "corrected_value": req.corrected_value,
            "reason": req.reason
        }
    )
    db.add(audit)
    db.commit()

    return APIResponse(data={
        "message": f"Field '{req.field_name}' successfully verified.",
        "page_id": page.id,
        "field_name": req.field_name,
        "final_value": req.corrected_value
    })
