import json
from sqlalchemy.orm import Session
from backend.models import FormRecord, MobileVerification, Feedback, Correction, ProcessingLog

def create_form_record(db: Session, file_path: str, status: str) -> FormRecord:
    form = FormRecord(file_path=file_path, status=status)
    db.add(form)
    db.commit()
    db.refresh(form)
    return form

def add_mobile_verification(db: Session, form_id: int, extracted: str, confidence: float, is_valid: bool, review_status: str):
    mv = MobileVerification(
        form_id=form_id,
        extracted_mobile=extracted,
        verified_mobile=extracted if review_status == "AUTO_ACCEPT" else None,
        confidence=confidence,
        is_valid=is_valid,
        review_status=review_status
    )
    db.add(mv)
    db.commit()
    return mv

def add_feedback(db: Session, form_id: int, omr_data: dict, omr_status: str):
    fb = Feedback(
        form_id=form_id,
        omr_data_json=json.dumps(omr_data),
        omr_status=omr_status
    )
    db.add(fb)
    db.commit()
    return fb

def add_processing_log(db: Session, form_id: int, step: str, status: str, details: dict = None):
    log = ProcessingLog(
        form_id=form_id,
        step=step,
        status=status,
        details_json=json.dumps(details) if details else None
    )
    db.add(log)
    db.commit()
    return log

def record_correction(db: Session, form_id: int, field_name: str, original_val: str, corrected_val: str, operator: str = "human_operator"):
    corr = Correction(
        form_id=form_id,
        field_name=field_name,
        original_value=original_val,
        corrected_value=corrected_val,
        corrected_by=operator
    )
    db.add(corr)
    
    # Update mobile_verification table if field is mobile_number
    if field_name == "mobile_number":
        mv = db.query(MobileVerification).filter(MobileVerification.form_id == form_id).first()
        if mv:
            mv.verified_mobile = corrected_val
            mv.review_status = "HUMAN_VERIFIED"

    # Update form status to VERIFIED
    form = db.query(FormRecord).filter(FormRecord.id == form_id).first()
    if form:
        form.status = "VERIFIED"

    db.commit()
    return corr

def get_form_by_id(db: Session, form_id: int):
    return db.query(FormRecord).filter(FormRecord.id == form_id).first()

def get_all_forms(db: Session):
    return db.query(FormRecord).all()
