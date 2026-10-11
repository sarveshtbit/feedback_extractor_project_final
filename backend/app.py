import os
import cv2
import json
import csv
import io
import datetime
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.database import engine, Base, get_db
from backend.models import FormRecord, MobileVerification, Feedback, Correction, ProcessingLog
from backend import crud

from cv_pipeline import CVPipeline
from guide_dot_removal import GuideDotRemover
from omr_detector import OMRDetector
from digit_classifier import DigitClassifier
from confidence_engine import ConfidenceEngine

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Bannari Amman Institute - Paper Form Digitization API",
    version="1.0.0",
    description="Phase 1 Paper Form Digitization & Verification Backend"
)

# Static directories
UPLOADS_DIR = "uploads"
CROPS_DIR = os.path.join(UPLOADS_DIR, "crops")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

app.mount("/crops", StaticFiles(directory=CROPS_DIR), name="crops")
if os.path.exists("frontend"):
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

# Initialize Pipeline Components
pipeline_cv = CVPipeline("form_config.json")
guide_remover = GuideDotRemover()
omr_detector = OMRDetector()
digit_classifier = DigitClassifier()
confidence_engine = ConfidenceEngine()

# Cache processing results in memory for quick UI serving
FORM_CACHE = {}

@app.post("/forms/upload")
async def upload_form(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads scanned form image/PDF, runs complete CV + ML pipeline, saves results & logs to DB.
    """
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp_str}_{file.filename}"
    file_path = os.path.join(UPLOADS_DIR, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Load image with OpenCV
    img = cv2.imread(file_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    # 1. CV Homography & Alignment
    try:
        warped, H = pipeline_cv.warp_and_align(img)
        rois = pipeline_cv.crop_field_rois(warped)
    except Exception as e:
        form_rec = crud.create_form_record(db, file_path, "FAILED_ALIGNMENT")
        crud.add_processing_log(db, form_rec.id, "MARKER_ALIGNMENT", "ERROR", {"error": str(e)})
        raise HTTPException(status_code=422, detail=f"Marker alignment failed: {str(e)}")

    # Create temporary crop output directory for this form ID
    temp_form_id = len(crud.get_all_forms(db)) + 1
    form_crop_dir = os.path.join(CROPS_DIR, str(temp_form_id))
    os.makedirs(form_crop_dir, exist_ok=True)

    # Save full warped image
    warped_path = os.path.join(form_crop_dir, "warped.png")
    cv2.imwrite(warped_path, warped)

    # 2. Guide Dot Removal + Digit/Char Classification (Mobile Number)
    mobile_preds = []
    mob_boxes = rois["mobile_number"]["boxes"]
    for b in mob_boxes:
        idx = b["index"]
        crop_raw = b["crop"]
        crop_clean, mask = guide_remover.process_roi(crop_raw)
        
        crop_filename = f"mobile_{idx}.png"
        cv2.imwrite(os.path.join(form_crop_dir, crop_filename), crop_clean)

        pred = digit_classifier.predict(crop_clean, is_digit_only=True)
        mobile_preds.append({
            "index": idx,
            "predicted_char": pred["predicted_char"],
            "confidence": pred["confidence"],
            "top_k": pred["top_k"],
            "crop_url": f"/crops/{temp_form_id}/{crop_filename}"
        })

    # 3. Student Name Classification
    name_preds = []
    name_boxes = rois["student_name"]["boxes"]
    for b in name_boxes:
        idx = b["index"]
        crop_clean, _ = guide_remover.process_roi(b["crop"])
        pred = digit_classifier.predict(crop_clean, is_digit_only=False)
        name_preds.append({
            "index": idx,
            "predicted_char": pred["predicted_char"],
            "confidence": pred["confidence"]
        })

    # 4. Cutoff Mark Classification
    cutoff_preds = []
    cutoff_boxes = rois["cutoff_mark"]["boxes"]
    for b in cutoff_boxes:
        idx = b["index"]
        b_type = b["type"]
        if b_type == "decimal":
            cutoff_preds.append({"index": idx, "type": "decimal", "predicted_char": ".", "confidence": 1.0})
        else:
            crop_clean, _ = guide_remover.process_roi(b["crop"])
            pred = digit_classifier.predict(crop_clean, is_digit_only=True)
            cutoff_preds.append({
                "index": idx,
                "type": "box",
                "predicted_char": pred["predicted_char"],
                "confidence": pred["confidence"]
            })

    # 5. School Name Classification
    sch_preds = []
    sch_boxes = rois["school_name"]["boxes"]
    for b in sch_boxes:
        idx = b["index"]
        crop_clean, _ = guide_remover.process_roi(b["crop"])
        pred = digit_classifier.predict(crop_clean, is_digit_only=False)
        sch_preds.append({
            "index": idx,
            "predicted_char": pred["predicted_char"],
            "confidence": pred["confidence"]
        })

    # 6. OMR Detection
    omr_results = omr_detector.process_all_questions(rois["omr_feedback"])

    # Save OMR bubble crops
    for q_id, q_data in rois["omr_feedback"]:
        for opt in q_data["options"]:
            label = opt["label"]
            crop_file = f"omr_{q_id}_{label}.png"
            cv2.imwrite(os.path.join(form_crop_dir, crop_file), opt["crop"])

    # 7. Confidence Engine Scoring & Validation
    mob_eval = confidence_engine.evaluate_mobile_number(mobile_preds)
    name_eval = confidence_engine.evaluate_name(name_preds)
    cutoff_eval = confidence_engine.evaluate_cutoff_mark(cutoff_preds)
    sch_eval = confidence_engine.evaluate_school_name(sch_preds)
    omr_eval = confidence_engine.evaluate_omr(omr_results)

    form_eval = confidence_engine.evaluate_form(mob_eval, name_eval, cutoff_eval, sch_eval, omr_eval)

    # 8. Save Form Record & Related Tables in DB
    form_rec = crud.create_form_record(db, file_path, form_eval["form_status"])

    mob_review_status = "AUTO_ACCEPT" if not mob_eval["needs_review"] else "PENDING_REVIEW"
    crud.add_mobile_verification(
        db, form_rec.id, mob_eval["value"], mob_eval["avg_confidence"], mob_eval["is_valid"], mob_review_status
    )

    omr_status = "CLEAN" if not omr_eval["needs_review"] else "NEEDS_REVIEW"
    crud.add_feedback(db, form_rec.id, omr_eval["value"], omr_status)

    crud.add_processing_log(db, form_rec.id, "HOMOGRAPHY", "SUCCESS", {"H": H.tolist()})
    crud.add_processing_log(db, form_rec.id, "CONFIDENCE_ENGINE", "SUCCESS", form_eval)

    # Save to memory cache for UI retrieval
    response_payload = {
        "form_id": form_rec.id,
        "status": form_eval["form_status"],
        "auto_accepted": form_eval["auto_accepted"],
        "warped_url": f"/crops/{temp_form_id}/warped.png",
        "fields": form_eval["fields"],
        "disclaimer": "not yet trained on real student handwriting, expect degraded accuracy until fine-tuned on real samples."
    }
    FORM_CACHE[form_rec.id] = response_payload

    return JSONResponse(status_code=200, content=response_payload)

@app.get("/forms/{form_id}")
async def get_form_details(form_id: int, db: Session = Depends(get_db)):
    """
    Retrieves form prediction details, field crops, and confidence levels.
    """
    if form_id in FORM_CACHE:
        return FORM_CACHE[form_id]

    form_rec = crud.get_form_by_id(db, form_id)
    if not form_rec:
        raise HTTPException(status_code=404, detail="Form record not found")

    mv = form_rec.mobile_verification
    fb = form_rec.feedback

    return {
        "form_id": form_rec.id,
        "status": form_rec.status,
        "created_at": form_rec.created_at.isoformat(),
        "mobile_verification": {
            "extracted_mobile": mv.extracted_mobile if mv else None,
            "verified_mobile": mv.verified_mobile if mv else None,
            "confidence": mv.confidence if mv else None,
            "is_valid": mv.is_valid if mv else False,
            "review_status": mv.review_status if mv else None
        },
        "feedback": json.loads(fb.omr_data_json) if fb else {}
    }

@app.post("/forms/{form_id}/verify")
async def verify_form_corrections(
    form_id: int,
    field_name: str = Form(...),
    corrected_value: str = Form(...),
    operator: str = Form("human_verifier"),
    db: Session = Depends(get_db)
):
    """
    Submits human corrections for uncertain or low-confidence fields.
    """
    form_rec = crud.get_form_by_id(db, form_id)
    if not form_rec:
        raise HTTPException(status_code=404, detail="Form record not found")

    original_val = None
    if form_id in FORM_CACHE and field_name in FORM_CACHE[form_id]["fields"]:
        original_val = FORM_CACHE[form_id]["fields"][field_name].get("value")

    correction = crud.record_correction(db, form_id, field_name, str(original_val), corrected_value, operator)
    crud.add_processing_log(db, form_id, "HUMAN_CORRECTION", "SUCCESS", {
        "field_name": field_name, "original": str(original_val), "corrected": corrected_value, "operator": operator
    })

    # Update memory cache
    if form_id in FORM_CACHE:
        FORM_CACHE[form_id]["fields"][field_name]["value"] = corrected_value
        FORM_CACHE[form_id]["fields"][field_name]["needs_review"] = False
        FORM_CACHE[form_id]["fields"][field_name]["confidence_level"] = "HUMAN_VERIFIED"
        FORM_CACHE[form_id]["status"] = "VERIFIED"

    return {
        "message": f"Successfully verified {field_name}",
        "form_id": form_id,
        "field_name": field_name,
        "corrected_value": corrected_value,
        "status": "VERIFIED"
    }

@app.get("/forms/export/csv")
async def export_forms_csv(db: Session = Depends(get_db)):
    """
    Exports all verified/processed student records to a downloadable CSV file.
    """
    forms = crud.get_all_forms(db)
    
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Form ID", "Status", "Mobile Number (Final)", "AI Extracted Mobile", 
        "Mobile Confidence", "Student Name", "Cutoff Mark", "School Name", 
        "OMR Feedback (Q1-Q5)", "Created At"
    ])

    for f in forms:
        mv = f.mobile_verification
        final_mobile = mv.verified_mobile if (mv and mv.verified_mobile) else (mv.extracted_mobile if mv else "")
        ai_mobile = mv.extracted_mobile if mv else ""
        mob_conf = f"{mv.confidence:.4f}" if mv else ""

        # Extract values from cache or defaults
        cached = FORM_CACHE.get(f.id, {})
        fields = cached.get("fields", {})
        
        name_val = fields.get("student_name", {}).get("value", "")
        cutoff_val = fields.get("cutoff_mark", {}).get("value", "")
        school_val = fields.get("school_name", {}).get("value", "")
        omr_val = json.dumps(fields.get("omr_feedback", {}).get("value", {}))

        writer.writerow([
            f.id, f.status, final_mobile, ai_mobile, mob_conf, name_val, cutoff_val, school_val, omr_val, f.created_at.isoformat()
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bannari_students_digitized.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
