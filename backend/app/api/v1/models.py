from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.services.ocr_service import OCRService
from backend.app.models.domain import ModelVersion
from backend.app.schemas.pydantic_models import APIResponse

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/cnn/status", response_model=APIResponse)
def get_cnn_status():
    """
    Returns pipeline engine status for the frontend Model Status page.
    Preserves exact ModelStatusSchema shape.
    """
    ocr_service = OCRService()
    ocr_status = ocr_service.get_status()
    trocr_ready = ocr_status.get("trocr", {}).get("is_loaded", False)

    status_info = {
        "status": "READY" if trocr_ready else "NOT_TRAINED",
        "weights_exist": trocr_ready,
        "architecture": "PaddleOCR + TrOCR + OpenCV Bubble Detector",
        "notice": "Modular OCR Extraction Pipeline (TrOCR + PaddleOCR + OpenCV Bubble Detector) active."
    }
    return APIResponse(data=status_info)

@router.get("/ocr/status", response_model=APIResponse)
def get_ocr_status():
    """
    Detailed OCR pipeline subsystem diagnostic status.
    """
    ocr_service = OCRService()
    return APIResponse(data=ocr_service.get_status())

@router.get("", response_model=APIResponse)
def list_model_versions(db: Session = Depends(get_db)):
    versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    out = []
    for v in versions:
        out.append({
            "id": v.id,
            "version_name": v.version_name,
            "architecture": v.architecture,
            "is_active": v.is_active,
            "metrics": v.metrics_json,
            "created_at": v.created_at.isoformat()
        })
    return APIResponse(data=out)
