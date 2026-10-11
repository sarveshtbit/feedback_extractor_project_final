from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.core.config import settings
from backend.app.ml.inference.cnn_service import CNNInferenceService
from backend.app.schemas.pydantic_models import APIResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=APIResponse)
def health_check(db: Session = Depends(get_db)):
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    cnn_service = CNNInferenceService()
    cnn_status = cnn_service.get_status()

    return APIResponse(data={
        "status": "healthy" if db_ok else "unhealthy",
        "database_connected": db_ok,
        "gemini_api_configured": bool(settings.GEMINI_API_KEY),
        "cnn_model": cnn_status
    })
