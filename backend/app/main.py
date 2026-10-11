import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.db.session import engine
from backend.app.db.base import Base

# Import models to register tables
from backend.app.models.domain import (
    Document, Page, ExtractionJob, FieldExtraction, GeminiResult,
    CNNResult, VerificationDecision, ModelVersion, TrainingRun, DatasetVersion, AuditLog
)

# Import routers
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.pages import router as pages_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.training import router as training_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.metrics import router as metrics_router

# Create database tables automatically
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Production Handwritten Form Extraction & Verification Platform API"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
os.makedirs(settings.STORAGE_PATH, exist_ok=True)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_PATH), name="storage")

if os.path.exists("./frontend/dist"):
    app.mount("/app", StaticFiles(directory="./frontend/dist", html=True), name="frontend")

# Include Routers
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(metrics_router, prefix=settings.API_V1_STR)
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(pages_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(training_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
        "metrics": f"{settings.API_V1_STR}/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
