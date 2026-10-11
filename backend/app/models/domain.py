import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from backend.app.db.base import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # image/png, image/jpeg, application/pdf
    total_pages = Column(Integer, default=1)
    status = Column(String, default="UPLOADED")  # UPLOADED, PROCESSING, COMPLETED, PARTIAL_FAILURE, FAILED
    storage_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    extraction_jobs = relationship("ExtractionJob", back_populates="document", cascade="all, delete-orphan")

class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    original_image_path = Column(String, nullable=False)
    aligned_image_path = Column(String, nullable=True)
    status = Column(String, default="PENDING")  # PENDING, PREPROCESSED, GEMINI_DONE, CNN_DONE, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="pages")
    field_extractions = relationship("FieldExtraction", back_populates="page", cascade="all, delete-orphan")
    gemini_result = relationship("GeminiResult", back_populates="page", uselist=False, cascade="all, delete-orphan")
    cnn_results = relationship("CNNResult", back_populates="page", cascade="all, delete-orphan")
    verification_decisions = relationship("VerificationDecision", back_populates="page", cascade="all, delete-orphan")

class ExtractionJob(Base):
    __tablename__ = "extraction_jobs"

    id = Column(String, primary_key=True, index=True)  # UUID string
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    job_type = Column(String, default="FORM_EXTRACTION")
    stage = Column(String, default="UPLOADED")  # UPLOADED, PDF_RENDERING, PREPROCESSING, GEMINI_EXTRACTION, CNN_EXTRACTION, COMPARING, VALIDATING, COMPLETED, FAILED
    progress = Column(Float, default=0.0)  # 0.0 to 100.0
    status = Column(String, default="PENDING")  # PENDING, RUNNING, COMPLETED, PARTIAL_FAILURE, FAILED
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="extraction_jobs")

class FieldExtraction(Base):
    __tablename__ = "field_extractions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    field_name = Column(String, nullable=False)
    raw_value = Column(Text, nullable=True)
    normalized_value = Column(Text, nullable=True)
    source = Column(String, nullable=False)  # gemini, cnn, decision_engine, human
    confidence = Column(Float, nullable=True)
    status = Column(String, default="EXTRACTED")  # EXTRACTED, VERIFIED, CONFLICT, MANUAL_REVIEW

    page = relationship("Page", back_populates="field_extractions")

class GeminiResult(Base):
    __tablename__ = "gemini_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False, unique=True)
    raw_json = Column(Text, nullable=False)
    model_name = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    page = relationship("Page", back_populates="gemini_result")

class CNNResult(Base):
    __tablename__ = "cnn_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    field_name = Column(String, nullable=False)  # student_mobile, cutoff_marks
    value = Column(String, nullable=True)
    per_character_results = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    model_version = Column(String, default="v1")
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    page = relationship("Page", back_populates="cnn_results")

class VerificationDecision(Base):
    __tablename__ = "verification_decisions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    field_name = Column(String, nullable=False)
    gemini_value = Column(Text, nullable=True)
    cnn_value = Column(Text, nullable=True)
    final_value = Column(Text, nullable=True)
    decision = Column(String, nullable=False)  # AGREEMENT, CNN_SELECTED, GEMINI_SELECTED, REVERIFIED, LOW_CONFIDENCE, MANUAL_REVIEW, CNN_UNAVAILABLE, INVALID_INPUT
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    reviewer_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    page = relationship("Page", back_populates="verification_decisions")

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version_name = Column(String, nullable=False, unique=True)  # e.g., digit-resnet18-v1
    framework = Column(String, default="pytorch")
    architecture = Column(String, default="resnet18")
    weights_path = Column(String, nullable=False)
    metrics_json = Column(JSON, nullable=True)  # accuracy, phone_exact_match, confusion_matrix
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True, index=True)  # UUID string
    dataset_version = Column(String, nullable=False)
    hyperparameters_json = Column(JSON, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    val_acc = Column(Float, nullable=True)
    test_acc = Column(Float, nullable=True)
    phone_exact_match_acc = Column(Float, nullable=True)
    cutoff_exact_match_acc = Column(Float, nullable=True)
    confusion_matrix_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    total_samples = Column(Integer, default=0)
    writer_count = Column(Integer, default=0)
    split_info_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    action = Column(String, nullable=False)
    operator = Column(String, default="system")
    details_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
