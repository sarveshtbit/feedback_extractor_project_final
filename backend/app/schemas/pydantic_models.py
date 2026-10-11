from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import datetime

class APIResponse(BaseModel):
    success: bool = True
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    request_id: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))

class FieldResultSchema(BaseModel):
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    source: str
    confidence: float
    status: str
    gemini_value: Optional[str] = None
    cnn_value: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None

class VerificationDecisionSchema(BaseModel):
    field_name: str
    gemini_value: Optional[str] = None
    cnn_value: Optional[str] = None
    final_value: Optional[str] = None
    decision: str
    confidence: float
    reason: Optional[str] = None

class PageResultSchema(BaseModel):
    page_id: int
    page_number: int
    status: str
    original_image_url: str
    aligned_image_url: Optional[str] = None
    fields: Dict[str, FieldResultSchema] = {}
    verification_decisions: List[VerificationDecisionSchema] = []
    error_message: Optional[str] = None

class DocumentResponseSchema(BaseModel):
    document_id: int
    filename: str
    file_type: str
    total_pages: int
    status: str
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None
    pages: List[PageResultSchema] = []

class ExtractionJobStatusSchema(BaseModel):
    job_id: str
    document_id: int
    stage: str
    progress: float
    status: str
    error: Optional[str] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

class ManualVerificationRequest(BaseModel):
    field_name: str
    corrected_value: str
    operator: str = "human_verifier"
    reason: Optional[str] = None

class ModelStatusSchema(BaseModel):
    status: str  # READY, NOT_TRAINED, TRAINING
    active_version: Optional[str] = None
    architecture: str = "ResNet-18"
    weights_exist: bool = False
    metrics: Optional[Dict[str, Any]] = None
    notice: Optional[str] = None

class TrainingStartRequest(BaseModel):
    dataset_version: str = "default_v1"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    architecture: str = "ResNet-18"

class TrainingRunStatusSchema(BaseModel):
    training_id: str
    status: str
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    test_acc: Optional[float] = None
    phone_exact_match_acc: Optional[float] = None
    cutoff_exact_match_acc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
