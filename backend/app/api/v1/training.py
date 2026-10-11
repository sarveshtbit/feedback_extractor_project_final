import os
import uuid
import datetime
import zipfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.domain import TrainingRun, DatasetVersion, ModelVersion, AuditLog
from backend.app.schemas.pydantic_models import APIResponse, TrainingStartRequest
from backend.app.ml.training.dataset_manager import DatasetManager
from backend.app.ml.training.trainer import CNNModelTrainer

router = APIRouter(prefix="/training", tags=["training"])
DATASET_STORAGE_DIR = "./storage/datasets"
os.makedirs(DATASET_STORAGE_DIR, exist_ok=True)

@router.post("/datasets", response_model=APIResponse)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads zip dataset (0-9 folders or metadata.csv) for CNN model training.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Dataset must be uploaded as a .zip file")

    ds_id = str(uuid.uuid4())[:8]
    ds_name = f"dataset_{ds_id}"
    extract_dir = os.path.join(DATASET_STORAGE_DIR, ds_name)
    os.makedirs(extract_dir, exist_ok=True)

    zip_bytes = await file.read()
    with zipfile.ZipFile(zipfile.BytesIO(zip_bytes)) as z:
        z.extractall(extract_dir)

    dm = DatasetManager(extract_dir)
    data_info = dm.validate_and_parse_dataset()
    if not data_info.get("valid"):
        raise HTTPException(status_code=400, detail=data_info.get("error", "Invalid dataset format"))

    splits = dm.create_writer_independent_splits()

    ds_record = DatasetVersion(
        name=ds_name,
        total_samples=data_info["total_samples"],
        writer_count=data_info["writer_count"],
        split_info_json=splits["split_summary"]
    )
    db.add(ds_record)
    db.commit()

    return APIResponse(data={
        "dataset_name": ds_name,
        "total_samples": data_info["total_samples"],
        "writer_count": data_info["writer_count"],
        "class_distribution": data_info["class_distribution"],
        "split_summary": splits["split_summary"]
    })

@router.post("/start", response_model=APIResponse)
def start_training_run(req: TrainingStartRequest, db: Session = Depends(get_db)):
    """
    Triggers ResNet-18 model training pipeline.
    """
    run_id = str(uuid.uuid4())
    dataset_path = os.path.join(DATASET_STORAGE_DIR, req.dataset_version)
    if not os.path.exists(dataset_path):
        # Fallback to default demo dataset path if specific name not found
        dataset_path = "./ml/datasets/digits"

    output_model_path = f"./ml/models/resnet18_{run_id[:8]}.pth"

    run_record = TrainingRun(
        id=run_id,
        dataset_version=req.dataset_version,
        hyperparameters_json=req.model_dump(),
        status="RUNNING"
    )
    db.add(run_record)
    db.commit()

    try:
        trainer = CNNModelTrainer(
            dataset_dir=dataset_path,
            output_model_path=output_model_path,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.learning_rate,
            weight_decay=req.weight_decay
        )

        res = trainer.train()

        run_record.status = "COMPLETED"
        run_record.val_loss = res["best_val_loss"]
        run_record.completed_at = datetime.datetime.utcnow()

        # Register new ModelVersion
        version_name = f"digit-resnet18-v{run_id[:6]}"
        mv = ModelVersion(
            version_name=version_name,
            framework="pytorch",
            architecture="resnet18",
            weights_path=output_model_path,
            metrics_json={
                "val_loss": res["best_val_loss"],
                "split_summary": res["split_summary"]
            },
            is_active=False
        )
        db.add(mv)
        db.commit()

        return APIResponse(data={
            "training_id": run_id,
            "version_name": version_name,
            "status": "COMPLETED",
            "val_loss": res["best_val_loss"]
        })
    except Exception as e:
        run_record.status = "FAILED"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/{training_id}", response_model=APIResponse)
def get_training_status(training_id: str, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.id == training_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return APIResponse(data={
        "training_id": run.id,
        "dataset_version": run.dataset_version,
        "status": run.status,
        "val_loss": run.val_loss,
        "hyperparameters": run.hyperparameters_json,
        "created_at": run.created_at.isoformat()
    })
