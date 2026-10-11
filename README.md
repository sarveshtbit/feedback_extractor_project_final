# Bannari Amman Institute of Technology — Career Guidance Form Extraction & Verification Platform

A production-oriented, modular, and runnable end-to-end handwritten form extraction and verification platform for fixed-layout feedback forms.

---

## 1. Project Overview

This platform processes scanned copies, images, and multi-page PDFs of the fixed feedback form:
**BANNARI AMMAN INSTITUTE OF TECHNOLOGY CAREER GUIDANCE PROGRAM – FEEDBACK**

It employs a **Dual-Engine Recognition System**:
1. **Gemini Vision Model**: Primary general-purpose extraction engine for ALL form fields (Student Name, School Name, Date, Student Mobile, Parent Mobile, Email, Cut Off Marks, and Questionnaire Q1–Q4 options).
2. **PyTorch ResNet-18 CNN**: Independent verification engine for **Student Mobile Number** (10 handwritten digit boxes) and **Cut Off Marks**.
3. **Evidence-Based Decision Engine**: Character-level conflict detection, multi-view image variant evaluation, deterministic field validation, and conflict resolution states (`AGREEMENT`, `CONFLICT`, `MANUAL_REVIEW`, `CNN_UNAVAILABLE`).

---

## 2. System Architecture

```
                       USER
                        |
                        v
                  REACT FRONTEND
                        |
                        v
                   FASTAPI API
                        |
                        v
                  JOB CREATION
                        |
            +-----------+-----------+
            |                       |
            v                       v
      Image/PDF handling       Database
            |
            v
      Page extraction
            |
            v
     Image preprocessing
            |
            v
     Fixed form alignment
            |
            v
      Field/ROI extraction
            |
     +------+----------------+
     |                       |
     v                       v
GEMINI EXTRACTION       CNN VERIFICATION
     |                       |
     |                  Phone + Cutoff
     |                       |
     +-----------+-----------+
                 |
                 v
          COMPARISON ENGINE
                 |
                 v
        VALIDATION ENGINE
                 |
                 v
        CONFLICT RESOLUTION
                 |
                 v
          FINAL RESULTS
                 |
                 v
              DATABASE
                 |
                 v
             FRONTEND
```

---

## 3. Technology Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React icons, Axios.
- **Backend**: Python 3.10+, FastAPI, Pydantic, SQLAlchemy 2.0, Alembic.
- **Database**: PostgreSQL (Production) / SQLite (Local development/testing).
- **Machine Learning**: PyTorch, torchvision, ResNet-18, OpenCV, NumPy, Pillow, scikit-learn.
- **Gemini Vision**: Official Google GenAI Python SDK (`google-genai`), model configurable via `GEMINI_MODEL`.
- **PDF Engine**: PyMuPDF (`fitz`).
- **Task Queue**: Redis + Celery / FastAPI background tasks fallback.
- **Containerization**: Docker & Docker Compose.

---

## 4. Directory Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/v1/          # FastAPI routers (documents, pages, models, training, health, metrics)
│   │   ├── core/            # Configuration & Settings
│   │   ├── db/              # SQLAlchemy session & Base model
│   │   ├── decision_engine/ # Evidence-based comparison engine
│   │   ├── ml/              # ResNet-18 architecture, inference, training, evaluation
│   │   ├── models/          # SQLAlchemy domain models
│   │   ├── schemas/         # Pydantic request/response models
│   │   ├── services/        # OpenCV preprocessor, homography aligner, ROI crop, PyMuPDF, Gemini
│   │   ├── validation/      # Field validator rules
│   │   ├── workers/         # Celery tasks
│   │   └── main.py          # Application entrypoint
│   ├── alembic/             # DB migration scripts
│   ├── tests/               # Pytest suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/                 # React components, pages, services, types
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
├── config/
│   ├── form_template.yaml   # Normalized coordinates template
│   └── model_config.yaml  # ML & pipeline configuration
├── storage/                 # Uploaded files & cropped images
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 5. Local Setup & Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (Optional)

### Step 1: Clone & Configure Environment
```bash
cp .env.example .env
```
Edit `.env` to provide your `GEMINI_API_KEY` (optional for local mock mode).

### Step 2: Install Backend Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 3: Run Database Migrations & Initial Setup
```bash
alembic -c backend/alembic.ini upgrade head
```

### Step 4: Run Backend Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```
API Documentation available at: `http://127.0.0.1:8000/docs`

### Step 5: Install & Run Frontend
```bash
cd frontend
npm install
npm run dev
```
Access Frontend Dashboard at: `http://localhost:3000`

---

## 6. Running with Docker Compose

Run the complete multi-container stack (Backend, Frontend, Celery Worker, PostgreSQL, Redis):
```bash
docker-compose up --build
```

---

## 7. CNN Model Training & Dataset Instructions

### Untrained Mode Safety
If no trained CNN weights exist at `CNN_MODEL_PATH`, the system boots up in `NOT_TRAINED` state without crashing. Gemini extractions continue normally, displaying "CNN verification unavailable until a trained model is provided".

### Dataset Format
Datasets should be organized as a zip file containing digit folders `0` to `9`:
```
datasets/digits/
  0/
  1/
  ...
  9/
  metadata.csv (optional: image_path, label, writer_id, source)
```

### Writer-Independent Splitting
To evaluate generalization to unseen handwriting styles, samples are grouped by `writer_id` before splitting into **70% Train / 15% Validation / 15% Test**. No writer appears in more than one split.

### Triggering Training via CLI
```bash
python -m backend.app.ml.training.trainer --dataset ./ml/datasets/digits --epochs 30
```
Or use the **Model Training UI** in the frontend dashboard.

---

## 8. Running Automated Tests

Run the backend pytest suite covering API endpoints, database schemas, field validators, and decision engine cases:
```bash
pytest backend/tests -v
```

---

## 9. Security & Production Deployment

- Secrets are managed strictly via environment variables.
- Gemini API calls are made backend-only; API keys are never exposed to the browser.
- File upload types (`.png`, `.jpg`, `.jpeg`, `.pdf`) and size limits are strictly enforced.
- Path traversal protection is active on storage file paths.
