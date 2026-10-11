# Handwritten Digit Classification using CNN

An end-to-end Machine Learning and Deep Learning solution for recognizing 28×28 grayscale handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed via Streamlit.

---

## 1. Problem Statement

The goal of this project is to recognize handwritten digits from raw 28×28 pixel grayscale images. Given 784 pixel intensity values (range 0 to 255), the system classifies each image into one of 10 digit classes (`0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`).

---

## 2. Dataset Description

- **Training Set (`train.csv`)**: 42,000 rows, 785 columns (`label` target column + `pixel0` through `pixel783`).
- **Test Set (`test.csv`)**: 28,000 rows, 784 pixel columns (`pixel0` through `pixel783`).
- **Sample Submission (`sample_submission.csv`)**: 28,000 rows (`ImageId`, `Label`).

---

## 3. Exploratory Data Analysis (EDA)

The `train_cnn.py` script automatically performs EDA and exports visual charts to the `outputs/` folder:

- **`outputs/class_distribution.png`**: Class counts across digits 0–9 showing balanced representation (~4,000 samples per digit class).
- **`outputs/pixel_distribution.png`**: Histogram of raw pixel intensities showing strong bimodal distribution (mostly background pixels at 0 and foreground stroke pixels near 255).
- **`outputs/sample_digits.png`**: Grid display of representative handwritten samples from `train.csv`.
- **`outputs/average_digit.png`**: Average intensity template image for each digit (0 to 9).

---

## 4. Preprocessing Pipeline

1. **Feature-Label Separation**: `X` extracted from `pixel0`..`pixel783`, `y` extracted from `label`.
2. **Normalization**: Scaling pixel intensities from `[0, 255]` to `[0.0, 1.0]` via division by `255.0`.
3. **Tensor Reshaping**: Reshaping flat 784-dimensional vectors into 3D image tensors of shape `(28, 28, 1)`.
4. **Stratified Split**: 80% training split (33,600 samples) and 20% validation split (8,400 samples) with fixed seed `random_state=42`.
5. **Test Set Preservation**: 28,000 test images normalized and reshaped to `(28, 28, 1)` without label leaks.

---

## 5. Feature Engineering & Augmentation

Deep Convolutional Networks automatically extract spatial hierarchical features:
- **Block 1**: Low-level edge and orientation detectors.
- **Block 2**: Mid-level stroke curves, corners, and loops.
- **Block 3**: High-level semantic digit shapes.

To enhance model robustness against variations in handwriting styles, online **Data Augmentation** is integrated directly into the Keras model:
- `RandomRotation(0.1)` (rotations up to ±10%)
- `RandomTranslation(0.1, 0.1)` (horizontal & vertical shifts up to ±10%)
- `RandomZoom(0.1)` (zoom variations up to ±10%)

---

## 6. CNN Architecture

```text
Input Layer (28 x 28 x 1)
   │
Data Augmentation (RandomRotation, RandomTranslation, RandomZoom)
   │
[Block 1] Conv2D(32, 3x3, same) ➔ BatchNorm ➔ ReLU ➔ Conv2D(32, 3x3, same) ➔ ReLU ➔ MaxPool(2x2) ➔ Dropout(0.25)
   │
[Block 2] Conv2D(64, 3x3, same) ➔ BatchNorm ➔ ReLU ➔ Conv2D(64, 3x3, same) ➔ ReLU ➔ MaxPool(2x2) ➔ Dropout(0.30)
   │
[Block 3] Conv2D(128, 3x3, same) ➔ BatchNorm ➔ ReLU ➔ MaxPool(2x2) ➔ Dropout(0.30)
   │
[Classification Head] Flatten ➔ Dense(128) ➔ BatchNorm ➔ ReLU ➔ Dropout(0.50) ➔ Dense(10, Softmax)
```

---

## 7. Model Training

- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: `sparse_categorical_crossentropy`
- **Metric**: `accuracy`
- **Batch Size**: 128 (with auto-fallback to 64 if GPU/CPU memory limit is reached)
- **Epochs**: 15 max
- **Callbacks**:
  - `ModelCheckpoint`: Saves best model based on validation accuracy to `outputs/digit_cnn.keras`.
  - `EarlyStopping`: Monitors `val_accuracy`, `patience=4`, restores best weights.
  - `ReduceLROnPlateau`: Monitors `val_loss`, `factor=0.5`, `patience=2`, `min_lr=1e-6`.

---

## 8. Evaluation & Visualization

After training completes, the model is evaluated on the 8,400 validation images:

- **Validation Loss & Accuracy**: Evaluated directly from the saved best checkpoint.
- **Precision, Recall, F1-Score**: Detailed per-class breakdown saved to `outputs/classification_report.csv`.
- **Confusion Matrix**: Heatmap visualization saved to `outputs/confusion_matrix.png`.
- **Training Curves**: Loss and Accuracy epoch progressions saved to `outputs/accuracy_curve.png` and `outputs/loss_curve.png`.
- **Misclassified Digits**: Inspection grid of misclassified images with True vs Predicted labels saved to `outputs/misclassified_digits.png`.

---

## 9. Test Predictions (`submission.csv`)

- The trained CNN predicts labels for all 28,000 unlabeled test images in `test.csv`.
- Predictions are generated using `argmax(probabilities, axis=1)`.
- Results are saved to `submission.csv` adhering to Kaggle specifications (`ImageId`: 1 to 28000, `Label`: predicted digit integer 0..9).

---

## 10. Streamlit Web Application

An interactive web application built with Streamlit (`app.py`) allows users to upload custom handwritten digit images.

### Key Application Features:
- **Title**: "Handwritten Digit Recognizer - CNN"
- **Cached Loading**: Uses `@st.cache_resource` for efficient model loading from `outputs/digit_cnn.keras`.
- **Robust Preprocessing Pipeline**:
  - Border-based background color estimation.
  - Automatic contrast inversion (supports both black ink on white paper and white digits on dark backgrounds).
  - Digit bounding-box extraction with margin.
  - Aspect-ratio preserving scaling into a 20x20 bounding box.
  - Intensity-weighted center of mass centering on a 28x28 canvas.
  - Normalization to `[0, 1]` and reshaping to `(1, 28, 28, 1)`.
- **UI Sections**: Original image, preprocessed 28x28 image, predicted digit, confidence score, and interactive 10-class probability bar chart.

---

## 11. How to Run the Project

### Environment Setup

Activate your Python virtual environment:

```bash
# Windows
.\venv\Scripts\activate
```

Install requirements:

```bash
python -m pip install -r requirements.txt
```

### Step 1: Train Model & Generate Outputs

Run the complete training, evaluation, and submission pipeline:

```bash
python train_cnn.py
```

Trained model output location:
```text
outputs/digit_cnn.keras
```

### Step 2: Run Streamlit Application

Launch the interactive web UI:

```bash
python -m streamlit run app.py
```

---

## 12. Generated Artifacts Summary

```text
outputs/
├── digit_cnn.keras            # Best trained CNN model checkpoint
├── metadata.json              # Training & evaluation metric metadata
├── class_distribution.png     # Class count plot
├── pixel_distribution.png     # Raw pixel value histogram
├── sample_digits.png          # Grid of sample training digits
├── average_digit.png          # Average pixel templates per digit
├── accuracy_curve.png         # Training vs validation accuracy plot
├── loss_curve.png             # Training vs validation loss plot
├── confusion_matrix.png       # 10x10 validation confusion matrix heatmap
├── classification_report.csv  # Precision, recall, and F1 per class CSV
└── misclassified_digits.png   # Visualization of error cases
submission.csv                 # Kaggle test set predictions (28,000 rows)
```
