import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# -----------------------------
# Configuration
# -----------------------------
RANDOM_STATE = 42
IMG_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 15

DATA_DIR = "."
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def main():
    print("=" * 60)
    print("1. LOADING DATA & INITIAL CHECKS")
    print("=" * 60)

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if os.path.exists(sample_sub_path):
        sample_sub_df = pd.read_csv(sample_sub_path)
    else:
        sample_sub_df = pd.DataFrame({"ImageId": range(1, len(test_df) + 1), "Label": 0})

    # Separate X and y
    X_raw = train_df.drop(columns=["label"]).values
    y_raw = train_df["label"].values
    X_test_raw = test_df.values

    # Check missing, duplicates, class distribution, pixel range
    missing_train = train_df.isnull().sum().sum()
    missing_test = test_df.isnull().sum().sum()
    duplicate_train = train_df.duplicated().sum()
    class_counts = pd.Series(y_raw).value_counts().sort_index()
    pixel_min, pixel_max = X_raw.min(), X_raw.max()
    pixel_mean, pixel_std = X_raw.mean(), X_raw.std()

    print(f"Train dimensions    : {train_df.shape}")
    print(f"Test dimensions     : {test_df.shape}")
    print(f"Sample Sub dimensions: {sample_sub_df.shape}")
    print(f"Missing values (train): {missing_train}")
    print(f"Missing values (test) : {missing_test}")
    print(f"Duplicate rows (train): {duplicate_train}")
    print(f"Pixel stats (raw)     : min={pixel_min}, max={pixel_max}, mean={pixel_mean:.2f}, std={pixel_std:.2f}")
    print("\nClass Counts:")
    for digit, count in class_counts.items():
        print(f"  Digit {digit}: {count} samples")

    print("\n" + "=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Class distribution plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title("Digit Class Distribution")
    plt.xlabel("Digit Label")
    plt.ylabel("Count")
    for idx, val in enumerate(class_counts.values):
        plt.text(idx, val + 100, str(val), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # 2. Pixel distribution plot
    plt.figure(figsize=(8, 5))
    plt.hist(X_raw.ravel(), bins=50, color="skyblue", edgecolor="black")
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Value (0 - 255)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pixel_distribution.png"), dpi=150)
    plt.close()

    # 3. Sample digits plot
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_raw[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {y_raw[i]}")
        ax.axis("off")
    plt.suptitle("Sample Handwritten Digits", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_digits.png"), dpi=150)
    plt.close()

    # 4. Average digit plot
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for digit in range(10):
        ax = axes.flat[digit]
        digit_mean = X_raw[y_raw == digit].mean(axis=0).reshape(28, 28)
        ax.imshow(digit_mean, cmap="gray")
        ax.set_title(f"Avg Digit {digit}")
        ax.axis("off")
    plt.suptitle("Average Digit Patterns (0-9)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_digit.png"), dpi=150)
    plt.close()

    print("Saved EDA plots to outputs/")

    print("\n" + "=" * 60)
    print("3. DATA PREPROCESSING")
    print("=" * 60)

    # Normalize pixel values
    X_norm = X_raw.astype("float32") / 255.0
    X_test_norm = X_test_raw.astype("float32") / 255.0

    # Reshape 784 -> 28 x 28 x 1
    X_reshaped = X_norm.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test_reshaped = X_test_norm.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Stratified train/validation split (test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_reshaped,
        y_raw,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_raw
    )

    print(f"Training split shape   : {X_train.shape}")
    print(f"Validation split shape : {X_val.shape}")
    print(f"Test data shape        : {X_test_reshaped.shape}")

    print("\n" + "=" * 60)
    print("4. CNN ARCHITECTURE")
    print("=" * 60)

    # Build model per requested specification
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Data augmentation
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),

        # Conv Block 1
        tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Conv Block 2
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.30),

        # Conv Block 3
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.30),

        # Classification Head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("\n" + "=" * 60)
    print("5. MODEL TRAINING")
    print("=" * 60)

    model_save_path = os.path.join(OUTPUT_DIR, "digit_cnn.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    current_batch_size = BATCH_SIZE
    try:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=current_batch_size,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"\n[Warning] Training with batch_size={current_batch_size} failed: {e}")
        current_batch_size = 64
        print(f"Retrying training with reduced batch_size={current_batch_size}...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=current_batch_size,
            callbacks=callbacks,
            verbose=1
        )

    print("\n" + "=" * 60)
    print("6. TRAINING CURVES")
    print("=" * 60)

    # Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="s")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()

    # Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss", marker="o")
    plt.plot(history.history["val_loss"], label="Validation Loss", marker="s")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    print("Saved training curves to outputs/")

    print("\n" + "=" * 60)
    print("7. EVALUATION ON VALIDATION SET")
    print("=" * 60)

    # Load best saved model
    print(f"Loading best saved model from: {model_save_path}")
    best_model = tf.keras.models.load_model(model_save_path)

    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)

    y_val_prob = best_model.predict(X_val, batch_size=current_batch_size, verbose=0)
    y_val_pred = np.argmax(y_val_prob, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average="macro")

    print(f"\nFinal Validation Loss    : {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Macro Precision          : {precision:.4f}")
    print(f"Macro Recall             : {recall:.4f}")
    print(f"Macro F1-Score           : {f1:.4f}")

    print("\nDetailed Classification Report:")
    clf_report_text = classification_report(y_val, y_val_pred, digits=4)
    print(clf_report_text)

    # Save classification report CSV
    report_dict = classification_report(y_val, y_val_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Inspect misclassified digits
    misclassified_indices = np.where(y_val_pred != y_val)[0]
    print(f"\nTotal misclassified digits in validation set: {len(misclassified_indices)} / {len(y_val)}")

    if len(misclassified_indices) > 0:
        num_to_plot = min(15, len(misclassified_indices))
        fig, axes = plt.subplots(3, 5, figsize=(12, 7))
        for i, ax in enumerate(axes.flat):
            if i < num_to_plot:
                idx = misclassified_indices[i]
                img = X_val[idx].reshape(28, 28)
                true_lbl = y_val[idx]
                pred_lbl = y_val_pred[idx]
                conf = y_val_prob[idx][pred_lbl] * 100

                ax.imshow(img, cmap="gray")
                ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl}\nConf: {conf:.1f}%", fontsize=9, color="red")
            ax.axis("off")
        plt.suptitle("Misclassified Digits (Validation Set)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "misclassified_digits.png"), dpi=150)
        plt.close()
        print("Saved misclassified_digits.png to outputs/")

    # Save metadata JSON
    metadata = {
        "image_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test_reshaped)),
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1)
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("8. TEST PREDICTIONS & SUBMISSION GENERATION")
    print("=" * 60)

    test_probs = best_model.predict(X_test_reshaped, batch_size=current_batch_size, verbose=1)
    test_preds = np.argmax(test_probs, axis=1)

    submission_df = pd.DataFrame({
        "ImageId": sample_sub_df["ImageId"],
        "Label": test_preds
    })

    sub_path = "submission.csv"
    submission_df.to_csv(sub_path, index=False)

    print(f"\nSaved submission to: {sub_path}")
    print("\n--- Submission Verification ---")
    print(f"Row count               : {len(submission_df)} (Expected: 28000)")
    print(f"Columns                 : {list(submission_df.columns)} (Expected: ['ImageId', 'Label'])")
    print(f"Null values in predictions: {submission_df.isnull().sum().sum()}")
    print(f"Label data type         : {submission_df['Label'].dtype}")
    print(f"Unique label values     : {sorted(submission_df['Label'].unique())}")

    assert len(submission_df) == 28000, "Submission row count mismatch!"
    assert list(submission_df.columns) == list(sample_sub_df.columns), "Submission columns mismatch!"
    assert submission_df.isnull().sum().sum() == 0, "Submission contains null values!"
    assert all(0 <= label <= 9 for label in submission_df["Label"]), "Invalid label range!"

    print("\nSubmission verification PASSED successfully.")
    print("=" * 60)
    print(f"TRAINING & EVALUATION COMPLETE. Best model saved at {model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
