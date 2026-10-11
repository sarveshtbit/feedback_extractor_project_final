import os
import argparse
from backend.app.ml.training.trainer import CNNModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 model on real handwritten digit dataset.")
    parser.add_argument("--dataset_dir", type=str, default="real_digit_dataset", help="Path to real digit dataset directory")
    parser.add_argument("--output_model_path", type=str, default="ml/models/pretrained_resnet18.pth", help="Path to save trained weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    args = parser.parse_args()

    print(f"Loading REAL dataset from: {args.dataset_dir}")
    print(f"Target model output path: {args.output_model_path}")
    print("Starting ResNet-18 Digit Model Training on Real Dataset...")

    trainer = CNNModelTrainer(
        dataset_dir=args.dataset_dir,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    result = trainer.train()

    # Also save a copy to backend/app/ml/models/pretrained_resnet18.pth if different
    secondary_path = "backend/app/ml/models/pretrained_resnet18.pth"
    if os.path.exists(args.output_model_path):
        os.makedirs(os.path.dirname(secondary_path), exist_ok=True)
        import shutil
        shutil.copy2(args.output_model_path, secondary_path)

    print("--- REAL DATASET TRAINING RESULT ---")
    print(result)

if __name__ == "__main__":
    main()

