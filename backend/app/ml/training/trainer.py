import os
import time

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

from PIL import Image
from typing import Dict, List, Any, Optional

from backend.app.ml.models.resnet_digit import ResNet18DigitModel
from backend.app.ml.training.dataset_manager import DatasetManager

class DigitImageDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not TORCH_AVAILABLE:
            return None
        item = self.samples[idx]
        img_path = item["image_path"]
        label = item["label"]
        pil_img = Image.open(img_path).convert("RGB")
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, torch.tensor(label, dtype=torch.long)

class CNNModelTrainer:
    def __init__(
        self,
        dataset_dir: str,
        output_model_path: str = "./ml/models/pretrained_resnet18.pth",
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        patience: int = 7
    ):
        self.dataset_dir = dataset_dir
        self.output_model_path = output_model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def train(self) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return {
                "status": "FAILED",
                "error": "PyTorch / torchvision packages are not installed."
            }

        dm = DatasetManager(self.dataset_dir)
        splits = dm.create_writer_independent_splits()
        
        train_ds = DigitImageDataset(splits["train"], transform=self.train_transform)
        val_ds = DigitImageDataset(splits["val"], transform=self.val_transform) if splits["val"] else None

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False) if val_ds else None

        model = ResNet18DigitModel(num_classes=10, pretrained=True).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))

            if val_loader:
                model.eval()
                total_val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(self.device), labels.to(self.device)
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                        total_val_loss += loss.item()

                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                avg_val_loss = total_val_loss / max(1, len(val_loader))
                val_acc = correct / max(1, total)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), self.output_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            else:
                # Save model directly if no validation split exists
                torch.save(model.state_dict(), self.output_model_path)
                best_val_loss = 0.0

        return {
            "status": "COMPLETED",
            "best_val_loss": round(best_val_loss, 4),
            "output_model_path": self.output_model_path,
            "split_summary": splits["split_summary"],
            "reliability_notice": splits.get("reliability_notice", "INSUFFICIENT WRITERS FOR RELIABLE GENERALIZATION EVALUATION")
        }
