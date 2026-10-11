import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from backend.app.ml.models.base import DigitRecognizer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class ResNet18DigitModel(nn.Module):
        def __init__(self, num_classes: int = 10, pretrained: bool = True):
            super(ResNet18DigitModel, self).__init__()
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )

        def forward(self, x):
            return self.backbone(x)
else:
    class ResNet18DigitModel:
        pass

class ResNet18Recognizer(DigitRecognizer):
    def __init__(self, weights_path: Optional[str] = None, num_classes: int = 10):
        self.weights_loaded = False
        self.num_classes = num_classes

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ResNet18DigitModel(num_classes=num_classes, pretrained=True).to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if weights_path and os.path.exists(weights_path):
                self.load_weights(weights_path)

    def load_weights(self, weights_path: str) -> bool:
        if not TORCH_AVAILABLE or not os.path.exists(weights_path):
            self.weights_loaded = False
            return False
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.weights_loaded = True
            return True
        except Exception:
            self.weights_loaded = False
            return False

    def predict_digit_crop(self, crop: np.ndarray) -> Dict[str, Any]:
        if not TORCH_AVAILABLE or not self.weights_loaded:
            return {
                "digit": None,
                "confidence": 0.0,
                "probabilities": {},
                "status": "NOT_TRAINED"
            }

        if len(crop.shape) == 2:
            pil_img = Image.fromarray(crop).convert("RGB")
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_class_idx = int(np.argmax(probs))
        max_prob = float(probs[pred_class_idx])
        prob_dict = {str(i): float(round(p, 4)) for i, p in enumerate(probs)}
        digit_str = str(pred_class_idx) if pred_class_idx < 10 else "."

        return {
            "digit": digit_str,
            "confidence": round(max_prob, 4),
            "probabilities": prob_dict,
            "status": "READY"
        }
