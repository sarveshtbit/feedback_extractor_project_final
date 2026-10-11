from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, List

class DigitRecognizer(ABC):
    @abstractmethod
    def load_weights(self, weights_path: str) -> bool:
        pass

    @abstractmethod
    def predict_digit_crop(self, crop: np.ndarray) -> Dict[str, Any]:
        """
        Predicts digit/character for a cropped numpy array.
        Returns:
        {
          "digit": "8",
          "confidence": 0.932,
          "probabilities": {"0": 0.001, ..., "8": 0.932, ...}
        }
        """
        pass
