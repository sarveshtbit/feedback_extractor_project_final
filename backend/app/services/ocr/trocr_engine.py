"""
TrOCR Handwritten Recognition Engine.

This module encapsulates a reusable, lazy-loading Hugging Face TrOCR engine
specifically designed for recognizing handwritten text on cropped field regions
(e.g., student name, school name, date boxes, mobile numbers).

Design Principles:
1. Reusable Singleton: Model and processor are loaded once into memory on demand.
2. Crop-Only Input: TrOCR is strictly applied to localized field crops, never full forms.
3. Graceful Error Handling: Missing dependencies, unavailable weights, or failed downloads
   yield descriptive status dictionaries without crashing the application.
4. Confidence Documentation: Token probabilities are heuristic estimates and must NOT
   be interpreted as calibrated ground-truth accuracy.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class TrOCREngine:
    """
    Reusable handwritten text recognition engine based on Hugging Face TrOCR.
    """

    _instance: Optional["TrOCREngine"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "TrOCREngine":
        if cls._instance is None:
            cls._instance = super(TrOCREngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        checkpoint: str = "microsoft/trocr-base-handwritten",
        device: Optional[str] = None,
        max_crop_height: int = 1500,
        max_crop_width: int = 2400,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self.checkpoint: str = checkpoint
        self.max_crop_height: int = max_crop_height
        self.max_crop_width: int = max_crop_width

        # Model / Processor references
        self._processor: Any = None
        self._model: Any = None
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None

        # Determine target device (CPU baseline by default; CUDA if requested/available)
        self.device: str = device or self._detect_device()

        self._initialized = True
        logger.info(
            "TrOCREngine initialized (checkpoint: %s, device: %s, loaded: False)",
            self.checkpoint,
            self.device,
        )

    def _detect_device(self) -> str:
        """
        Detects default device (CPU by default; CUDA if torch detects GPU).
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Returns whether model weights and processor are loaded in memory."""
        return self._is_loaded

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the operational status and metadata of the TrOCR engine.
        """
        return {
            "engine": "TrOCR",
            "checkpoint": self.checkpoint,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "load_error": self._load_error,
            "max_crop_dimensions": {
                "max_height": self.max_crop_height,
                "max_width": self.max_crop_width,
            },
            "status": "READY" if self._is_loaded else ("ERROR" if self._load_error else "NOT_LOADED"),
        }

    def load_model(self) -> bool:
        """
        Explicitly loads processor and model weights into memory.
        Returns True if loaded successfully, False otherwise.
        """
        if self._is_loaded:
            return True

        try:
            import sys
            # Block sklearn import to avoid Windows Application Control policy blocking murmurhash.pyd
            sys.modules.setdefault("sklearn", None)
            sys.modules.setdefault("sklearn.metrics", None)

            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            logger.info("Loading TrOCR processor and model from checkpoint: %s...", self.checkpoint)
            start_t = time.time()

            self._processor = TrOCRProcessor.from_pretrained(self.checkpoint)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.checkpoint)
            self._model.to(self.device)
            self._model.eval()

            self._is_loaded = True
            self._load_error = None
            latency = (time.time() - start_t) * 1000.0
            logger.info("TrOCR engine loaded successfully in %.2f ms", latency)
            return True

        except ImportError as e:
            self._load_error = f"TrOCR dependencies not installed: {str(e)}"
            self._is_loaded = False
            logger.warning("[TrOCREngine] %s", self._load_error)
            return False
        except Exception as e:
            self._load_error = f"Failed to load TrOCR checkpoint '{self.checkpoint}': {str(e)}"
            self._is_loaded = False
            logger.error("[TrOCREngine] %s", self._load_error, exc_info=True)
            return False

    def validate_crop(self, crop: Optional[np.ndarray]) -> Tuple[bool, str]:
        """
        Validates that the input image is an acceptable cropped region and NOT a full form.
        """
        if crop is None:
            return False, "Input crop is None"
        if not isinstance(crop, np.ndarray):
            return False, f"Expected numpy.ndarray, got {type(crop).__name__}"
        if crop.size == 0:
            return False, "Input crop is empty (zero elements)"
        if len(crop.shape) not in (2, 3):
            return False, f"Invalid crop shape: {crop.shape}. Expected 2D or 3D array."

        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return False, f"Crop dimensions ({w}x{h}) are too small for OCR recognition (min 5x5)."

        # Reject full-page forms to enforce crop-only usage
        if h > self.max_crop_height or w > self.max_crop_width:
            return (
                False,
                f"Crop dimensions ({w}x{h}) exceed maximum allowed field size "
                f"({self.max_crop_width}x{self.max_crop_height}). TrOCR must only be run on cropped fields.",
            )

        return True, "Valid"

    def predict_crop(
        self,
        crop_image: np.ndarray,
        max_new_tokens: int = 64,
        return_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Recognizes handwritten text from a cropped field ROI.

        Parameters:
            crop_image: NumPy array (BGR, RGB, or Grayscale) of the cropped field.
            max_new_tokens: Maximum tokens to generate.
            return_confidence: Whether to compute token sequence score.

        Returns:
            Dict containing:
                - text (str or None): Recognized handwritten text.
                - confidence (float or None): Heuristic sequence confidence score [0.0, 1.0].
                - status (str): "SUCCESS", "INVALID_INPUT", "MODEL_UNAVAILABLE", or "INFERENCE_FAILED".
                - latency_ms (float): Inference execution time.
                - reasoning / notice: Documentation on confidence limitations.
        """
        start_time = time.time()

        # 1. Validate crop
        is_valid, validation_msg = self.validate_crop(crop_image)
        if not is_valid:
            return {
                "text": None,
                "confidence": None,
                "status": "INVALID_INPUT",
                "error": validation_msg,
                "latency_ms": round((time.time() - start_time) * 1000.0, 2),
                "engine": "TrOCR",
            }

        # 2. Lazy load model if needed
        if not self._is_loaded:
            loaded = self.load_model()
            if not loaded:
                return {
                    "text": None,
                    "confidence": None,
                    "status": "MODEL_UNAVAILABLE",
                    "error": self._load_error or "TrOCR model is not available",
                    "latency_ms": round((time.time() - start_time) * 1000.0, 2),
                    "engine": "TrOCR",
                }

        # 3. Preprocess crop
        try:
            import cv2
            import torch
            from PIL import Image

            if len(crop_image.shape) == 2:
                rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2RGB)
            elif crop_image.shape[2] == 4:
                rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGRA2RGB)
            elif crop_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = crop_image

            pil_img = Image.fromarray(rgb_image)
            pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # 4. Generate text with token scores
            with torch.no_grad():
                gen_outputs = self._model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = gen_outputs.sequences
            generated_text = self._processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()

            # 5. Compute heuristic confidence score from output token logits
            confidence_val: Optional[float] = None
            if return_confidence and hasattr(gen_outputs, "scores") and gen_outputs.scores:
                try:
                    # Softmax over logits per generated step
                    step_probs: List[float] = []
                    for step_idx, step_logits in enumerate(gen_outputs.scores):
                        if step_idx < len(sequences[0]) - 1:
                            token_id = sequences[0][step_idx + 1].item()
                            probs = torch.softmax(step_logits[0], dim=-1)
                            token_prob = probs[token_id].item()
                            step_probs.append(token_prob)

                    if step_probs:
                        # Geometric mean of token probabilities
                        log_sum = sum(np.log(max(p, 1e-6)) for p in step_probs)
                        confidence_val = float(np.exp(log_sum / len(step_probs)))
                        confidence_val = round(max(0.0, min(1.0, confidence_val)), 4)
                except Exception as conf_err:
                    logger.debug("Failed to calculate token confidence: %s", conf_err)
                    confidence_val = None

            latency_ms = (time.time() - start_time) * 1000.0

            return {
                "text": generated_text,
                "confidence": confidence_val,
                "status": "SUCCESS",
                "latency_ms": round(latency_ms, 2),
                "engine": "TrOCR",
                "checkpoint": self.checkpoint,
                "confidence_notice": (
                    "Token confidence is an uncalibrated heuristic derived from output logits; "
                    "do not treat as ground-truth accuracy."
                ),
            }

        except Exception as e:
            logger.error("[TrOCREngine] Inference failed: %s", e, exc_info=True)
            return {
                "text": None,
                "confidence": None,
                "status": "INFERENCE_FAILED",
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000.0, 2),
                "engine": "TrOCR",
            }
