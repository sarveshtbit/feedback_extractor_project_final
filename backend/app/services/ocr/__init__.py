"""
OCR Engine package for feedback form processing.

Provides modular engines for:
- PaddleOCR (Printed text & layout detection)
- TrOCR (Handwritten text recognition on cropped field ROIs)
- BubbleDetector (OpenCV response circle evaluation)
- FieldExtractor (Template coordinate routing & OCR orchestration)
"""

from backend.app.services.ocr.bubble_detector import BubbleDetector
from backend.app.services.ocr.field_extractor import FieldExtractor
from backend.app.services.ocr.paddle_engine import PaddleOCREngine
from backend.app.services.ocr.trocr_engine import TrOCREngine

__all__ = [
    "BubbleDetector",
    "FieldExtractor",
    "PaddleOCREngine",
    "TrOCREngine",
]
