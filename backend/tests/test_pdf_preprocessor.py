import os
import tempfile
import cv2
import numpy as np
import pytest

from backend.app.services.ocr.pdf_preprocessor import PDFPreprocessor

def test_blank_image_detection():
    # Pure white image
    white_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    assert PDFPreprocessor.is_blank_image(white_img) is True

    # Image with dark content
    content_img = white_img.copy()
    cv2.circle(content_img, (500, 500), 50, (0, 0, 0), -1)
    assert PDFPreprocessor.is_blank_image(content_img) is False

    # Empty image
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    assert PDFPreprocessor.is_blank_image(empty_img) is True

def test_single_form_not_split():
    # A single portrait form (height 1500)
    img = np.full((1500, 1000, 3), 255, dtype=np.uint8)
    # Add content
    cv2.putText(img, "FEEDBACK FORM", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    spans, split_y = PDFPreprocessor.detect_form_splits(img)
    assert len(spans) == 1
    assert split_y is None
    assert spans[0] == (0, 1500)

def test_stacked_form_detected_and_split():
    # Tall 3300px image with top content, bottom content, and whitespace divider at y=1600
    img = np.full((3300, 2550, 3), 255, dtype=np.uint8)
    # Top form content
    cv2.rectangle(img, (200, 200), (2000, 1400), (0, 0, 0), 3)
    cv2.putText(img, "TOP FORM CONTENT", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    # Bottom form content
    cv2.rectangle(img, (200, 1800), (2000, 3000), (0, 0, 0), 3)
    cv2.putText(img, "BOTTOM FORM CONTENT", (300, 2100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    # y=1500..1700 is pure white divider

    spans, split_y = PDFPreprocessor.detect_form_splits(img)
    assert len(spans) == 2
    assert split_y is not None
    assert 1450 <= split_y <= 1750
    assert spans[0] == (0, split_y)
    assert spans[1] == (split_y, 3300)

def test_deskew_image():
    # Test deskew with slight rotation
    img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    # Draw horizontal lines
    for y in [200, 400, 600, 800]:
        cv2.line(img, (100, y), (900, y), (0, 0, 0), 3)
    
    # Rotate by 1.5 degrees
    M = cv2.getRotationMatrix2D((500, 500), 1.5, 1.0)
    rotated = cv2.warpAffine(img, M, (1000, 1000), borderValue=(255, 255, 255))
    
    deskewed, detected_skew = PDFPreprocessor.deskew_image(rotated)
    assert deskewed is not None
    assert deskewed.shape == img.shape
    # detected skew magnitude should be within ~0.5 deg of 1.5 deg
    assert abs(abs(detected_skew) - 1.5) < 0.5
    # Deskewed image should have minimal residual skew
    _, residual_skew = PDFPreprocessor.deskew_image(deskewed)
    assert abs(residual_skew) < 0.2

def test_process_rendered_page_preserves_ordering():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create stacked form
        img = np.full((3300, 2550, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (200, 200), (2000, 1400), (0, 0, 0), 3)
        cv2.rectangle(img, (200, 1800), (2000, 3000), (0, 0, 0), 3)
        
        results = PDFPreprocessor.process_rendered_page(img, page_num=1, output_dir=tmp_dir)
        assert len(results) == 2
        assert results[0]["page_number"] == 1
        assert results[0]["form_index"] == 1
        assert results[1]["page_number"] == 1
        assert results[1]["form_index"] == 2
        assert os.path.exists(results[0]["image_path"])
        assert os.path.exists(results[1]["image_path"])

def test_invalid_pdf_handling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        non_existent = os.path.join(tmp_dir, "non_existent.pdf")
        with pytest.raises(FileNotFoundError):
            PDFPreprocessor.process_pdf(non_existent, tmp_dir)
            
        # Corrupt file
        corrupt_path = os.path.join(tmp_dir, "corrupt.pdf")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a valid pdf header")
        with pytest.raises(ValueError):
            PDFPreprocessor.process_pdf(corrupt_path, tmp_dir)
