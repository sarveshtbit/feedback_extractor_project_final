import os
import cv2
import numpy as np
import pytest
from backend.app.services.alignment import FormAligner
from backend.app.services.preprocessing import ImagePreprocessor
from backend.app.services.roi_extractor import ROIExtractor

def test_homography_validation_rejects_bad_matrix():
    aligner = FormAligner()
    # Bad degenerate homography matrix
    H_bad = np.array([
        [0.001, 0.0, 10.0],
        [0.0, 0.0001, 20.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    src_pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [2000, 0], [2000, 3000], [0, 3000]], dtype=np.float32)

    assert aligner.is_valid_homography(H_bad, src_pts, dst_pts) is False

def test_letterbox_fallback_preserves_aspect_ratio():
    aligner = FormAligner()
    # Create test image 1000 x 500 (2:1 aspect ratio)
    img = np.zeros((500, 1000, 3), dtype=np.uint8)
    
    letterboxed = aligner.letterbox_to_canvas(img)
    assert letterboxed.shape == (3508, 2479, 3)  # Target canvas dimensions

    # Verify scale calculation: scale = min(2479/1000, 3508/500) = 2.479
    # Scaled image size: 2479 x 1239
    # Padded top/bottom = (3508 - 1239)//2 = 1134
    # Check that padded area outside centered image is white (255, 255, 255)
    assert np.all(letterboxed[10, 10] == [255, 255, 255])

def test_align_with_unreliable_markers_uses_fallback():
    aligner = FormAligner()
    # Plain image with no corner markers
    img = np.full((1200, 800, 3), 200, dtype=np.uint8)
    aligned, H, is_warped = aligner.align(img)

    assert aligned.shape == (3508, 2479, 3)
    assert is_warped is False  # Fallback triggered, no destructive homography warp
    assert H is None

def test_draw_roi_overlay_generates_debug_image():
    img = np.full((3508, 2479, 3), 255, dtype=np.uint8)
    overlay = ImagePreprocessor.draw_roi_overlay(img)

    assert overlay.shape == (3508, 2479, 3)
    # Overlay contains bounding boxes (non-white pixels)
    assert not np.array_equal(overlay, img)

def test_save_debug_stages(tmp_path):
    output_dir = str(tmp_path / "debug_stages")
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    stages = {
        "01_original": img,
        "02_orientation_corrected": img,
        "03_deskewed": img,
        "04_perspective_corrected": None,
        "05_final_preprocessed": img,
        "06_ROI_overlay": img
    }

    saved = ImagePreprocessor.save_debug_stages(output_dir, stages)
    assert "01_original" in saved
    assert "04_perspective_corrected" not in saved  # Omitted when unwarped
    assert "06_ROI_overlay" in saved
    assert os.path.exists(saved["01_original"])
    assert os.path.exists(saved["06_ROI_overlay"])
