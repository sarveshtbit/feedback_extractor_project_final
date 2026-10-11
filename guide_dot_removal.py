import cv2
import numpy as np

class GuideDotRemover:
    def __init__(self):
        pass

    def remove_guide_dots_color(self, bgr_crop):
        """
        Color-threshold based guide-dot removal.
        Isolates black and dark blue ink while eliminating light gray/blue guide dots and light borders.
        """
        if len(bgr_crop.shape) < 3 or bgr_crop.shape[2] != 3:
            return self.remove_guide_dots_morphology(bgr_crop)

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Dark black ink: Value (brightness) < 110
        black_ink = v < 110

        # Dark blue ink: Hue around 100-130, Saturation > 50, Value < 200
        dark_blue_ink = (h >= 90) & (h <= 130) & (s > 40) & (v < 200)

        # Combined ink mask
        ink_mask = (black_ink | dark_blue_ink).astype(np.uint8) * 255

        # Small morphological closing to smooth ink strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)

        # Construct clean BGR image (white background with black ink)
        clean = np.full_like(bgr_crop, 255)
        clean[ink_mask > 0] = (0, 0, 0)
        return clean, ink_mask

    def remove_guide_dots_morphology(self, crop):
        """
        Grayscale morphological fallback.
        Removes small isolated noise/guide-dots using connected component size thresholding.
        """
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Adaptive thresholding to convert to binary mask (ink = 255, background = 0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )

        # Connected component analysis to remove small isolated dots (< 15 px)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        clean_mask = np.zeros_like(binary)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Keep components larger than 12 pixels (handwriting strokes)
            if area >= 12:
                clean_mask[labels == i] = 255

        clean_gray = np.full_like(gray, 255)
        clean_gray[clean_mask > 0] = 0

        clean_bgr = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
        return clean_bgr, clean_mask

    def process_roi(self, roi):
        """
        Processes ROI crop with color removal primary and morphological fallback.
        """
        try:
            return self.remove_guide_dots_color(roi)
        except Exception:
            return self.remove_guide_dots_morphology(roi)

if __name__ == "__main__":
    remover = GuideDotRemover()
    # Test on a dummy box
    dummy_box = np.full((140, 110, 3), 255, dtype=np.uint8)
    # Add light blue guide dot
    cv2.circle(dummy_box, (30, 30), 4, (240, 200, 180), -1)
    # Add dark black stroke
    cv2.line(dummy_box, (20, 20), (80, 120), (10, 10, 10), 5)

    clean_bgr, mask = remover.process_roi(dummy_box)
    print("Guide dot remover test complete! Mask sum:", np.sum(mask))
