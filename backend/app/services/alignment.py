import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from backend.app.core.config import get_form_template_config

class FormAligner:
    def __init__(self, template_config: Optional[Dict[str, Any]] = None):
        self.config = template_config or get_form_template_config()
        canvas_cfg = self.config.get("canvas", {"width": 2479, "height": 3508})
        self.canvas_width = canvas_cfg.get("width", 2479)
        self.canvas_height = canvas_cfg.get("height", 3508)
        self.reference_template_path = "bannari_amman_form_template.png"
        self.ref_image = None
        if os.path.exists(self.reference_template_path):
            self.ref_image = cv2.imread(self.reference_template_path)

        reg = self.config.get("registration_markers", {})
        if reg:
            tl = reg["top_left"]
            tr = reg["top_right"]
            br = reg["bottom_right"]
            bl = reg["bottom_left"]
            
            if isinstance(tl, list) and len(tl) == 4 and tl[0] <= 1.0:
                tl_x = (tl[0] + tl[2]) / 2.0 * self.canvas_width
                tl_y = (tl[1] + tl[3]) / 2.0 * self.canvas_height
                tr_x = (tr[0] + tr[2]) / 2.0 * self.canvas_width
                tr_y = (tr[1] + tr[3]) / 2.0 * self.canvas_height
                br_x = (br[0] + br[2]) / 2.0 * self.canvas_width
                br_y = (br[1] + br[3]) / 2.0 * self.canvas_height
                bl_x = (bl[0] + bl[2]) / 2.0 * self.canvas_width
                bl_y = (bl[1] + bl[3]) / 2.0 * self.canvas_height
            else:
                tl_x, tl_y = reg["top_left"]["x"], reg["top_left"]["y"]
                tr_x, tr_y = reg["top_right"]["x"], reg["top_right"]["y"]
                br_x, br_y = reg["bottom_right"]["x"], reg["bottom_right"]["y"]
                bl_x, bl_y = reg["bottom_left"]["x"], reg["bottom_left"]["y"]

            self.dst_markers = np.array([
                [tl_x, tl_y],
                [tr_x, tr_y],
                [br_x, br_y],
                [bl_x, bl_y]
            ], dtype=np.float32)
        else:
            self.dst_markers = np.array([
                [140, 140],
                [2339, 140],
                [2339, 3368],
                [140, 3368]
            ], dtype=np.float32)

    def is_valid_homography(self, H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        if H is None or H.shape != (3, 3):
            return False

        det = np.linalg.det(H)
        if det <= 0.001 or det > 1000.0:
            return False

        try:
            pts_homo = np.hstack([src_pts, np.ones((4, 1))])
            transformed = (H @ pts_homo.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]

            src_w = np.linalg.norm(src_pts[1] - src_pts[0])
            dst_w = np.linalg.norm(transformed[1] - transformed[0])
            if src_w == 0 or not (0.2 <= (dst_w / src_w) <= 4.0):
                return False

            src_h = np.linalg.norm(src_pts[3] - src_pts[0])
            dst_h = np.linalg.norm(transformed[3] - transformed[0])
            if src_h == 0 or not (0.2 <= (dst_h / src_h) <= 4.0):
                return False

            return True
        except Exception:
            return False

    def detect_markers(self, image: np.ndarray) -> Optional[np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        img_h, img_w = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = (img_h * img_w) * 0.0001
        max_area = (img_h * img_w) * 0.015

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        candidates.append((cx, cy, area))

        if len(candidates) < 4:
            return None

        mid_x, mid_y = img_w / 2.0, img_h / 2.0
        quad_tl, quad_tr, quad_br, quad_bl = [], [], [], []

        for cx, cy, area in candidates:
            if cx < mid_x and cy < mid_y:
                quad_tl.append((cx, cy))
            elif cx >= mid_x and cy < mid_y:
                quad_tr.append((cx, cy))
            elif cx >= mid_x and cy >= mid_y:
                quad_br.append((cx, cy))
            elif cx < mid_x and cy >= mid_y:
                quad_bl.append((cx, cy))

        def best_corner(pts, tx, ty):
            if not pts:
                return None
            return min(pts, key=lambda p: (p[0] - tx)**2 + (p[1] - ty)**2)

        tl = best_corner(quad_tl, 0, 0)
        tr = best_corner(quad_tr, img_w, 0)
        br = best_corner(quad_br, img_w, img_h)
        bl = best_corner(quad_bl, 0, img_h)

        if None in (tl, tr, br, bl):
            return None

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def align_with_orb_features(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Feature-based homography alignment against reference template using ORB keypoint matching.
        """
        if self.ref_image is None:
            return None, None

        try:
            orb = cv2.ORB_create(3000)
            kp1, des1 = orb.detectAndCompute(image, None)
            kp2, des2 = orb.detectAndCompute(self.ref_image, None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return None, None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 15:
                return None, None

            top_matches = matches[:50]
            src_pts = np.float32([kp1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None and H.shape == (3, 3):
                warped = cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height))
                return warped, H
            return None, None
        except Exception:
            return None, None

    def letterbox_to_canvas(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.full((self.canvas_height, self.canvas_width, 3), 255, dtype=np.uint8)

        img_h, img_w = image.shape[:2]
        target_w, target_h = self.canvas_width, self.canvas_height

        scale = min(target_w / float(img_w), target_h / float(img_h))
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(image.shape) == 3:
            canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        else:
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas

    def align(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Aligns form image to template canvas (2479x3508).
        1. Tries 4-corner marker RANSAC perspective transformation.
        2. Tries ORB feature matching homography alignment against reference template.
        3. Fallback: Direct canvas resolution scaling (2479x3508).
        """
        try:
            src_pts = self.detect_markers(image)
            if src_pts is not None:
                H, _ = cv2.findHomography(src_pts, self.dst_markers, cv2.RANSAC, 5.0)
                if self.is_valid_homography(H, src_pts, self.dst_markers):
                    warped = cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height))
                    return warped, H, True

            # 2. Try ORB feature matching homography alignment
            orb_warped, orb_H = self.align_with_orb_features(image)
            if orb_warped is not None:
                return orb_warped, orb_H, True

            # 3. Fallback: Direct canvas resolution scaling
            scaled = self.letterbox_to_canvas(image)
            return scaled, None, False
        except Exception:
            scaled = self.letterbox_to_canvas(image)
            return scaled, None, False

