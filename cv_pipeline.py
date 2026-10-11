import cv2
import json
import numpy as np
import os

class CVPipeline:
    def __init__(self, config_path="form_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.canvas_width = self.config["canvas"]["width"]
        self.canvas_height = self.config["canvas"]["height"]
        
        # Target marker centroids from single source of truth config
        reg = self.config["registration_markers"]
        self.dst_markers = np.array([
            [reg["top_left"]["x"], reg["top_left"]["y"]],
            [reg["top_right"]["x"], reg["top_right"]["y"]],
            [reg["bottom_right"]["x"], reg["bottom_right"]["y"]],
            [reg["bottom_left"]["x"], reg["bottom_left"]["y"]]
        ], dtype=np.float32)

    def detect_markers(self, image):
        """
        Detects 4 corner registration markers on the scanned/photographed form.
        Returns 4 detected centroid coordinates (TL, TR, BR, BL) as float32 array.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        img_h, img_w = gray.shape[:2]

        # Gaussian blur + Thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        min_area = (img_h * img_w) * 0.00005
        max_area = (img_h * img_w) * 0.02

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                
                # Check for 4-sided quadrilaterals / squares
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        candidates.append((cx, cy, area))

        if len(candidates) < 4:
            # Fallback using adaptive thresholding
            thresh_adapt = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
            )
            contours_adapt, _ = cv2.findContours(thresh_adapt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in contours_adapt:
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
            raise ValueError(f"Could not locate 4 registration markers (found {len(candidates)} candidates). Ensure form corners are visible.")

        # Classify candidate centers into 4 quadrants
        quad_tl, quad_tr, quad_br, quad_bl = [], [], [], []
        mid_x, mid_y = img_w / 2.0, img_h / 2.0

        for cx, cy, area in candidates:
            if cx < mid_x and cy < mid_y:
                quad_tl.append((cx, cy, area))
            elif cx >= mid_x and cy < mid_y:
                quad_tr.append((cx, cy, area))
            elif cx >= mid_x and cy >= mid_y:
                quad_br.append((cx, cy, area))
            elif cx < mid_x and cy >= mid_y:
                quad_bl.append((cx, cy, area))

        # Select closest to corner target in each quadrant
        def select_corner(quad_pts, target_x, target_y):
            if not quad_pts:
                return None
            best = min(quad_pts, key=lambda pt: (pt[0] - target_x)**2 + (pt[1] - target_y)**2)
            return [best[0], best[1]]

        tl = select_corner(quad_tl, 0, 0)
        tr = select_corner(quad_tr, img_w, 0)
        br = select_corner(quad_br, img_w, img_h)
        bl = select_corner(quad_bl, 0, img_h)

        if tl is None or tr is None or br is None or bl is None:
            raise ValueError("Failed to locate one or more corner markers in document quadrants.")

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        return src_pts

    def warp_and_align(self, image):
        """
        Detects markers, computes homography, and warps input image to reference 2479x3508 canvas.
        """
        src_pts = self.detect_markers(image)
        
        # Compute Homography matrix H
        H, _ = cv2.findHomography(src_pts, self.dst_markers, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Homography calculation failed.")

        # Warp perspective to exact template canvas dimensions
        warped = cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height))
        return warped, H

    def crop_field_rois(self, warped_image):
        """
        Crops field ROIs using pixel coordinates defined in form_config.json.
        Returns dictionary containing cropped image arrays per field and box/bubble.
        """
        cropped_data = {}
        fields_config = self.config["fields"]

        # 1. Mobile Number (Character Boxes)
        if "mobile_number" in fields_config:
            mob_cfg = fields_config["mobile_number"]
            digit_crops = []
            for b in mob_cfg["boxes"]:
                x, y, w, h = b["box"]
                crop = warped_image[y:y+h, x:x+w]
                digit_crops.append({
                    "index": b["index"],
                    "crop": crop,
                    "bbox": [x, y, w, h]
                })
            cropped_data["mobile_number"] = {
                "boxes": digit_crops
            }

        # 2. Student Name
        if "student_name" in fields_config:
            name_cfg = fields_config["student_name"]
            char_crops = []
            for b in name_cfg["boxes"]:
                x, y, w, h = b["box"]
                crop = warped_image[y:y+h, x:x+w]
                char_crops.append({
                    "index": b["index"],
                    "crop": crop,
                    "bbox": [x, y, w, h]
                })
            cropped_data["student_name"] = {
                "boxes": char_crops
            }

        # 3. Cutoff Mark
        if "cutoff_mark" in fields_config:
            cutoff_cfg = fields_config["cutoff_mark"]
            cutoff_crops = []
            for b in cutoff_cfg["boxes"]:
                x, y, w, h = b["box"]
                crop = warped_image[y:y+h, x:x+w]
                cutoff_crops.append({
                    "index": b["index"],
                    "type": b["type"],
                    "crop": crop,
                    "bbox": [x, y, w, h]
                })
            cropped_data["cutoff_mark"] = {
                "boxes": cutoff_crops
            }

        # 4. School Name
        if "school_name" in fields_config:
            sch_cfg = fields_config["school_name"]
            sch_crops = []
            for b in sch_cfg["boxes"]:
                x, y, w, h = b["box"]
                crop = warped_image[y:y+h, x:x+w]
                sch_crops.append({
                    "index": b["index"],
                    "crop": crop,
                    "bbox": [x, y, w, h]
                })
            cropped_data["school_name"] = {
                "boxes": sch_crops
            }

        # 5. OMR Feedback
        if "omr_feedback" in fields_config:
            omr_cfg = fields_config["omr_feedback"]
            omr_questions = []
            for q in omr_cfg["questions"]:
                q_opts = []
                for opt in q["options"]:
                    cx, cy = opt["center"]
                    r = opt["radius"]
                    # Crop bounding square around bubble
                    x1, y1 = max(0, cx - r - 5), max(0, cy - r - 5)
                    x2, y2 = min(self.canvas_width, cx + r + 5), min(self.canvas_height, cy + r + 5)
                    bubble_crop = warped_image[y1:y2, x1:x2]
                    q_opts.append({
                        "label": opt["label"],
                        "crop": bubble_crop,
                        "center": [cx, cy],
                        "radius": r
                    })
                omr_questions.append({
                    "question_id": q["question_id"],
                    "label": q["label"],
                    "options": q_opts
                })
            cropped_data["omr_feedback"] = omr_questions

        return cropped_data

if __name__ == "__main__":
    # Quick sanity test on template image
    pipeline = CVPipeline("form_config.json")
    test_img = cv2.imread("bannari_amman_form_template.png")
    warped, H = pipeline.warp_and_align(test_img)
    crops = pipeline.crop_field_rois(warped)
    print("CV Pipeline alignment and ROI cropping successful!")
    print(f"Mobile number crops: {len(crops['mobile_number']['boxes'])}")
    print(f"OMR questions cropped: {len(crops['omr_feedback'])}")
