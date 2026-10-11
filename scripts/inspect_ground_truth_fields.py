import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.modules.setdefault('sklearn', None)

import cv2
import yaml
import json
import numpy as np

# Load template config
with open("config/form_template_experimental.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Standalone aligner
class Aligner:
    def __init__(self):
        self.canvas_w = 2479
        self.canvas_h = 3508
        self.ref = cv2.imread("bannari_amman_form_template.png")
        self.dst_markers = np.array([[140, 140], [2339, 140], [2339, 3368], [140, 3368]], dtype=np.float32)

    def align(self, img):
        if self.ref is not None:
            try:
                orb = cv2.ORB_create(3000)
                kp1, des1 = orb.detectAndCompute(img, None)
                kp2, des2 = orb.detectAndCompute(self.ref, None)
                if des1 is not None and des2 is not None and len(des1) >= 10 and len(des2) >= 10:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
                    if len(matches) >= 15:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if H is not None and H.shape == (3, 3):
                            return cv2.warpPerspective(img, H, (self.canvas_w, self.canvas_h)), True
            except Exception:
                pass
        scale = min(self.canvas_w / float(img.shape[1]), self.canvas_h / float(img.shape[0]))
        nw, nh = int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 255, dtype=np.uint8)
        canvas[(self.canvas_h - nh)//2:(self.canvas_h - nh)//2 + nh, (self.canvas_w - nw)//2:(self.canvas_w - nw)//2 + nw] = resized
        return canvas, False

aligner = Aligner()
os.makedirs("storage/stage9_investigation/field_crops", exist_ok=True)

test_forms = ["form1.jpeg", "form2.jpeg", "form3.jpeg", "form4.jpeg", "form5.jpeg"]
fields = ["student_name", "school_name", "date", "student_mobile", "parent_mobile", "email", "cutoff_marks"]

for fname in test_forms:
    fpath = os.path.join("cnn_dataset", fname)
    if not os.path.exists(fpath):
        continue
    img = cv2.imread(fpath)
    aligned, success = aligner.align(img)
    h, w = aligned.shape[:2]
    
    for f in fields:
        b = cfg["fields"][f]["normalized_bbox"]
        x1, y1, x2, y2 = int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)
        crop = aligned[y1:y2, x1:x2]
        crop_path = f"storage/stage9_investigation/field_crops/{fname.replace('.jpeg','')}_{f}.png"
        cv2.imwrite(crop_path, crop)
        
print("Field crops generated successfully in storage/stage9_investigation/field_crops/")
