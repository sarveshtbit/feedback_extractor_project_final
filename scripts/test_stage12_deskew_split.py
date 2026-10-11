import cv2
import numpy as np
import os

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/1800, threshold=150, minLineLength=200, maxLineGap=10)
    angles = []
    if lines is not None:
        for l in lines:
            if len(l.shape) > 0 and l.shape[0] == 4:
                x1, y1, x2, y2 = l
            else:
                x1, y1, x2, y2 = l[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) > 150:
                angle = np.degrees(np.arctan2(dy, dx))
                if abs(angle) < 5.0:
                    angles.append(angle)
    skew = float(np.median(angles)) if angles else 0.0
    h, w = img.shape[:2]
    if abs(skew) > 0.05:
        # Note: if line has dy > 0, angle is positive, meaning rotated clockwise.
        # Rotating by angle restores it.
        M = cv2.getRotationMatrix2D((w // 2, h // 2), skew, 1.0)
        img_deskewed = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return img_deskewed, skew
    return img, 0.0

def split_stacked_forms(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y_min, y_max = int(0.35 * h), int(0.65 * h)
    row_means = np.mean(gray[y_min:y_max, :], axis=1)
    white_indices = np.where(row_means > 254.0)[0] + y_min
    if len(white_indices) >= 15:
        split_y = int(np.median(white_indices))
        return [
            (img[0:split_y, :], "form_1"),
            (img[split_y:h, :], "form_2")
        ], split_y
    return [(img, "form_1")], None

for path in ['storage/stage12_independent_validation/val_doc_unseen_1.png', 'storage/stage12_independent_validation/val_doc_unseen_2.png']:
    img = cv2.imread(path)
    deskewed, skew = deskew_image(img)
    forms, split_y = split_stacked_forms(deskewed)
    print(f"{path}: detected skew={skew:.3f} deg, split_y={split_y}, forms={[f[0].shape for f in forms]}")
