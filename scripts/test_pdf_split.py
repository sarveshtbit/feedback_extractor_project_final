import os
import cv2
import numpy as np

img = cv2.imread('storage/stage11_dummy_pdf_verification/rendered_page_1.png')
h, w = img.shape[:2]
print(f"Page dimensions: {w} x {h}")

# The page is 3300 tall, exactly 2 equal halves of 1650 px each
# Check the whitespace between y=1520 and y=1630
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
search_zone = gray[1520:1630, :]
row_brightness = np.mean(search_zone, axis=1)
best_split_local = int(np.argmax(row_brightness))
split_y = 1520 + best_split_local
print(f"Optimal clean split line: y = {split_y} (search row brightness = {row_brightness[best_split_local]:.1f})")

# If middle is exactly h // 2 = 1650, check if splitting at split_y gives clean top and bottom
# Top form: [0:split_y, :]
# Bottom form: [split_y:h, :]
form1 = img[0:split_y, :]
form2 = img[split_y:h, :]

print(f"Form 1 shape: {form1.shape}")
print(f"Form 2 shape: {form2.shape}")

os.makedirs('storage/stage11_dummy_pdf_verification', exist_ok=True)
cv2.imwrite('storage/stage11_dummy_pdf_verification/separated_form_1.png', form1)
cv2.imwrite('storage/stage11_dummy_pdf_verification/separated_form_2.png', form2)

print("Saved separated_form_1.png and separated_form_2.png successfully.")
