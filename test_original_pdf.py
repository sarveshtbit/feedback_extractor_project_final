import fitz
import cv2
import numpy as np
from pathlib import Path

# Original PDF file
PDF_PATH = Path(
    r"D:\Downloads\BANNARI AMMAN INSTITUTE OF TECHNOLOGY (1).pdf"
)

OUTPUT_DIR = Path("debug_original_forms")
OUTPUT_DIR.mkdir(exist_ok=True)

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}\n"
        "Update PDF_PATH to the actual location of your PDF."
    )

# Open the PDF
pdf = fitz.open(PDF_PATH)

if len(pdf) == 0:
    raise ValueError("The PDF contains no pages.")

for page_number, page in enumerate(pdf, start=1):

    # Render the PDF page at 300 DPI
    scale = 300 / 72
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False
    )

    image = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Split the page into upper and lower forms
    middle_y = image.shape[0] // 2

    top_form = image[:middle_y, :]
    bottom_form = image[middle_y:, :]

    # Save both forms separately
    top_path = OUTPUT_DIR / f"page_{page_number}_form_1.png"
    bottom_path = OUTPUT_DIR / f"page_{page_number}_form_2.png"

    cv2.imwrite(str(top_path), top_form)
    cv2.imwrite(str(bottom_path), bottom_form)

    print(f"Page {page_number}: {image.shape[1]} x {image.shape[0]}")
    print(f"Saved: {top_path}")
    print(f"Saved: {bottom_path}")

pdf.close()

print("\nPDF conversion and splitting completed.")
