import os
import pymupdf as fitz
import cv2
import numpy as np
from typing import List, Tuple

class PDFProcessor:
    @staticmethod
    def render_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[Tuple[int, str]]:
        """
        Renders multi-page PDF into high-res PNG files page-by-page.
        Returns list of tuples: (page_number, image_file_path).
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        rendered_pages = []

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for i in range(len(doc)):
            page_num = i + 1
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            output_filename = f"page_{page_num}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            pix.save(output_filepath)
            rendered_pages.append((page_num, output_filepath))

        doc.close()
        return rendered_pages

    @staticmethod
    def get_pdf_page_count(pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
