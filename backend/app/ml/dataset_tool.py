import os
import cv2
import csv
import argparse
from typing import List, Dict, Any
from backend.app.services.alignment import FormAligner
from backend.app.services.roi_extractor import ROIExtractor

def crop_dataset_from_form(
    image_path: str, output_dir: str, writer_id: str, ground_truth_phone: str = "", ground_truth_cutoff: str = ""
):
    """
    Crops 10 mobile digit boxes and cutoff digit boxes from aligned form image for dataset generation.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        return

    aligner = FormAligner()
    extractor = ROIExtractor()

    img = cv2.imread(image_path)
    aligned, _, success = aligner.align(img)
    rois = extractor.crop_rois(aligned)

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    file_exists = os.path.exists(metadata_path)

    samples_added = 0
    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path", "label", "writer_id", "source"])

        # Mobile number crops
        mob_boxes = rois.get("student_mobile", {}).get("boxes", [])
        for i, box_item in enumerate(mob_boxes):
            crop = box_item["crop"]
            label = int(ground_truth_phone[i]) if (ground_truth_phone and i < len(ground_truth_phone)) else 0
            fname = f"mobile_{writer_id}_{i}_{label}.png"
            rel_path = os.path.join("crops", fname)
            full_path = os.path.join(output_dir, "crops", fname)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            cv2.imwrite(full_path, crop)

            writer.writerow([rel_path, label, writer_id, "form_cropped"])
            samples_added += 1

    print(f"Successfully extracted {samples_added} digit crops from {image_path} into {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop digit boxes from scanned form for CNN dataset")
    parser.add_argument("--image", required=True, help="Path to scanned form image")
    parser.add_argument("--out", default="./ml/datasets/digits", help="Output dataset directory")
    parser.add_argument("--writer", required=True, help="Writer ID (e.g. student_001)")
    parser.add_argument("--phone", default="", help="10-digit ground truth phone number")
    args = parser.parse_args()

    crop_dataset_from_form(args.image, args.out, args.writer, args.phone)
