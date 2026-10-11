import os
import sys
import time
import cv2
import json
import numpy as np

def main():
    print("=== TASK 2: VERIFY REAL PADDLEOCR INFERENCE ===")
    t0 = time.time()
    print("Initializing PaddleOCR engine on CPU...")
    from paddleocr import PaddleOCR
    
    # Initialize PaddleOCR with English language recognition
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    init_time = time.time() - t0
    print(f"PaddleOCR initialized in {init_time:.2f} seconds.")
    
    # Test on real feedback form
    test_image_path = "cnn_dataset/form1.jpeg"
    if not os.path.exists(test_image_path):
        test_image_path = "bannari_amman_form_template.png"
        
    print(f"\nRunning real inference on: {test_image_path}")
    img = cv2.imread(test_image_path)
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    t_start = time.time()
    results = ocr.ocr(test_image_path, cls=True)
    infer_time = time.time() - t_start
    print(f"Inference completed in {infer_time:.2f} seconds.")
    
    print("\n--- SAMPLE RECOGNIZED TEXT BLOCKS (TOP 10) ---")
    extracted_blocks = []
    if results and len(results) > 0 and results[0] is not None:
        for idx, line in enumerate(results[0][:15]):
            bbox, (text, conf) = line
            # Mask any potential PII
            masked_text = text if len(text) <= 3 else text[:2] + "***" + text[-2:]
            print(f"[{idx+1:02d}] Text: '{masked_text}' | Conf: {conf:.4f} | Box: {bbox}")
            extracted_blocks.append({
                "bbox": bbox,
                "text_raw": text,
                "text_masked": masked_text,
                "confidence": float(conf)
            })
            
    os.makedirs("storage/stage8_paddleocr_verification", exist_ok=True)
    with open("storage/stage8_paddleocr_verification/real_inference_sample.json", "w") as f:
        json.dump({
            "init_time_seconds": round(init_time, 2),
            "infer_time_seconds": round(infer_time, 2),
            "image": test_image_path,
            "total_blocks_detected": len(results[0]) if results and results[0] else 0,
            "sample_blocks": extracted_blocks
        }, f, indent=2)
        
    print("\nVerification report saved to storage/stage8_paddleocr_verification/real_inference_sample.json")

if __name__ == '__main__':
    main()
