"""
Standalone PaddleOCR Worker Bridge.

Runs inside the isolated `.venv-paddleocr` Python environment to execute
PaddleOCR inference without package dependency collisions with PyTorch/FastAPI in `.venv`.

Communication protocol:
- Input: JSON lines read from stdin.
- Output: Single-line JSON responses printed to stdout (flushed immediately).
- Logging: Diagnostic and error logs sent exclusively to stderr. No PII is logged.
"""

import os
import sys
import json
import time
import logging
import cv2
import numpy as np

# Configure stderr logging (never stdout, which is reserved for JSON-RPC transport)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[PaddleWorker %(levelname)s] %(message)s",
)
logger = logging.getLogger("paddle_bridge_worker")

# Ensure torch lib is in dll search path if present (Windows)
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

sys.modules.setdefault('sklearn', None)


def send_response(data: dict) -> None:
    """Writes a single-line JSON response to stdout and flushes."""
    out_str = json.dumps(data)
    sys.stdout.write(out_str + "\n")
    sys.stdout.flush()


def run_worker() -> None:
    logger.info("Initializing PaddleOCR worker inside %s", sys.prefix)
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        logger.info("PaddleOCR engine loaded successfully")
        send_response({"status": "READY", "version": "PaddleOCR 2.8.1"})
    except Exception as e:
        logger.error("Failed to initialize PaddleOCR: %s", str(e), exc_info=True)
        send_response({"status": "INIT_FAILED", "error": str(e)})
        sys.exit(1)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                logger.info("Stdin EOF encountered. Exiting cleanly.")
                break

            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as je:
                send_response({"status": "ERROR", "error": f"Invalid JSON input: {je}"})
                continue

            action = cmd.get("action", "ocr")

            if action == "ping":
                send_response({"status": "PONG"})
                continue

            elif action == "shutdown":
                logger.info("Shutdown action requested. Exiting.")
                send_response({"status": "SHUTDOWN_ACK"})
                break

            elif action == "ocr":
                image_path = cmd.get("image_path")
                crop_box = cmd.get("crop_box")  # Optional [y1, y2, x1, x2]
                det = cmd.get("det", True)
                rec = cmd.get("rec", True)
                cls = cmd.get("cls", False)

                if not image_path or not isinstance(image_path, str):
                    send_response({"status": "INVALID_INPUT", "error": "Missing or invalid 'image_path'"})
                    continue

                if not os.path.exists(image_path):
                    send_response({"status": "FILE_NOT_FOUND", "error": f"Image path does not exist: {image_path}"})
                    continue

                # Load image safely
                img = cv2.imread(image_path)
                if img is None or img.size == 0:
                    send_response({"status": "INVALID_IMAGE", "error": f"Could not read image from {image_path}"})
                    continue

                if crop_box and len(crop_box) == 4:
                    y1, y2, x1, x2 = crop_box
                    h, w = img.shape[:2]
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(y1 + 1, min(h, y2))
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(x1 + 1, min(w, x2))
                    img = img[y1:y2, x1:x2]

                t0 = time.time()
                try:
                    results = ocr.ocr(img, det=det, rec=rec, cls=cls)
                except Exception as oe:
                    logger.error("OCR inference failure: %s", str(oe), exc_info=True)
                    send_response({"status": "INFERENCE_FAILED", "error": str(oe)})
                    continue

                latency_ms = round((time.time() - t0) * 1000.0, 2)
                items = []
                tokens = []

                if results and len(results) > 0 and results[0] is not None:
                    for line_res in results[0]:
                        if not line_res:
                            continue

                        # When det=True: line_res is [polygon, (text, conf)]
                        # When det=False: line_res is (text, conf)
                        if isinstance(line_res, (list, tuple)) and len(line_res) >= 2 and isinstance(line_res[0], (list, tuple)):
                            box = line_res[0]
                            text_conf = line_res[1]
                            text = str(text_conf[0]).strip() if isinstance(text_conf, (list, tuple)) and len(text_conf) > 0 else str(text_conf)
                            conf = float(text_conf[1]) if isinstance(text_conf, (list, tuple)) and len(text_conf) > 1 else 0.0
                        elif isinstance(line_res, (list, tuple)) and len(line_res) >= 2:
                            box = None
                            text = str(line_res[0]).strip()
                            conf = float(line_res[1]) if isinstance(line_res[1], (int, float)) else 0.0
                        else:
                            box = None
                            text = str(line_res).strip()
                            conf = 0.0

                        if box:
                            xs = [pt[0] for pt in box]
                            ys = [pt[1] for pt in box]
                            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                            poly = [[int(pt[0]), int(pt[1])] for pt in box]
                        else:
                            bbox = [0, 0, img.shape[1], img.shape[0]]
                            poly = []

                        items.append({
                            "text": text,
                            "confidence": round(conf, 4),
                            "bbox": bbox,
                            "polygon": poly,
                        })
                        if text:
                            tokens.append(text)

                # Return response (clean, structured, zero PII logging)
                send_response({
                    "status": "SUCCESS",
                    "items": items,
                    "full_text": " ".join(tokens),
                    "token_count": len(items),
                    "latency_ms": latency_ms,
                })

            else:
                send_response({"status": "UNKNOWN_ACTION", "error": f"Unknown action '{action}'"})

        except Exception as e:
            logger.error("Unhandled exception in worker loop: %s", str(e), exc_info=True)
            send_response({"status": "SERVER_ERROR", "error": str(e)})


if __name__ == "__main__":
    run_worker()
