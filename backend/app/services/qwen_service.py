import os
import json
import time
import base64
import cv2
import numpy as np
import httpx
from typing import Dict, Tuple, Any, Optional
from backend.app.core.config import settings

class QwenVisionService:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.QWEN_API_KEY or settings.GEMINI_API_KEY or os.getenv("QWEN_API_KEY", "")
        self.model_name = "dots-studio/dots-3-note-preview:free"

        # Determine endpoint URL based on key structure
        if self.api_key.startswith("sk-or-v1-"):
            self.endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
        elif "dashscope" in self.api_key.lower():
            self.endpoint_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        else:
            self.endpoint_url = "https://openrouter.ai/api/v1/chat/completions"

        print(f"[QWEN SERVICE] Initialized | QWEN_API_KEY configured: {bool(self.api_key)} | model={self.model_name} | endpoint={self.endpoint_url}")

    def encode_image_to_base64(self, image: np.ndarray, max_dim: int = 1024) -> str:
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image

        success, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode image to JPEG format")
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _parse_json_from_response(res_json: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        choices = res_json.get("choices", [])
        if not choices:
            return {}, False
        msg = choices[0].get("message", {})
        content_str = msg.get("content") or msg.get("reasoning") or ""
        if not content_str and "text" in choices[0]:
            content_str = choices[0]["text"] or ""
        
        if not isinstance(content_str, str):
            return {}, False

        s_idx = content_str.find("{")
        e_idx = content_str.rfind("}")
        if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
            json_sub = content_str[s_idx:e_idx+1]
            try:
                return json.loads(json_sub), True
            except Exception:
                pass
        return {}, False

    def extract_form_fields(self, image: np.ndarray) -> Tuple[Dict[str, Any], float]:
        if not self.api_key:
            print("[QWEN VISION WARNING] No Qwen/Gemini API key configured in environment.")
            return {}, 0.0

        start_time = time.time()
        base64_img = self.encode_image_to_base64(image, max_dim=1024)

        prompt = """Extract handwritten fields from this form into JSON:
{
  "student_name": "handwritten student name",
  "school_name": "school name",
  "date": "date on form",
  "student_mobile_number": "10-digit student mobile",
  "parent_mobile_number": "10-digit parent mobile",
  "email": "student email",
  "cut_off_marks": "cut off marks e.g. 185.50",
  "interested_in_bit": "yes or no",
  "career_guidance_usefulness": "excellent, good, or satisfactory",
  "speaker_effectiveness": "excellent, good, or satisfactory",
  "interested_in_choice_filling_assistance": "yes or no"
}
Return ONLY a raw valid JSON object.
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if "openrouter" in self.endpoint_url:
            headers["HTTP-Referer"] = "https://bannari-amman.ac.in"
            headers["X-Title"] = "BIT Form Digitization"

        candidate_models = [
            "dots-studio/dots-3-note-preview:free",
            "qwen/qwen-2.5-vl-72b-instruct",
            "google/gemini-2.0-flash-exp:free",
            "openrouter/auto"
        ]

        for current_model in candidate_models:
            payload = {
                "model": current_model,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1
            }

            print(f"[QWEN EXTRACTION REQUEST] Trying model={current_model}")

            try:
                with httpx.Client(timeout=40.0) as client:
                    resp = client.post(self.endpoint_url, headers=headers, json=payload)
                    
                    if resp.status_code != 200:
                        print(f"[QWEN EXTRACTION API ERROR] Model {current_model} returned Status {resp.status_code}: {resp.text[:200]}")
                        continue

                    res_json = resp.json()
                    parsed_dict, ok = self._parse_json_from_response(res_json)
                    if ok:
                        latency = (time.time() - start_time) * 1000.0
                        print(f"[QWEN EXTRACTION RESPONSE RECEIVED] Successfully parsed structured JSON with model={current_model} in {latency:.2f} ms")
                        return parsed_dict, latency

            except Exception as e:
                print(f"[QWEN EXTRACTION ATTEMPT FAILED] Model {current_model} failed: {e}")

        print("[QWEN SERVICE WARNING] All Qwen/Gemini vision model endpoints failed. Pipeline will rely on Pure CV + CNN digit recognizer.")
        return {}, (time.time() - start_time) * 1000.0

    def extract_field_crop(self, crop_image: np.ndarray, field_type: str) -> Tuple[Optional[str], float]:
        if not self.api_key:
            return None, 0.0

        start_time = time.time()
        base64_crop = self.encode_image_to_base64(crop_image, max_dim=800)

        prompt = f"Independently inspect the handwritten {field_type} field in this image and return JSON: {{\"value\": \"extracted-value\"}}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "dots-studio/dots-3-note-preview:free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_crop}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.endpoint_url, headers=headers, json=payload)
                resp.raise_for_status()
                res_json = resp.json()
                parsed_dict, ok = self._parse_json_from_response(res_json)
                latency = (time.time() - start_time) * 1000.0
                return parsed_dict.get("value") if ok else None, latency
        except Exception as e:
            return None, (time.time() - start_time) * 1000.0
