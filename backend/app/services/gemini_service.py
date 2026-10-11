import os
import json
import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from backend.app.core.config import settings
from backend.app.services.qwen_service import QwenVisionService

class GeminiVisionService:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or settings.QWEN_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL or settings.QWEN_MODEL
        self.client = None

        # Route to Qwen service if API key is OpenRouter (sk-or-v1-...) or model is Qwen
        if self.api_key.startswith("sk-or-v1-") or "qwen" in self.model_name.lower():
            self.qwen_service = QwenVisionService(api_key=self.api_key, model_name=self.model_name)
        else:
            self.qwen_service = None
            if self.api_key:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=self.api_key)
                except Exception as e:
                    print(f"[GeminiVisionService] Google GenAI SDK init warning: {e}")

    def extract_field_crop(self, crop_image: np.ndarray, field_type: str) -> Tuple[Optional[str], float]:
        """
        Submits field crop image for independent second-pass verification.
        """
        if self.qwen_service:
            return self.qwen_service.extract_field_crop(crop_image, field_type)
        return None, 0.0

    def extract_form_fields(self, image: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Submits image to Vision API (Qwen or Gemini).
        Returns (extracted_dict, latency_ms).
        """
        if self.qwen_service:
            return self.qwen_service.extract_form_fields(image)

        if not self.api_key:
            raise RuntimeError("Vision API key is not configured in environment")

        start_time = time.time()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        prompt = """You are an expert OCR and vision document analysis AI.
Extract ALL handwritten fields from the provided feedback form for Bannari Amman Institute of Technology (BIT).

Form Layout & Fields to Extract:
1. student_name: Handwritten student full name.
2. school_name: Handwritten school name.
3. date: Date written on the form.
4. student_mobile_number: 10-digit handwritten student mobile number.
5. parent_mobile_number: 10-digit handwritten parent mobile number.
6. email: Student email ID.
7. cut_off_marks: Engineering cut off marks (e.g. 195, 184.56, 167.5).
8. interested_in_bit: "yes", "no", or "unclear" for Question 1.
9. career_guidance_usefulness: "excellent", "good", "satisfactory", "fair", "poor", or "unclear" for Question 2.
10. speaker_effectiveness: "excellent", "good", "satisfactory", "fair", "poor", or "unclear" for Question 3.
11. interested_in_choice_filling_assistance: "yes", "no", or "unclear" for Question 4.

Strict Rules:
- Preserve digits exactly as written on the image.
- Do NOT invent missing characters or silently correct ambiguous handwriting.
- Return null or "unclear" when unreadable.
- Preserve decimal points in cut_off_marks (e.g., 184.56). Do NOT convert 184.56 to 18456.
- Phone numbers should contain only digits (10 digits).
- Respond ONLY with valid JSON matching the specified structure without markdown formatting wrappers.
"""

        schema_structure = {
            "type": "OBJECT",
            "properties": {
                "student_name": {"type": "STRING"},
                "school_name": {"type": "STRING"},
                "date": {"type": "STRING"},
                "student_mobile_number": {"type": "STRING"},
                "parent_mobile_number": {"type": "STRING"},
                "email": {"type": "STRING"},
                "cut_off_marks": {"type": "STRING"},
                "interested_in_bit": {"type": "STRING"},
                "career_guidance_usefulness": {"type": "STRING"},
                "speaker_effectiveness": {"type": "STRING"},
                "interested_in_choice_filling_assistance": {"type": "STRING"}
            },
            "required": [
                "student_name", "school_name", "date", "student_mobile_number",
                "parent_mobile_number", "email", "cut_off_marks", "interested_in_bit",
                "career_guidance_usefulness", "speaker_effectiveness", "interested_in_choice_filling_assistance"
            ]
        }

        max_retries = 3
        backoff = 1.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pil_img, prompt],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema_structure,
                        "temperature": 0.1
                    }
                )
                latency = (time.time() - start_time) * 1000.0
                parsed_json = json.loads(response.text)
                return parsed_json, latency
            except Exception as e:
                last_exception = e
                print(f"[GeminiVisionService] Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0

        raise RuntimeError(f"Gemini Vision API extraction failed after {max_retries} attempts: {last_exception}")
