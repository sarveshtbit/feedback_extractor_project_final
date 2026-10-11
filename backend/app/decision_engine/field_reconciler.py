"""
Field-Level Evidence Reconciliation & Validation Engine.

This module validates and reconciles extracted values from the OCR pipeline
(TrOCR, PaddleOCR, BubbleDetector) and multi-modal secondary inputs, performing
business rule validation and producing structured provenance records for persistence.
"""

import re
from typing import Any, Dict, Optional, Tuple
from difflib import SequenceMatcher
from backend.app.validation.validator import FieldValidator


class FieldReconciler:
    """
    Reconciles OCR extraction output with validation rules and optional secondary evidence.
    """

    @staticmethod
    def calculate_string_similarity(str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
        s1 = re.sub(r'[^a-zA-Z0-9]', '', str1.lower())
        s2 = re.sub(r'[^a-zA-Z0-9]', '', str2.lower())
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    @classmethod
    def reconcile_ocr_field(
        cls,
        field_name: str,
        ocr_result: Dict[str, Any],
        secondary_val: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reconciles an extracted OCR field result against validation rules and secondary inputs.

        Parameters:
            field_name: Canonical schema field name (e.g. student_mobile_number, cut_off_marks).
            ocr_result: Dict produced by FieldExtractor (contains value, source, confidence, status).
            secondary_val: Optional secondary vision prediction for multi-evidence comparison.

        Returns:
            Dict containing:
                - field_name
                - final_value
                - source
                - final_confidence
                - status ("VERIFIED", "EXTRACTED", "NEEDS_REVIEW", "CONFLICT", "BLANK", "ENGINE_UNAVAILABLE")
                - reasoning
                - ocr_value
                - secondary_value
        """
        ocr_raw = ocr_result.get("value")
        ocr_clean = str(ocr_raw).strip() if ocr_raw is not None and str(ocr_raw).strip() not in ("null", "None", "") else None
        ocr_conf = ocr_result.get("confidence")
        ocr_source = ocr_result.get("source", "ocr")
        ocr_status = ocr_result.get("status", "EXTRACTED")

        sec_clean = str(secondary_val).strip() if secondary_val is not None and str(secondary_val).strip() not in ("null", "None", "") else None

        # 1. Handle Engine Unavailable
        if ocr_status == "ENGINE_UNAVAILABLE":
            return {
                "field_name": field_name,
                "final_value": sec_clean,
                "source": "secondary_fallback" if sec_clean else "none",
                "final_confidence": 0.50 if sec_clean else 0.0,
                "status": "NEEDS_REVIEW" if sec_clean else "ENGINE_UNAVAILABLE",
                "reasoning": ocr_result.get("error") or f"OCR engine unavailable for {field_name}",
                "ocr_value": None,
                "secondary_value": sec_clean,
            }

        # 2. Handle Blank
        if not ocr_clean and not sec_clean:
            return {
                "field_name": field_name,
                "final_value": None,
                "source": "none",
                "final_confidence": 0.0,
                "status": "BLANK",
                "reasoning": f"Field '{field_name}' left blank by student (no content detected).",
                "ocr_value": None,
                "secondary_value": None,
            }

        # 3. Field-specific validations

        # A. Mobile Numbers
        if "mobile" in field_name:
            ocr_valid, ocr_norm, ocr_err = FieldValidator.validate_phone_number(ocr_clean)
            sec_valid, sec_norm, _ = FieldValidator.validate_phone_number(sec_clean)

            if ocr_valid and sec_valid and ocr_norm == sec_norm:
                return {
                    "field_name": field_name,
                    "final_value": ocr_norm,
                    "source": f"{ocr_source}+secondary",
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "reasoning": "OCR and secondary predictions agree on valid 10-digit mobile number.",
                    "ocr_value": ocr_norm,
                    "secondary_value": sec_norm,
                }

            if ocr_valid:
                conf = ocr_conf if ocr_conf is not None else 0.90
                return {
                    "field_name": field_name,
                    "final_value": ocr_norm,
                    "source": ocr_source,
                    "final_confidence": conf,
                    "status": "VALID" if conf >= 0.80 else "NEEDS_REVIEW",
                    "reasoning": f"Extracted valid 10-digit mobile number via {ocr_source}.",
                    "ocr_value": ocr_norm,
                    "secondary_value": sec_clean,
                }

            if sec_valid:
                return {
                    "field_name": field_name,
                    "final_value": sec_norm,
                    "source": "secondary",
                    "final_confidence": 0.85,
                    "status": "NEEDS_REVIEW",
                    "reasoning": f"Secondary vision produced valid mobile ({sec_norm}) while OCR was invalid ({ocr_err}).",
                    "ocr_value": ocr_clean,
                    "secondary_value": sec_norm,
                }

            return {
                "field_name": field_name,
                "final_value": ocr_clean or sec_clean,
                "source": ocr_source if ocr_clean else "secondary",
                "final_confidence": 0.30,
                "status": "NEEDS_REVIEW",
                "reasoning": f"Invalid mobile format: {ocr_err}",
                "ocr_value": ocr_clean,
                "secondary_value": sec_clean,
            }

        # B. Cutoff Marks
        if "cutoff" in field_name:
            ocr_valid, ocr_norm, ocr_err = FieldValidator.validate_cutoff_marks(ocr_clean)
            sec_valid, sec_norm, _ = FieldValidator.validate_cutoff_marks(sec_clean)

            if ocr_valid and sec_valid and ocr_norm == sec_norm:
                return {
                    "field_name": field_name,
                    "final_value": ocr_norm,
                    "source": f"{ocr_source}+secondary",
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "reasoning": "OCR and secondary match on valid cut-off marks.",
                    "ocr_value": ocr_norm,
                    "secondary_value": sec_norm,
                }

            if ocr_valid:
                conf = ocr_conf if ocr_conf is not None else 0.88
                return {
                    "field_name": field_name,
                    "final_value": ocr_norm,
                    "source": ocr_source,
                    "final_confidence": conf,
                    "status": "VALID" if conf >= 0.80 else "NEEDS_REVIEW",
                    "reasoning": f"Extracted valid cut-off mark ({ocr_norm}) via {ocr_source}.",
                    "ocr_value": ocr_norm,
                    "secondary_value": sec_clean,
                }

            return {
                "field_name": field_name,
                "final_value": ocr_clean or sec_clean,
                "source": ocr_source if ocr_clean else "secondary",
                "final_confidence": 0.30,
                "status": "NEEDS_REVIEW",
                "reasoning": f"Invalid cutoff format: {ocr_err}",
                "ocr_value": ocr_clean,
                "secondary_value": sec_clean,
            }

        # C. Date
        if field_name == "date":
            ocr_valid, ocr_norm, ocr_err = FieldValidator.validate_date(ocr_clean)
            if ocr_valid:
                conf = ocr_conf if ocr_conf is not None else 0.90
                return {
                    "field_name": field_name,
                    "final_value": ocr_norm,
                    "source": ocr_source,
                    "final_confidence": conf,
                    "status": "VALID" if conf >= 0.75 else "NEEDS_REVIEW",
                    "reasoning": f"Valid date ({ocr_norm}) extracted via {ocr_source}.",
                    "ocr_value": ocr_norm,
                    "secondary_value": sec_clean,
                }
            return {
                "field_name": field_name,
                "final_value": ocr_clean,
                "source": ocr_source,
                "final_confidence": 0.40,
                "status": "NEEDS_REVIEW",
                "reasoning": f"Date requires review: {ocr_err}",
                "ocr_value": ocr_clean,
                "secondary_value": sec_clean,
            }

        # D. Email
        if field_name == "email":
            ocr_valid, ocr_norm, ocr_err = FieldValidator.validate_email(ocr_clean)
            if ocr_valid:
                conf = ocr_conf if ocr_conf is not None else 0.90
                return {
                    "field_name": field_name,
                    "final_value": ocr_norm,
                    "source": ocr_source,
                    "final_confidence": conf,
                    "status": "VALID" if conf >= 0.80 else "NEEDS_REVIEW",
                    "reasoning": f"Valid email format ({ocr_norm}) extracted via {ocr_source}.",
                    "ocr_value": ocr_norm,
                    "secondary_value": sec_clean,
                }
            return {
                "field_name": field_name,
                "final_value": ocr_clean,
                "source": ocr_source,
                "final_confidence": 0.40,
                "status": "NEEDS_REVIEW",
                "reasoning": f"Email format check: {ocr_err}",
                "ocr_value": ocr_clean,
                "secondary_value": sec_clean,
            }

        # E. Questionnaire responses (BubbleDetector)
        if ocr_source == "bubble_detector" or field_name in (
            "interested_in_bit",
            "career_guidance_usefulness",
            "speaker_effectiveness",
            "interested_in_choice_filling_assistance",
        ):
            q_valid, q_norm, q_err = FieldValidator.validate_questionnaire_option(field_name, ocr_clean)
            if ocr_status in ("SELECTED", "VALID") and q_valid:
                conf = ocr_conf if ocr_conf is not None else 0.95
                return {
                    "field_name": field_name,
                    "final_value": q_norm,
                    "source": "bubble_detector",
                    "final_confidence": conf,
                    "status": "VALID",
                    "reasoning": f"Unambiguous bubble selection '{q_norm}' detected.",
                    "ocr_value": q_norm,
                    "secondary_value": sec_clean,
                }
            elif ocr_status == "MULTIPLE_MARKS":
                return {
                    "field_name": field_name,
                    "final_value": None,
                    "source": "bubble_detector",
                    "final_confidence": 0.0,
                    "status": "CONFLICT",
                    "reasoning": "Multiple bubbles marked for single-choice question.",
                    "ocr_value": None,
                    "secondary_value": sec_clean,
                }
            elif ocr_status == "AMBIGUOUS":
                return {
                    "field_name": field_name,
                    "final_value": q_norm if q_valid else ocr_clean,
                    "source": "bubble_detector",
                    "final_confidence": ocr_conf or 0.50,
                    "status": "NEEDS_REVIEW",
                    "reasoning": "Borderline bubble fill requires human review.",
                    "ocr_value": ocr_clean,
                    "secondary_value": sec_clean,
                }
            else:
                return {
                    "field_name": field_name,
                    "final_value": None,
                    "source": "bubble_detector",
                    "final_confidence": 0.0,
                    "status": "BLANK",
                    "reasoning": "Question left unanswered.",
                    "ocr_value": None,
                    "secondary_value": sec_clean,
                }

        # F. General text fields (student_name, school_name)
        final_val = ocr_clean or sec_clean
        conf = ocr_conf if ocr_conf is not None else (0.85 if ocr_clean else 0.0)
        return {
            "field_name": field_name,
            "final_value": final_val,
            "source": ocr_source if ocr_clean else "secondary",
            "final_confidence": conf,
            "status": "EXTRACTED" if final_val else "BLANK",
            "reasoning": f"Extracted via {ocr_source}." if ocr_clean else "Field empty.",
            "ocr_value": ocr_clean,
            "secondary_value": sec_clean,
        }

    @classmethod
    def reconcile_field(
        cls,
        field_name: str,
        qwen_val: Optional[str],
        cnn_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Backward-compatible reconciliation engine matching exact legacy test expectations.
        """
        cnn_text = cnn_result.get("consensus_text") if cnn_result else None
        cnn_conf = cnn_result.get("ocr_confidence", cnn_result.get("confidence", 0.0)) if cnn_result else 0.0
        cnn_agreement = cnn_result.get("variant_agreement", 1.0) if cnn_result else 0.0

        qwen_clean = str(qwen_val).strip() if qwen_val is not None and str(qwen_val).strip() not in ("null", "None", "") else None
        cnn_clean = str(cnn_text).strip() if cnn_text is not None and str(cnn_text).strip() not in ("null", "None", "") else None

        # 1. Blank / None
        if not qwen_clean and not cnn_clean:
            return {
                "field_name": field_name,
                "final_value": None,
                "source": "none",
                "qwen_value": None,
                "ocr_value": None,
                "vlm_confidence": 0.0,
                "ocr_confidence": 0.0,
                "ocr_variant_agreement": 0.0,
                "final_confidence": 0.0,
                "status": "BLANK",
                "decision": "BLANK",
                "reasoning": "Field left blank by student (both Qwen VLM and CV/CNN detected no content)."
            }

        # 2. Mobile Numbers
        if "mobile" in field_name:
            g_valid, g_norm, g_err = FieldValidator.validate_phone_number(qwen_clean)
            c_valid, c_norm, c_err = FieldValidator.validate_phone_number(cnn_clean)

            if g_valid and c_valid:
                if g_norm == c_norm:
                    return {
                        "field_name": field_name,
                        "final_value": g_norm,
                        "source": "ocr+vlm",
                        "qwen_value": g_norm,
                        "ocr_value": c_norm,
                        "vlm_confidence": 0.90,
                        "ocr_confidence": cnn_conf,
                        "ocr_variant_agreement": cnn_agreement,
                        "final_confidence": 0.98,
                        "status": "CORROBORATED",
                        "decision": "CORROBORATED",
                        "reasoning": "Qwen VLM and CV/CNN match perfectly on valid 10-digit mobile number."
                    }
                else:
                    if cnn_conf >= 0.85:
                        return {
                            "field_name": field_name,
                            "final_value": c_norm,
                            "source": "ocr_override",
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.85,
                            "ocr_confidence": cnn_conf,
                            "ocr_variant_agreement": cnn_agreement,
                            "final_confidence": 0.92,
                            "status": "OCR_OVERRIDE",
                            "decision": "OCR_OVERRIDE",
                            "reasoning": f"CV/CNN override selected due to high confidence (conf={cnn_conf:.2f})."
                        }
                    else:
                        return {
                            "field_name": field_name,
                            "final_value": g_norm,
                            "source": "qwen_vlm",
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.85,
                            "ocr_confidence": cnn_conf,
                            "ocr_variant_agreement": cnn_agreement,
                            "final_confidence": 0.80,
                            "status": "VLM_SELECTED",
                            "decision": "VLM_SELECTED",
                            "reasoning": f"Qwen VLM selected for mobile number. Disagreement with CV/CNN ({c_norm})."
                        }

            if c_valid and not g_valid:
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "source": "ocr_override",
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.40,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.90,
                    "status": "OCR_OVERRIDE",
                    "decision": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN passed 10-digit validation ({c_norm}) while Qwen failed."
                }

            if g_valid and not c_valid:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "qwen_vlm",
                    "qwen_value": g_norm,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.88,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.88,
                    "status": "VLM_SELECTED",
                    "decision": "VLM_SELECTED",
                    "reasoning": "Qwen VLM passed 10-digit validation while CV/CNN was invalid."
                }

        # 3. Cutoff Marks
        if "cutoff" in field_name:
            g_valid, g_norm, g_err = FieldValidator.validate_cutoff_marks(qwen_clean)
            c_valid, c_norm, c_err = FieldValidator.validate_cutoff_marks(cnn_clean)

            if g_valid and c_valid and g_norm == c_norm:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "ocr+vlm",
                    "qwen_value": g_norm,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "decision": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN match on valid cut-off marks."
                }

            if c_valid and (not g_valid or cnn_conf >= 0.85):
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "source": "ocr_override",
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.40 if qwen_clean else 0.0,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.90,
                    "status": "OCR_OVERRIDE",
                    "decision": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN passed cut-off validation ({c_norm})."
                }

        # 4. Date
        if field_name == "date":
            g_valid, g_norm, _ = FieldValidator.validate_date(qwen_clean)
            c_valid, c_norm, _ = FieldValidator.validate_date(cnn_clean)

            if g_valid and c_valid and g_norm == c_norm:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "ocr+vlm",
                    "qwen_value": g_norm,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "decision": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN match on valid date."
                }

            if c_valid and (not g_valid or cnn_conf >= 0.85):
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "source": "ocr_override",
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.50 if qwen_clean else 0.0,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.92,
                    "status": "OCR_OVERRIDE",
                    "decision": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN agreed on date ({c_norm})."
                }

        # 5. General Text Fields (school_name, student_name)
        if qwen_clean and cnn_clean:
            sim = cls.calculate_string_similarity(qwen_clean, cnn_clean)
            if sim >= 0.80 or qwen_clean.lower() == cnn_clean.lower():
                # Prefer the more complete string if one is a substring of the other
                chosen = qwen_clean if len(qwen_clean) >= len(cnn_clean) else cnn_clean
                return {
                    "field_name": field_name,
                    "final_value": chosen,
                    "source": "ocr+vlm",
                    "qwen_value": qwen_clean,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "decision": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN corroborated."
                }

        if qwen_clean:
            return {
                "field_name": field_name,
                "final_value": qwen_clean,
                "source": "qwen_vlm",
                "qwen_value": qwen_clean,
                "ocr_value": cnn_clean,
                "vlm_confidence": 0.88,
                "ocr_confidence": cnn_conf,
                "ocr_variant_agreement": cnn_agreement,
                "final_confidence": 0.88,
                "status": "VLM_SELECTED",
                "decision": "VLM_SELECTED",
                "reasoning": "Qwen VLM selected as primary visual extraction source."
            }

        if cnn_clean:
            return {
                "field_name": field_name,
                "final_value": cnn_clean,
                "source": "ocr_override",
                "qwen_value": None,
                "ocr_value": cnn_clean,
                "vlm_confidence": 0.0,
                "ocr_confidence": cnn_conf,
                "ocr_variant_agreement": cnn_agreement,
                "final_confidence": 0.80,
                "status": "OCR_OVERRIDE",
                "decision": "OCR_OVERRIDE",
                "reasoning": f"CV/CNN extracted text (conf={cnn_conf:.2f})."
            }

        return {
            "field_name": field_name,
            "final_value": None,
            "source": "none",
            "qwen_value": qwen_clean,
            "ocr_value": cnn_clean,
            "vlm_confidence": 0.0,
            "ocr_confidence": cnn_conf,
            "ocr_variant_agreement": cnn_agreement,
            "final_confidence": 0.0,
            "status": "BLANK",
            "decision": "BLANK",
            "reasoning": "Unable to recognize content."
        }
