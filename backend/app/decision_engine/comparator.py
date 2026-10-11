from typing import Dict, Any, List, Optional
from backend.app.validation.validator import FieldValidator
from backend.app.core.config import settings

class DecisionEngine:
    @staticmethod
    def evaluate_phone_number(
        gemini_val: Optional[str],
        cnn_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuses Gemini extraction and CNN recognition for Student Mobile Number.
        Identifies character-level disagreements and decides final outcome.
        """
        cnn_val = cnn_result.get("value")
        cnn_conf = cnn_result.get("confidence", 0.0)
        cnn_status = cnn_result.get("status")

        # 1. Normalize Gemini and CNN strings
        g_valid, g_norm, g_err = FieldValidator.validate_phone_number(gemini_val)
        
        if cnn_status == "LOCALIZATION_FAILED":
            return {
                "final_value": g_norm if g_valid else (gemini_val or ""),
                "gemini_value": gemini_val,
                "cnn_value": None,
                "decision": "MANUAL_REVIEW_REQUIRED",
                "confidence": 0.50,
                "reason": "Student Mobile field localization failed to detect or validate 10 bounding boxes (CNN recognition was not evaluated).",
                "differing_positions": []
            }

        if cnn_status == "CNN_UNAVAILABLE":
            final_val = g_norm if g_valid else (gemini_val or "")
            conf = 0.85 if g_valid else 0.40
            return {
                "final_value": final_val,
                "gemini_value": gemini_val,
                "cnn_value": None,
                "decision": "CNN_UNAVAILABLE",
                "confidence": conf,
                "reason": f"CNN model unavailable. Retained Gemini extraction ({g_err or 'Valid'}).",
                "differing_positions": []
            }

        c_valid, c_norm, c_err = FieldValidator.validate_phone_number(cnn_val)

        # 2. Case: Agreement
        if g_valid and c_valid and g_norm == c_norm:
            return {
                "final_value": g_norm,
                "gemini_value": g_norm,
                "cnn_value": c_norm,
                "decision": "AGREEMENT",
                "confidence": 0.98,
                "reason": "Gemini and CNN extractions match perfectly and pass mobile validation.",
                "differing_positions": []
            }

        # 3. Case: Disagreement - Character Level Analysis
        differing_positions = []
        if g_norm and c_norm:
            max_len = max(len(g_norm), len(c_norm))
            for i in range(max_len):
                ch_g = g_norm[i] if i < len(g_norm) else ""
                ch_c = c_norm[i] if i < len(c_norm) else ""
                if ch_g != ch_c:
                    differing_positions.append(i + 1)  # 1-indexed

        # 4. Resolve Disagreement based on Validation and CNN Calibration Evidence
        if g_valid and not c_valid:
            return {
                "final_value": g_norm,
                "gemini_value": g_norm,
                "cnn_value": c_norm,
                "decision": "GEMINI_SELECTED",
                "confidence": 0.88,
                "reason": f"Gemini extraction passed 10-digit validation whereas CNN failed ({c_err}).",
                "differing_positions": differing_positions
            }

        if c_valid and not g_valid:
            if cnn_conf >= settings.CNN_CONFIDENCE_THRESHOLD:
                return {
                    "final_value": c_norm,
                    "gemini_value": g_norm,
                    "cnn_value": c_norm,
                    "decision": "CNN_SELECTED",
                    "confidence": 0.90,
                    "reason": f"CNN reconstruction passed validation with high confidence ({cnn_conf:.2f}) while Gemini failed ({g_err}).",
                    "differing_positions": differing_positions
                }

        if g_valid and c_valid and g_norm != c_norm:
            # Both valid but disagree on specific digits
            if cnn_conf >= 0.95:
                return {
                    "final_value": c_norm,
                    "gemini_value": g_norm,
                    "cnn_value": c_norm,
                    "decision": "CNN_SELECTED",
                    "confidence": 0.91,
                    "reason": f"Both valid but disagree at position(s) {differing_positions}. CNN selected due to high calibrated confidence ({cnn_conf:.2f}).",
                    "differing_positions": differing_positions
                }
            else:
                return {
                    "final_value": g_norm,
                    "gemini_value": g_norm,
                    "cnn_value": c_norm,
                    "decision": "MANUAL_REVIEW",
                    "confidence": 0.60,
                    "reason": f"Conflict detected at digit position(s) {differing_positions}. Flagged for manual verification.",
                    "differing_positions": differing_positions
                }

        # 5. Fallback: Neither valid or both uncertain
        return {
            "final_value": g_norm or c_norm or gemini_val or "",
            "gemini_value": gemini_val,
            "cnn_value": cnn_val,
            "decision": "MANUAL_REVIEW",
            "confidence": 0.30,
            "reason": f"Validation failure on both models (Gemini: {g_err}, CNN: {c_err}). Manual review required.",
            "differing_positions": differing_positions
        }

    @staticmethod
    def evaluate_cutoff_marks(
        gemini_val: Optional[str],
        cnn_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuses Gemini extraction and CNN recognition for Cut Off Marks.
        """
        cnn_val = cnn_result.get("value")
        cnn_conf = cnn_result.get("confidence", 0.0)
        cnn_status = cnn_result.get("status")

        g_valid, g_norm, g_err = FieldValidator.validate_cutoff_marks(gemini_val)

        if cnn_status == "CNN_UNAVAILABLE":
            final_val = g_norm if g_valid else (gemini_val or "")
            conf = 0.85 if g_valid else 0.40
            return {
                "final_value": final_val,
                "gemini_value": gemini_val,
                "cnn_value": None,
                "decision": "CNN_UNAVAILABLE",
                "confidence": conf,
                "reason": f"CNN model unavailable. Retained Gemini extraction ({g_err or 'Valid'}).",
                "differing_positions": []
            }

        c_valid, c_norm, c_err = FieldValidator.validate_cutoff_marks(cnn_val)

        if g_valid and c_valid and g_norm == c_norm:
            return {
                "final_value": g_norm,
                "gemini_value": g_norm,
                "cnn_value": c_norm,
                "decision": "AGREEMENT",
                "confidence": 0.98,
                "reason": "Gemini and CNN cut-off extractions match perfectly within valid 0-200 range.",
                "differing_positions": []
            }

        differing_positions = []
        if g_norm and c_norm and g_norm != c_norm:
            max_len = max(len(g_norm), len(c_norm))
            for i in range(max_len):
                ch_g = g_norm[i] if i < len(g_norm) else ""
                ch_c = c_norm[i] if i < len(c_norm) else ""
                if ch_g != ch_c:
                    differing_positions.append(i + 1)

        if g_valid and not c_valid:
            return {
                "final_value": g_norm,
                "gemini_value": g_norm,
                "cnn_value": c_norm,
                "decision": "GEMINI_SELECTED",
                "confidence": 0.88,
                "reason": f"Gemini cutoff passed validation whereas CNN failed ({c_err}).",
                "differing_positions": differing_positions
            }

        if c_valid and not g_valid:
            return {
                "final_value": c_norm,
                "gemini_value": g_norm,
                "cnn_value": c_norm,
                "decision": "CNN_SELECTED",
                "confidence": 0.90,
                "reason": f"CNN cutoff passed validation whereas Gemini failed ({g_err}).",
                "differing_positions": differing_positions
            }

        return {
            "final_value": g_norm or c_norm or gemini_val or "",
            "gemini_value": gemini_val,
            "cnn_value": cnn_val,
            "decision": "MANUAL_REVIEW",
            "confidence": 0.50,
            "reason": f"Conflict or validation failure for Cut Off Marks (Gemini: {g_norm}, CNN: {c_norm}).",
            "differing_positions": differing_positions
        }
