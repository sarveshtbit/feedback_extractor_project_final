import re
import datetime
from typing import Dict, Any, Tuple, Optional
from backend.app.core.config import settings

class FieldValidator:
    @staticmethod
    def validate_phone_number(phone_str: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        """
        Validates 10-digit Indian mobile number format (starts with 6-9).
        Returns (is_valid, normalized_phone, error_message).
        """
        if not phone_str:
            return False, "", "Phone number is empty or missing"

        # Remove spaces, hyphens, and optional country code prefix
        cleaned = re.sub(r"[^\d]", "", phone_str)
        if len(cleaned) == 12 and cleaned.startswith("91"):
            cleaned = cleaned[2:]

        if len(cleaned) != 10:
            return False, cleaned, f"Phone number must be exactly 10 digits (got {len(cleaned)})"

        if not re.match(r"^[6-9]\d{9}$", cleaned):
            return False, cleaned, "Invalid Indian mobile number prefix (must start with 6, 7, 8, or 9)"

        return True, cleaned, None

    @staticmethod
    def validate_cutoff_marks(cutoff_str: Optional[str], min_val: float = 0.0, max_val: float = 200.0) -> Tuple[bool, str, Optional[str]]:
        """
        Validates engineering cut off mark (0.0 <= cutoff <= 200.0). Preserves decimal string.
        """
        if not cutoff_str:
            return False, "", "Cutoff mark is empty"

        cleaned = cutoff_str.strip()
        try:
            val = float(cleaned)
            if val < min_val or val > max_val:
                return False, cleaned, f"Cutoff mark {val} out of valid range [{min_val}, {max_val}]"
            return True, cleaned, None
        except ValueError:
            return False, cleaned, f"Cutoff mark '{cleaned}' is not a valid numeric decimal"

    @staticmethod
    def validate_email(email_str: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        if not email_str:
            return False, "", "Email is empty"
        cleaned = email_str.strip()
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if re.match(pattern, cleaned):
            return True, cleaned, None
        return False, cleaned, "Invalid email address format"

    @staticmethod
    def validate_date(date_str: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        if not date_str:
            return False, "", "Date is empty"
        cleaned = date_str.strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y"):
            try:
                datetime.datetime.strptime(cleaned, fmt)
                return True, cleaned, None
            except ValueError:
                continue
        return False, cleaned, "Invalid date format"

    @staticmethod
    def validate_questionnaire_option(field_name: str, val: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        if not val:
            return False, "", "Questionnaire option is empty"
        cleaned = val.strip().lower()
        if field_name in ("interested_in_bit", "interested_in_choice_filling_assistance"):
            if cleaned in ("yes", "no"):
                return True, cleaned.capitalize(), None
        elif field_name in ("career_guidance_usefulness", "speaker_effectiveness"):
            rating_map = {
                "5": "Excellent",
                "4": "Good",
                "3": "Satisfactory",
                "2": "Fair",
                "1": "Poor",
                "excellent": "Excellent",
                "good": "Good",
                "satisfactory": "Satisfactory",
                "fair": "Fair",
                "poor": "Poor",
            }
            if cleaned in rating_map:
                return True, rating_map[cleaned], None
        return False, cleaned, f"Invalid selection '{val}' for field '{field_name}'"
