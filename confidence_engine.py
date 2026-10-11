import re

class ConfidenceEngine:
    def __init__(self, high_thresh=0.88, med_thresh=0.70):
        self.high_thresh = high_thresh
        self.med_thresh = med_thresh

    def evaluate_mobile_number(self, digit_predictions):
        """
        Evaluates 10-digit mobile number predictions (Priority #1).
        Indian mobile numbers must be 10 digits, starting with 6, 7, 8, or 9.
        """
        raw_digits = [p["predicted_char"] for p in digit_predictions]
        confidences = [p["confidence"] for p in digit_predictions]

        mobile_str = "".join(raw_digits)
        min_conf = min(confidences) if confidences else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Business validation checks
        is_len_valid = (len(mobile_str) == 10)
        is_prefix_valid = bool(re.match(r'^[6-9]', mobile_str)) if mobile_str else False
        is_numeric = mobile_str.isdigit()

        is_valid = is_len_valid and is_prefix_valid and is_numeric

        # Confidence level assignment
        if is_valid and min_conf >= 0.85 and avg_conf >= self.high_thresh:
            conf_level = "HIGH"
            needs_review = False
            reason = "Valid 10-digit mobile number with high AI confidence"
        elif is_valid and avg_conf >= self.med_thresh:
            conf_level = "MEDIUM"
            needs_review = True
            reason = "Valid mobile number format, but borderline character confidence"
        else:
            conf_level = "LOW"
            needs_review = True
            reasons = []
            if not is_len_valid: reasons.append(f"Invalid length ({len(mobile_str)} digits)")
            if not is_prefix_valid: reasons.append(f"Invalid prefix '{mobile_str[:1]}'")
            if not is_numeric: reasons.append("Contains non-digit characters")
            if min_conf < 0.70: reasons.append(f"Low digit confidence (min: {min_conf:.2f})")
            reason = "; ".join(reasons) if reasons else "Low confidence mobile number"

        return {
            "field_name": "mobile_number",
            "priority": 1,
            "value": mobile_str,
            "is_valid": is_valid,
            "confidence_level": conf_level,
            "avg_confidence": round(avg_conf, 4),
            "min_confidence": round(min_conf, 4),
            "needs_review": needs_review,
            "reason": reason,
            "digits": digit_predictions
        }

    def evaluate_name(self, char_predictions):
        """
        Evaluates Student Full Name (Priority #2).
        """
        chars = [p["predicted_char"] for p in char_predictions]
        confidences = [p["confidence"] for p in char_predictions]
        
        name_str = "".join(chars).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        is_valid = len(name_str) > 0
        if is_valid and avg_conf >= self.high_thresh:
            conf_level = "HIGH"
            needs_review = False
            reason = "Valid name with high confidence"
        elif is_valid and avg_conf >= self.med_thresh:
            conf_level = "MEDIUM"
            needs_review = True
            reason = "Borderline character confidence"
        else:
            conf_level = "LOW"
            needs_review = True
            reason = "Low confidence or empty name"

        return {
            "field_name": "student_name",
            "priority": 2,
            "value": name_str,
            "is_valid": is_valid,
            "confidence_level": conf_level,
            "avg_confidence": round(avg_conf, 4),
            "needs_review": needs_review,
            "reason": reason
        }

    def evaluate_cutoff_mark(self, cutoff_predictions):
        """
        Evaluates Cutoff Mark (Priority #3). Valid range: 0.00 to 200.00.
        """
        raw_chars = [p["predicted_char"] if p["type"] == "box" else "." for p in cutoff_predictions]
        confidences = [p["confidence"] if p["type"] == "box" else 1.0 for p in cutoff_predictions]

        cutoff_str = "".join(raw_chars)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        try:
            val = float(cutoff_str)
            is_valid = (0.0 <= val <= 200.0)
            reason_suffix = f"Value: {val:.2f}"
        except ValueError:
            is_valid = False
            val = None
            reason_suffix = f"Invalid numeric format '{cutoff_str}'"

        if is_valid and avg_conf >= self.high_thresh:
            conf_level = "HIGH"
            needs_review = False
            reason = f"Valid cutoff ({reason_suffix})"
        elif is_valid and avg_conf >= self.med_thresh:
            conf_level = "MEDIUM"
            needs_review = True
            reason = f"Borderline confidence cutoff ({reason_suffix})"
        else:
            conf_level = "LOW"
            needs_review = True
            reason = f"Cutoff validation failed: {reason_suffix}"

        return {
            "field_name": "cutoff_mark",
            "priority": 3,
            "value": cutoff_str,
            "numeric_value": val,
            "is_valid": is_valid,
            "confidence_level": conf_level,
            "avg_confidence": round(avg_conf, 4),
            "needs_review": needs_review,
            "reason": reason
        }

    def evaluate_school_name(self, char_predictions):
        """
        Evaluates School Name (Priority #4).
        """
        chars = [p["predicted_char"] for p in char_predictions]
        confidences = [p["confidence"] for p in char_predictions]
        
        sch_str = "".join(chars).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        is_valid = len(sch_str) > 0
        if is_valid and avg_conf >= self.high_thresh:
            conf_level = "HIGH"
            needs_review = False
            reason = "High confidence school name"
        else:
            conf_level = "MEDIUM" if avg_conf >= self.med_thresh else "LOW"
            needs_review = True
            reason = "Requires review"

        return {
            "field_name": "school_name",
            "priority": 4,
            "value": sch_str,
            "is_valid": is_valid,
            "confidence_level": conf_level,
            "avg_confidence": round(avg_conf, 4),
            "needs_review": needs_review,
            "reason": reason
        }

    def evaluate_omr(self, omr_results):
        """
        Evaluates OMR Feedback questions (Priority #5).
        """
        unanswered = 0
        multiple = 0
        review = 0
        confidences = []

        summary = {}
        for q_id, q_data in omr_results.items():
            summary[q_id] = q_data["selected_option"]
            confidences.append(q_data["confidence"])
            if q_data["status"] == "UNANSWERED": unanswered += 1
            elif q_data["status"] == "MULTIPLE": multiple += 1
            elif q_data["status"] == "REVIEW": review += 1

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        is_clean = (multiple == 0 and review == 0 and unanswered == 0)

        if is_clean and avg_conf >= self.high_thresh:
            conf_level = "HIGH"
            needs_review = False
            reason = "All 5 OMR feedback questions answered unambiguously"
        else:
            conf_level = "LOW" if (multiple > 0 or review > 0) else "MEDIUM"
            needs_review = True
            reason = f"OMR issues: {multiple} multiple, {unanswered} unanswered, {review} review needed"

        return {
            "field_name": "omr_feedback",
            "priority": 5,
            "value": summary,
            "is_valid": is_clean,
            "confidence_level": conf_level,
            "avg_confidence": round(avg_conf, 4),
            "needs_review": needs_review,
            "reason": reason,
            "details": omr_results
        }

    def evaluate_form(self, mobile_eval, name_eval, cutoff_eval, school_eval, omr_eval):
        """
        Aggregates overall form status.
        """
        all_evals = [mobile_eval, name_eval, cutoff_eval, school_eval, omr_eval]
        
        # Priority ranking sorting
        all_evals.sort(key=lambda x: x["priority"])

        any_needs_review = any(e["needs_review"] for e in all_evals)
        mobile_needs_review = mobile_eval["needs_review"]

        if not any_needs_review:
            form_status = "AUTO_ACCEPTED"
        elif mobile_needs_review:
            form_status = "NEEDS_REVIEW_MOBILE_PRIORITY"
        else:
            form_status = "NEEDS_REVIEW"

        return {
            "form_status": form_status,
            "auto_accepted": not any_needs_review,
            "fields": {
                "mobile_number": mobile_eval,
                "student_name": name_eval,
                "cutoff_mark": cutoff_eval,
                "school_name": school_eval,
                "omr_feedback": omr_eval
            }
        }

if __name__ == "__main__":
    engine = ConfidenceEngine()
    dummy_mobile_preds = [{"predicted_char": "9", "confidence": 0.95} for _ in range(10)]
    res = engine.evaluate_mobile_number(dummy_mobile_preds)
    print("Mobile evaluation test:", res)
