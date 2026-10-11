import pytest
from backend.app.decision_engine.comparator import DecisionEngine
from backend.app.validation.validator import FieldValidator

def test_case_1_gemini_equals_cnn_accept():
    gemini_val = "9384064560"
    cnn_res = {"value": "9384064560", "confidence": 0.98, "status": "COMPLETED"}
    res = DecisionEngine.evaluate_phone_number(gemini_val, cnn_res)
    assert res["decision"] == "AGREEMENT"
    assert res["final_value"] == "9384064560"
    assert res["confidence"] >= 0.95

def test_case_2_gemini_differs_cnn_conflict():
    gemini_val = "9384064560"
    cnn_res = {"value": "9384064568", "confidence": 0.70, "status": "COMPLETED"}
    res = DecisionEngine.evaluate_phone_number(gemini_val, cnn_res)
    assert res["decision"] in ("CONFLICT", "MANUAL_REVIEW")
    assert 10 in res["differing_positions"]

def test_case_3_cnn_unavailable_gemini_retained():
    gemini_val = "9384064560"
    cnn_res = {"value": None, "confidence": 0.0, "status": "CNN_UNAVAILABLE"}
    res = DecisionEngine.evaluate_phone_number(gemini_val, cnn_res)
    assert res["decision"] == "CNN_UNAVAILABLE"
    assert res["final_value"] == "9384064560"

def test_case_4_both_uncertain_manual_review():
    gemini_val = "abc12345"
    cnn_res = {"value": "12345", "confidence": 0.30, "status": "COMPLETED"}
    res = DecisionEngine.evaluate_phone_number(gemini_val, cnn_res)
    assert res["decision"] == "MANUAL_REVIEW"

def test_case_5_phone_has_9_digits_invalid():
    is_valid, cleaned, err = FieldValidator.validate_phone_number("938406456")
    assert is_valid is False
    assert "must be exactly 10 digits" in err

def test_case_6_cutoff_250_invalid():
    is_valid, cleaned, err = FieldValidator.validate_cutoff_marks("250")
    assert is_valid is False
    assert "out of valid range" in err

def test_case_7_cutoff_184_56_valid():
    is_valid, cleaned, err = FieldValidator.validate_cutoff_marks("184.56")
    assert is_valid is True
    assert cleaned == "184.56"
    assert err is None

def test_case_8_cutoff_167_5_valid():
    is_valid, cleaned, err = FieldValidator.validate_cutoff_marks("167.5")
    assert is_valid is True
    assert cleaned == "167.5"

def test_cutoff_decision_engine_agreement():
    gemini_cutoff = "184.56"
    cnn_cutoff_res = {"value": "184.56", "confidence": 0.97, "status": "COMPLETED"}
    res = DecisionEngine.evaluate_cutoff_marks(gemini_cutoff, cnn_cutoff_res)
    assert res["decision"] == "AGREEMENT"
    assert res["final_value"] == "184.56"

def test_case_9_localization_failed_triggers_manual_review():
    gemini_val = "9432164321"
    cnn_res = {"value": None, "confidence": 0.0, "status": "LOCALIZATION_FAILED"}
    res = DecisionEngine.evaluate_phone_number(gemini_val, cnn_res)
    assert res["decision"] == "MANUAL_REVIEW_REQUIRED"
    assert "localization failed" in res["reason"].lower()

