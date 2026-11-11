from classifier.predictor import triage

LABELS = ["cash_withdrawal_charge", "request_refund"]


def test_valid_prediction():
    r = triage("Why was I charged for ATM?", lambda t: "cash_withdrawal_charge\nextra", LABELS)
    assert r.validation_status == "valid" and r.department == "Cash and ATM Support"
    assert not r.requires_human_review


def test_invalid_prediction_goes_to_review():
    r = triage("hello", lambda t: "pizza", LABELS)
    assert r.validation_status == "invalid_prediction" and r.requires_human_review


def test_empty_input():
    r = triage("   ", lambda t: "request_refund", LABELS)
    assert r.validation_status == "empty_input"
