import json, os
import pytest
from classifier.routing import department_for, DEFAULT

LABELS = "data/labels.json"


def test_known_examples():
    assert department_for("declined_cash_withdrawal") == "Cash and ATM Support"
    assert department_for("request_refund") == "Refund Support"
    assert department_for("verify_my_identity") == "KYC and Verification Support"
    assert department_for("lost_or_stolen_card") == "Card Support"


def test_unknown_goes_to_default():
    assert department_for("something_else") == DEFAULT


@pytest.mark.skipif(not os.path.exists(LABELS), reason="run prepare_dataset.py first")
def test_all_77_labels_mapped():
    labels = json.load(open(LABELS))
    assert len(labels) == 77
    assert all(department_for(l) for l in labels)
