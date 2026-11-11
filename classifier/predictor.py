import json
from classifier.routing import department_for
from classifier.schemas import TicketResult


def load_labels(path="data/labels.json"):
    with open(path) as f:
        return json.load(f)


def triage(text: str, generate, labels: list[str]) -> TicketResult:
    """generate: function that maps a message to the raw model output string."""
    text = (text or "").strip()
    if not text:
        return TicketResult(input_text=text, intent=None, department=None,
                            requires_human_review=True, validation_status="empty_input")
    raw = generate(text).strip().split("\n")[0].strip()
    if raw not in labels:
        return TicketResult(input_text=text, intent=None, department=None,
                            requires_human_review=True, validation_status="invalid_prediction")
    return TicketResult(input_text=text, intent=raw, department=department_for(raw),
                        requires_human_review=False, validation_status="valid")
