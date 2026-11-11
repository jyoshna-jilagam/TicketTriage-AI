from pydantic import BaseModel


class TicketResult(BaseModel):
    input_text: str
    intent: str | None
    department: str | None
    requires_human_review: bool
    validation_status: str  # valid, invalid_prediction, empty_input
