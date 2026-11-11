"""Deterministic intent to department mapping. First matching rule wins."""
RULES = [
    (("verify", "identity", "source_of_funds"), "KYC and Verification Support"),
    (("refund",), "Refund Support"),
    (("cash", "atm"), "Cash and ATM Support"),
    (("top_up", "payment", "direct_debit", "exchange", "charge", "apple_pay"), "Payments Support"),
    (("transfer", "receiving_money", "beneficiary"), "Transfers Support"),
    (("card", "pin", "contactless", "visa"), "Card Support"),
    (("account", "personal_details", "passcode", "phone"), "Account Support"),
]
DEFAULT = "General Customer Support"


def department_for(intent: str) -> str:
    for keywords, department in RULES:
        if any(k in intent.lower() for k in keywords):
            return department
    return DEFAULT
