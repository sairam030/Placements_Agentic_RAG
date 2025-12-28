# scripts/agentic/query_classifier.py

def classify_query(question: str):
    """
    Intent-only classifier.
    NO company extraction.
    NO facts access.
    """

    q = question.lower()

    # ---- COMPANY META ----
    if "how many" in q and "company" in q:
        return {"route": "symbolic", "intent": "count_companies"}

    if "list" in q and "company" in q:
        return {"route": "symbolic", "intent": "list_companies"}

    # ---- FACT-BASED ----
    if "stipend" in q:
        return {"route": "symbolic", "intent": "stipend"}

    if "cgpa" in q or "eligibility" in q:
        return {"route": "symbolic", "intent": "cgpa"}

    if "duration" in q:
        return {"route": "symbolic", "intent": "duration"}

    if "location" in q:
        return {"route": "symbolic", "intent": "location"}

    if "date" in q or "apply" in q:
        return {"route": "symbolic", "intent": "dates"}

    # ---- DEFAULT ----
    return {"route": "semantic"}
