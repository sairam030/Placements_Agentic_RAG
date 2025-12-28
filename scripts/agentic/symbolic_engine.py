import json
import re
from collections import defaultdict

FACTS_PATH = "output/facts/facts.json"

with open(FACTS_PATH, "r", encoding="utf-8") as f:
    FACTS = json.load(f)

# ---------------- NORMALIZATION ---------------- #

def normalize_company(name: str):
    return (
        name.lower()
        .replace("_mtech_2026", "")
        .replace("_", "")
        .strip()
    )

# build lookup once
COMPANY_LOOKUP = defaultdict(list)
for f in FACTS:
    key = normalize_company(f["company"])
    COMPANY_LOOKUP[key].append(f)

# ---------------- HELPERS ---------------- #

def resolve_company(question: str):
    q = question.lower()
    for f in FACTS:
        company = f["company"]
        key = company.lower().replace("_mtech_2026", "")
        if key in q:
            return company
    return None

def safe_unique(values):
    return sorted({v for v in values if v not in [None, "", []]})

# ---------------- SYMBOLIC ENGINE ---------------- #

def symbolic_search(parsed, question):
    intent = parsed["intent"]

    # ---- COUNT COMPANIES ----
    if intent == "count_companies":
        return len(COMPANY_LOOKUP)

    # ---- LIST COMPANIES ----
    if intent == "list_companies":
        return sorted({f["company"] for f in FACTS})

    # ---- COMPANY-SPECIFIC ----
    company_key = resolve_company(question)
    if not company_key:
        return None  # fallback to semantic

    records = COMPANY_LOOKUP.get(company_key, [])
    if not records:
        return None

    # ---- STIPEND ----
    if intent == "stipend":
        stipends = safe_unique(f.get("stipend") for f in records)
        return stipends if stipends else None

    # ---- CGPA ----
    if intent == "cgpa":
        cgpas = safe_unique(f.get("cgpa_min") for f in records)
        return cgpas if cgpas else None

    # ---- DURATION ----
    if intent == "duration":
        durations = safe_unique(f.get("duration_months") for f in records)
        return durations if durations else None

    # ---- LOCATION ----
    if intent == "location":
        locations = set()
        for f in records:
            for loc in f.get("location", []):
                locations.add(loc)
        return sorted(locations) if locations else None

    # ---- APPLY BY ----
    if intent == "dates":
        dates = set()
        for f in records:
            for d in f.get("apply_by", []):
                dates.add(d)
        return sorted(dates) if dates else None

    return None
