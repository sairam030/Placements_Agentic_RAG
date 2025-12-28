# scripts/extract_structured_facts.py

import json
import re
import os
from collections import defaultdict

# ================= PATHS ================= #

INPUT_FILE = "output/clean_semantic_chunks.json"
OUTPUT_FILE = "output/facts/facts.json"

# ================= REGEX ================= #

RE_STIPEND = re.compile(
    r'(₹|rs\.?|inr|€|e)?\s*'
    r'(\d{1,3}(?:\s?\d{2,3})?)\s*'
    r'(t|k|tp?m|thousand)?',
    re.I
)

RE_DURATION = re.compile(r'(\d{1,2})\s*(month|months)', re.I)
RE_CGPA = re.compile(r'(cgpa|c\.g\.p\.a).*?(\d\.\d)', re.I)

RE_DATE = re.compile(
    r'(\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*|\d{1,2}/\d{1,2}/\d{2,4})',
    re.I
)

RE_LOCATION = re.compile(
    r'\b(bangalore|bengaluru|hyderabad|chennai|pune|mumbai|delhi|noida|'
    r'gurgaon|kolkata|remote|hybrid|onsite|work from home)\b',
    re.I
)

# ================= HELPERS ================= #

def normalize_stipend(text):
    text = text.lower().replace(",", "").replace("€", "").replace("rs.", "")
    m = RE_STIPEND.search(text)
    if not m:
        return None

    num = int(m.group(2).replace(" ", ""))
    unit = m.group(3)

    if unit in ("t", "k", "thousand", "tpm"):
        num *= 1000

    if 5_000 <= num <= 300_000:
        return num
    return None


def normalize_duration(text):
    m = RE_DURATION.search(text)
    if not m:
        return None
    months = int(m.group(1))
    return months if 1 <= months <= 24 else None


# ================= MAIN ================= #

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

facts = []

for entry in data:
    company = entry["company"]
    role = entry["role_name"]

    # ---------------- START WITH EXISTING FACTS ---------------- #

    f = {
        "company": company,
        "role": role,
        "stipend": None,
        "duration_months": None,
        "cgpa_min": None,
        "apply_by": [],
        "location": [],
        "rounds": []
    }

    existing = entry.get("facts", {})

    if isinstance(existing.get("stipend"), dict):
        f["stipend"] = existing["stipend"].get("value")

    if isinstance(existing.get("duration"), dict):
        f["duration_months"] = existing["duration"].get("months")

    if isinstance(existing.get("location"), dict):
        f["location"].append(existing["location"].get("value"))

    # ---------------- FALLBACK FROM CHUNKS ---------------- #

    for chunk in entry["chunks"]:
        text = chunk["text"]
        ctype = chunk["chunk_type"]

        if not f["stipend"]:
            s = normalize_stipend(text)
            if s:
                f["stipend"] = s

        if not f["duration_months"]:
            d = normalize_duration(text)
            if d:
                f["duration_months"] = d

        if not f["cgpa_min"]:
            m = RE_CGPA.search(text.lower())
            if m:
                f["cgpa_min"] = float(m.group(2))

        if ctype in ("location", "role_overview"):
            f["location"].extend(
                x.title() for x in RE_LOCATION.findall(text)
            )

        if ctype == "selection_process":
            f["rounds"].append(text.strip())

        f["apply_by"].extend(m[0] for m in RE_DATE.findall(text))

    # ---------------- CLEAN ---------------- #

    f["apply_by"] = list(set(f["apply_by"]))
    f["location"] = list(set(filter(None, f["location"])))
    f["rounds"] = f["rounds"][:3]

    facts.append(f)

# ================= SAVE ================= #

os.makedirs("output/facts", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(facts, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted facts for {len(facts)} roles")
print(f"📁 Saved to {OUTPUT_FILE}")
