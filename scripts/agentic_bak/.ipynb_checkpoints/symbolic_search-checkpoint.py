import json

FACTS_FILE = "output/facts/facts.json"

with open(FACTS_FILE, "r", encoding="utf-8") as f:
    FACTS = json.load(f)

def symbolic_search(company=None, role=None, fields=None):
    results = []

    for item in FACTS:
        if company and company.lower() not in item["company"].lower():
            continue
        if role and role.lower() not in item["role"].lower():
            continue

        extracted = {}
        for field in fields:
            value = item.get(field)
            if value:
                extracted[field] = value

        if extracted:
            results.append({
                "company": item["company"],
                "role": item["role"],
                **extracted
            })

    return results
