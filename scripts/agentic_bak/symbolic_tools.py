import json

CLEAN_FILE = "output/clean_semantic_chunks.json"

with open(CLEAN_FILE, "r", encoding="utf-8") as f:
    DATA = json.load(f)

def list_all_companies():
    return sorted(set(entry["company"] for entry in DATA))

def count_companies():
    return len(set(entry["company"] for entry in DATA))

def get_company_stipend(company_name):
    for entry in DATA:
        if entry["company"].lower() == company_name.lower():
            return entry.get("stipend")
    return None
