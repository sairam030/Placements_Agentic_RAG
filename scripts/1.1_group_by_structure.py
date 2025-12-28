import os
import json
from collections import defaultdict

BASE_DIR = "placements_data/Placements"
RAW_TEXT_FILE = "output/raw_text.json"
OUTPUT_FILE = "output/company_role_groups_structural.json"

# ================= LOAD RAW TEXT ================= #

with open(RAW_TEXT_FILE, "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

# Map file_path → full raw record
raw_map = {d["file_path"]: d for d in raw_docs}

# ================= HELPERS ================= #

def merge_fact(existing, new):
    """Prefer first non-null fact (image > doc > llm later)."""
    return existing if existing is not None else new

# ================= MAIN ================= #

result = {}

for company in sorted(os.listdir(BASE_DIR)):
    company_path = os.path.join(BASE_DIR, company)
    if not os.path.isdir(company_path):
        continue

    result[company] = {
        "company_id": company,
        "roles": {}
    }

    subdirs = [
        d for d in os.listdir(company_path)
        if os.path.isdir(os.path.join(company_path, d))
        and not d.startswith(".")
    ]

    # ==================================================
    # CASE 1: Company has explicit role folders
    # ==================================================
    if subdirs:
        for role in sorted(subdirs):
            role_path = os.path.join(company_path, role)

            files = [
                os.path.join(role_path, f)
                for f in os.listdir(role_path)
                if os.path.isfile(os.path.join(role_path, f))
            ]

            role_block = {
                "role_id": role,
                "role_name": role,
                "files": files,
                "raw_text_blocks": [],
                "facts": {
                    "stipend": None,
                    "duration": None,
                    "location": None,
                    "important_dates": {}
                }
            }

            # 🔑 ENRICH FROM raw_text.json
            for file_path in files:
                raw = raw_map.get(file_path)
                if not raw:
                    continue

                # Attach text
                role_block["raw_text_blocks"].append({
                    "file_path": file_path,
                    "file_type": raw.get("file_type"),
                    "text": raw.get("text", "")
                })

                # Attach image / OCR facts if present
                facts = raw.get("image_facts") or raw.get("facts")
                if facts:
                    role_block["facts"]["stipend"] = merge_fact(
                        role_block["facts"]["stipend"],
                        facts.get("stipend")
                    )
                    role_block["facts"]["duration"] = merge_fact(
                        role_block["facts"]["duration"],
                        facts.get("duration")
                    )
                    role_block["facts"]["location"] = merge_fact(
                        role_block["facts"]["location"],
                        facts.get("location")
                    )

                    if "important_dates" in facts:
                        role_block["facts"]["important_dates"].update(
                            facts["important_dates"]
                        )

            result[company]["roles"][role] = role_block

    # ==================================================
    # CASE 2: No role folders → default_role
    # ==================================================
    else:
        files = [
            os.path.join(company_path, f)
            for f in os.listdir(company_path)
            if os.path.isfile(os.path.join(company_path, f))
        ]

        role_block = {
            "role_id": "default_role",
            "role_name": "default_role",
            "files": files,
            "raw_text_blocks": [],
            "facts": {
                "stipend": None,
                "duration": None,
                "location": None,
                "important_dates": {}
            }
        }

        for file_path in files:
            raw = raw_map.get(file_path)
            if not raw:
                continue

            role_block["raw_text_blocks"].append({
                "file_path": file_path,
                "file_type": raw.get("file_type"),
                "text": raw.get("text", "")
            })

            facts = raw.get("image_facts") or raw.get("facts")
            if facts:
                role_block["facts"]["stipend"] = merge_fact(
                    role_block["facts"]["stipend"],
                    facts.get("stipend")
                )
                role_block["facts"]["duration"] = merge_fact(
                    role_block["facts"]["duration"],
                    facts.get("duration")
                )
                role_block["facts"]["location"] = merge_fact(
                    role_block["facts"]["location"],
                    facts.get("location")
                )

                if "important_dates" in facts:
                    role_block["facts"]["important_dates"].update(
                        facts["important_dates"]
                    )

        result[company]["roles"]["default_role"] = role_block

# ================= SAVE ================= #

os.makedirs("output", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"✅ Structural grouping completed for {len(result)} companies")
print(f"📁 Saved to {OUTPUT_FILE}")
