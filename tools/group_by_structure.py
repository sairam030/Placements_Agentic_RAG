import os
import json

BASE_DIR = "placements_data/Placements"
OUTPUT_FILE = "output/company_role_groups_structural.json"

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
    ]

    # CASE 1: Company has role folders
    if subdirs:
        for role in subdirs:
            role_path = os.path.join(company_path, role)
            files = [
                os.path.join(role_path, f)
                for f in os.listdir(role_path)
                if os.path.isfile(os.path.join(role_path, f))
            ]

            result[company]["roles"][role] = {
                "role_name": role,
                "files": files
            }

    # CASE 2: No role folders → default role
    else:
        files = [
            os.path.join(company_path, f)
            for f in os.listdir(company_path)
            if os.path.isfile(os.path.join(company_path, f))
        ]

        result[company]["roles"]["default_role"] = {
            "role_name": "default_role",
            "files": files
        }

os.makedirs("output", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"✅ Structural grouping completed for {len(result)} companies")
