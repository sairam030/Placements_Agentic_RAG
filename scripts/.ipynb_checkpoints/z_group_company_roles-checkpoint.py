import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = "placements_data/Placements"
OUTPUT_FILE = "output/company_role_groups.json"

JD_PATTERN = re.compile(
    r"(jd|job description|intern|internship)", re.IGNORECASE
)

def normalize(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

final_output = {}

print("🔍 Scanning placements structure...")

for company in tqdm(os.listdir(BASE_DIR)):
    company_path = os.path.join(BASE_DIR, company)
    if not os.path.isdir(company_path):
        continue

    roles = defaultdict(list)

    # 1️⃣ Detect Role-* folders
    role_dirs = [
        d for d in os.listdir(company_path)
        if os.path.isdir(os.path.join(company_path, d))
        and re.search(r"role", d, re.IGNORECASE)
    ]

    if role_dirs:
        # -------- Pattern 1 --------
        for rd in role_dirs:
            role_key = normalize(rd)
            role_path = os.path.join(company_path, rd)

            for root, _, files in os.walk(role_path):
                for f in files:
                    roles[role_key].append(os.path.join(root, f))

    else:
        # -------- Pattern 2 / 3 --------
        files = [
            os.path.join(company_path, f)
            for f in os.listdir(company_path)
            if os.path.isfile(os.path.join(company_path, f))
        ]

        jd_files = [f for f in files if JD_PATTERN.search(os.path.basename(f))]

        if len(jd_files) > 1:
            # Each JD = one role
            for jd in jd_files:
                role_key = normalize(os.path.splitext(os.path.basename(jd))[0])
                roles[role_key].append(jd)

                # attach common files
                for f in files:
                    if f not in jd_files:
                        roles[role_key].append(f)
        elif len(jd_files) == 1:
            roles["general_role"] = files
        else:
            roles["general_role"] = files

    final_output[company] = {
        "company_id": company,
        "roles": {}
    }

    for role_key, files in roles.items():
        final_output[company]["roles"][role_key] = {
            "role_name": role_key.replace("_", " "),
            "files": sorted(set(files)),
            "confidence": "high" if role_key != "general_role" else "medium"
        }

# Save
os.makedirs("output", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2)

print(f"✅ Grouping completed for {len(final_output)} companies")
print(f"📁 Output saved to {OUTPUT_FILE}")
