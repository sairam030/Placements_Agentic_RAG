import json
import os
import re
from tqdm import tqdm
from collections import defaultdict


GROUPED_FILE = "output/company_role_groups_structural.json"
OUTPUT_FILE = "output/role_contexts.json"

# ---------- LOAD STRUCTURAL GROUP ----------
with open(GROUPED_FILE, "r", encoding="utf-8") as f:
    grouped = json.load(f)

# ---------- HELPERS ----------
JD_PATTERN = re.compile(r"(jd|job description|intern|internship)", re.I)

def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    return "\n".join(lines)

def role_title_from_filename(path: str) -> str:
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    name = re.sub(r"(jd|job description|internship|intern)", "", name, flags=re.I)
    return name.strip()

# ---------- BUILD ROLE CONTEXTS ----------
contexts = []

for company, cdata in tqdm(grouped.items(), desc="Building role contexts"):
    for role_id, rdata in cdata["roles"].items():

        role_blocks = {
            "role_description": [],
            "eligibility": [],
            "stipend_and_dates": [],
            "company_info": []
        }

        source_files = defaultdict(list)

        # 🔑 USE raw_text_blocks (NOT raw_text.json)
        for block in rdata.get("raw_text_blocks", []):
            file_path = block["file_path"]
            text = clean_text(block.get("text", ""))
            if not text:
                continue

            fname = os.path.basename(file_path).lower()

            if JD_PATTERN.search(fname):
                role_blocks["role_description"].append(text)
                source_files["jd_files"].append(file_path)

            elif fname == "info.txt":
                role_blocks["eligibility"].append(text)
                source_files["info_files"].append(file_path)

            elif fname.endswith((".png", ".jpg", ".jpeg")):
                role_blocks["stipend_and_dates"].append(text)
                source_files["poster_files"].append(file_path)

            else:
                role_blocks["company_info"].append(text)
                source_files["other_files"].append(file_path)

        # Skip empty roles
        if not any(role_blocks.values()):
            continue

        contexts.append({
            "company": company,
            "role_id": role_id,
            "role_title_hint": (
                role_title_from_filename(source_files["jd_files"][0])
                if source_files.get("jd_files")
                else role_id.replace("_", " ")
            ),
            "context": {
                "role_description": "\n\n".join(role_blocks["role_description"]),
                "eligibility": "\n\n".join(role_blocks["eligibility"]),
                "stipend_and_dates": "\n\n".join(role_blocks["stipend_and_dates"]),
                "company_info": "\n\n".join(role_blocks["company_info"])
            },

            # 🔥 CRITICAL: PASS FACTS FORWARD
            "facts": rdata.get("facts", {}),

            "source_files": dict(source_files)
        })

# ---------- SAVE ----------
os.makedirs("output", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(contexts, f, indent=2, ensure_ascii=False)

print(f"✅ Built role contexts: {len(contexts)}")
print(f"📁 Saved to {OUTPUT_FILE}")
