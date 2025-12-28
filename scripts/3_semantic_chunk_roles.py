import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= CONFIG ================= #

INPUT_FILE = "output/role_contexts.json"
OUTPUT_FILE = "output/semantic_chunks.json"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_TOKENS = 24000
MAX_NEW_TOKENS = 512

# ================= LOAD MODEL ================= #

print("🚀 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("🚀 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("✅ Model ready")

# ================= HELPERS ================= #

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def truncate_to_tokens(text, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

@torch.inference_mode()
def ask_llm(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================= PROMPT ================= #

CHUNK_PROMPT = """
You are an information extraction system.

Extract ONLY the text that belongs to the section:
"{section}"

Rules:
- Do NOT summarize
- Do NOT invent information
- If missing, return: NONE
- Return ONLY raw extracted text

ROLE TEXT:
----------------
{text}
----------------
"""

SECTIONS = [
    "Company Overview",
    "Role Overview",
    "Eligibility",
    "Skills Required",
    "Selection Process",
    "Location",
    "Additional Information"
]

SECTION_MAP = {
    "Company Overview": "company_overview",
    "Role Overview": "role_overview",
    "Eligibility": "eligibility",
    "Skills Required": "skills",
    "Selection Process": "selection_process",
    "Location": "location",
    "Additional Information": "additional_info"
}

# ================= MAIN ================= #

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    role_items = json.load(f)

output = []

for item in tqdm(role_items, desc="Semantic chunking"):
    company = item["company"]
    role_id = item["role_id"]
    context = item.get("context", {})
    facts = item.get("facts", {})

    # ✅ Build raw text ONLY from context
    raw_text = "\n\n".join([
        context.get("company_info", ""),
        context.get("role_description", ""),
        context.get("eligibility", ""),
        context.get("stipend_and_dates", "")
    ])

    raw_text = clean_text(raw_text)
    if not raw_text:
        continue

    raw_text = truncate_to_tokens(raw_text, MAX_INPUT_TOKENS)

    chunks = []

    for section in SECTIONS:
        prompt = CHUNK_PROMPT.format(section=section, text=raw_text)
        response = ask_llm(prompt)
        response = response.replace(prompt, "").strip()

        if response and response.upper() != "NONE":
            chunks.append({
                "chunk_type": SECTION_MAP[section],
                "text": response
            })

    if not chunks:
        continue

    output.append({
        "company": company,
        "role_id": role_id,
        "role_title_hint": item.get("role_title_hint"),
        "chunks": chunks,
        "facts": facts,              # ✅ pass-through
        "source_files": item.get("source_files", {})
    })

# ================= SAVE ================= #

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n✅ Semantic chunking completed successfully")
print(f"📁 Output saved to {OUTPUT_FILE}")
