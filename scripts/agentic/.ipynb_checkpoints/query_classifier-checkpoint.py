import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ---------------- #

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPANY_LIST_PATH = "output/facts/facts.json"

# ---------------- LOAD MODEL ---------------- #

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ---------------- LOAD COMPANIES ---------------- #

with open(COMPANY_LIST_PATH, "r", encoding="utf-8") as f:
    COMPANY_LIST = json.load(f)

# normalized lookup
COMPANY_MAP = {
    c.lower().replace("_mtech_2026", ""): c
    for c in COMPANY_LIST
}

# ---------------- HELPERS ---------------- #

def extract_company(question: str):
    q = question.lower()
    for key, canonical in COMPANY_MAP.items():
        if key in q:
            return canonical
    return None


# ---------------- PROMPT ---------------- #

PROMPT = """
Classify the user question into a retrieval route.

Routes:
- symbolic: counts, lists, stipend, duration, cgpa, location, dates
- semantic: explanation, role details, eligibility process, description
- hybrid: mix of both

Return STRICT JSON only.

Question: "{question}"

JSON:
"""

# ---------------- CLASSIFIER ---------------- #

@torch.inference_mode()
# scripts/agentic/query_classifier.py

def classify_query(question: str):
    """
    Intent-only classifier.
    No company extraction.
    No facts access.
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
