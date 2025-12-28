import os
import json
import re
from tqdm import tqdm
import pdfplumber
from docx import Document
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

DATA_DIR = "placements_data"
OUTPUT_FILE = "output/raw_text.json"

print("🔹 Loading DocTR model (GPU)...")
ocr_model = ocr_predictor(pretrained=True)

# ================== TEXT EXTRACTORS ================== #

def extract_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_image_doctr(path):
    doc = DocumentFile.from_images(path)
    result = ocr_model(doc)
    return result.render()

# ================== IMAGE FACT EXTRACTION ================== #

def extract_image_facts(text: str):
    facts = {}

    t = text.lower()

    # ---- STIPEND ----
    stipend_patterns = [
        r'₹?\s*(\d{1,3})\s*t\s*pm',        # 30T PM
        r'₹?\s*(\d{2,3})\s*k\s*pm',        # 40K PM
        r'₹?\s*(\d{4,6})\s*(pm|per month)' # 30000 PM
    ]

    for pat in stipend_patterns:
        m = re.search(pat, t)
        if m:
            amt = int(m.group(1)) * (1000 if int(m.group(1)) < 1000 else 1)
            facts["stipend"] = {
                "value": amt,
                "currency": "INR",
                "period": "month",
                "confidence": 0.95,
                "source": "image"
            }
            break

    # ---- DURATION ----
    m = re.search(r'duration\s*[:\-]?\s*(\d{1,2})\s*month', t)
    if m:
        facts["duration"] = {
            "months": int(m.group(1)),
            "confidence": 0.95,
            "source": "image"
        }

    # ---- LOCATION ----
    m = re.search(r'\b(bangalore|bengaluru|hyderabad|pune|chennai|delhi|mumbai)\b', t)
    if m:
        facts["location"] = {
            "value": m.group(1).title(),
            "confidence": 0.9,
            "source": "image"
        }

    return facts

# ================== MAIN ================== #

records = []

for root, _, files in os.walk(DATA_DIR):
    for file in tqdm(files, desc="Extracting text"):
        path = os.path.join(root, file)
        try:
            text = ""
            ftype = None
            image_facts = {}

            if file.lower().endswith(".pdf"):
                text = extract_pdf(path)
                ftype = "pdf"

            elif file.lower().endswith(".docx"):
                text = extract_docx(path)
                ftype = "docx"

            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                text = extract_image_doctr(path)
                image_facts = extract_image_facts(text)
                ftype = "image"

            elif file.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                ftype = "txt"

            else:
                continue

            if text.strip():
                records.append({
                    "file_path": path,
                    "file_type": ftype,
                    "text": text,
                    "image_facts": image_facts
                })

        except Exception as e:
            print(f"❌ Error in {path}: {e}")

os.makedirs("output", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted text from {len(records)} files")
