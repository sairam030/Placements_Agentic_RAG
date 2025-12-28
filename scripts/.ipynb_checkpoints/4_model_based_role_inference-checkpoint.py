import json
import torch
from sentence_transformers import SentenceTransformer, util

# ================= PATHS ================= #

INPUT_FILE = "output/semantic_chunks.json"
OUTPUT_FILE = "output/model_based_roles.json"

# ================= DEVICE ================= #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading embedding model...")
model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=DEVICE
)
print("✅ Model loaded on", DEVICE)

# ================= ROLE TEMPLATES ================= #

ROLE_TEMPLATES = {

    # ---------- DATA / AI ----------
    "data_science_intern": {
        "name": "Data Science Intern",
        "category": "data",
        "text": (
            "data science machine learning ai statistics python pandas numpy "
            "model training evaluation analytics"
        )
    },

    "ml_ai_intern": {
        "name": "ML / AI Intern",
        "category": "data",
        "text": (
            "machine learning deep learning pytorch tensorflow "
            "computer vision nlp transformers generative ai"
        )
    },

    "computer_vision_intern": {
        "name": "Computer Vision Intern",
        "category": "data",
        "text": (
            "computer vision image processing opencv cnn "
            "object detection segmentation"
        )
    },

    # ---------- SOFTWARE ----------
    "software_engineer_intern": {
        "name": "Software Engineer Intern",
        "category": "software",
        "text": (
            "software engineering programming algorithms "
            "data structures system design development"
        )
    },

    "backend_intern": {
        "name": "Backend Developer Intern",
        "category": "software",
        "text": (
            "backend development python java nodejs "
            "django fastapi rest api databases"
        )
    },

    "fullstack_intern": {
        "name": "Full Stack Developer Intern",
        "category": "software",
        "text": (
            "full stack frontend backend react javascript "
            "html css api integration"
        )
    },

    "devops_intern": {
        "name": "DevOps Intern",
        "category": "software",
        "text": (
            "devops ci cd docker kubernetes cloud aws "
            "infrastructure automation"
        )
    },

    # ---------- HARDWARE ----------
    "embedded_systems_intern": {
        "name": "Embedded Systems Intern",
        "category": "hardware",
        "text": (
            "embedded systems microcontroller firmware "
            "c c++ rtos embedded linux"
        )
    },

    "vlsi_intern": {
        "name": "VLSI / ASIC Intern",
        "category": "hardware",
        "text": (
            "vlsi asic digital design verilog "
            "systemverilog rtl synthesis"
        )
    },

    # ---------- CORE ----------
    "mechanical_engineering_intern": {
        "name": "Mechanical Engineering Intern",
        "category": "core",
        "text": (
            "mechanical engineering cad solidworks "
            "manufacturing materials mechanics"
        )
    },

    "electrical_engineering_intern": {
        "name": "Electrical Engineering Intern",
        "category": "core",
        "text": (
            "electrical engineering power systems "
            "control instrumentation motors"
        )
    },

    # ---------- BUSINESS ----------
    "business_analyst_intern": {
        "name": "Business Analyst Intern",
        "category": "business",
        "text": (
            "business analysis strategy operations "
            "stakeholder reporting dashboards"
        )
    },

    # ---------- SECURITY ----------
    "security_intern": {
        "name": "Security Intern",
        "category": "security",
        "text": (
            "cyber security product security "
            "vulnerability assessment secure systems"
        )
    },

    # ---------- RESEARCH ----------
    "research_intern": {
        "name": "Research Intern",
        "category": "research",
        "text": (
            "research experimentation algorithms "
            "publications innovation prototyping"
        )
    },

    # ---------- FALLBACK ----------
    "general_intern": {
        "name": "General Engineering Intern",
        "category": "general",
        "text": (
            "engineering intern problem solving "
            "projects teamwork interdisciplinary"
        )
    }
}

# Precompute template embeddings
template_texts = [v["text"] for v in ROLE_TEMPLATES.values()]
template_embeddings = model.encode(
    template_texts,
    normalize_embeddings=True
)

# ================= HELPERS ================= #

def get_role_text(chunks):
    """Prefer role_overview, fallback to skills + eligibility."""
    for c in chunks:
        if c["chunk_type"] == "role_overview":
            return c["text"]

    fallback = [
        c["text"]
        for c in chunks
        if c["chunk_type"] in ("skills", "eligibility", "additional_info")
    ]

    return " ".join(fallback) if fallback else None

# ================= MAIN ================= #

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for entry in data:
    company = entry["company"]
    original_role_id = entry.get("role_id", "default_role")

    role_text = get_role_text(entry["chunks"])
    if not role_text:
        continue

    role_embedding = model.encode(
        role_text,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(role_embedding, template_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    # Confidence gate
    if best_score < 0.45:
        role_id = "general_intern"
    else:
        role_id = list(ROLE_TEMPLATES.keys())[best_idx]

    role_meta = ROLE_TEMPLATES[role_id]

    results.append({
        "company": company,
        "original_role_id": original_role_id,
        "role_id": role_id,
        "role_name": role_meta["name"],
        "role_category": role_meta["category"],
        "confidence": round(best_score, 3),
        "chunks": entry["chunks"],
        "facts": entry.get("facts", {}),
        "source_files": entry.get("source_files", {})
    })

# ================= SAVE ================= #

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Model-based roles generated: {len(results)}")
print(f"📁 Saved to {OUTPUT_FILE}")
