import json
import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #

INPUT_FILE = "output/clean_semantic_chunks.json"
OUTPUT_DIR = "output/embeddings"
MODEL_ID = "BAAI/bge-large-en-v1.5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EMB_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
META_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

# Do NOT embed pure facts
SKIP_CHUNKS = {
    "stipend_and_duration",
    "important_dates",
    "location"
}

# ---------------- LOAD MODEL ---------------- #

print("🚀 Loading BGE model...")
model = SentenceTransformer(
    MODEL_ID,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("✅ Model loaded")

# ---------------- LOAD DATA ---------------- #

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
metadata = []

# ---------------- BUILD EMBEDDING INPUT ---------------- #

chunk_counter = 0

for entry in data:
    company = entry["company"]
    role_id = entry["role_id"]
    role_name = entry["role_name"]
    category = entry["role_category"]

    for chunk in entry["chunks"]:
        if chunk["chunk_type"] in SKIP_CHUNKS:
            continue

        chunk_text = chunk["text"].strip()
        if len(chunk_text) < 40:
            continue

        context_text = (
            "This is a job description passage.\n"
            f"Company: {company}\n"
            f"Role: {role_name}\n"
            f"Category: {category}\n"
            f"Section: {chunk['chunk_type']}\n"
            f"Text: {chunk_text}"
        )

        texts.append("passage: " + context_text)

        metadata.append({
            "chunk_id": f"chunk_{chunk_counter}",
            "company": company,
            "role_id": role_id,
            "role_name": role_name,
            "category": category,
            "chunk_type": chunk["chunk_type"],
            "text": chunk_text
        })

        chunk_counter += 1

print(f"📄 Prepared {len(texts)} semantic chunks for embedding")

# ---------------- EMBEDDING ---------------- #

print("⚙️ Generating embeddings (GPU)...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# ---------------- SAVE ---------------- #

np.save(EMB_FILE, embeddings)

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Embeddings saved: {EMB_FILE}")
print(f"✅ Metadata saved: {META_FILE}")
print(f"📐 Embedding shape: {embeddings.shape}")
