import json
import re
import hashlib
from typing import List

import torch
from sentence_transformers import SentenceTransformer, util

# ================= PATHS ================= #

INPUT_FILE = "output/model_based_roles.json"
OUTPUT_FILE = "output/clean_semantic_chunks.json"

# ================= MODEL ================= #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = SentenceTransformer(
    "all-mpnet-base-v2",
    device=DEVICE
)

SEMANTIC_DUP_THRESHOLD = 0.92

# ================= HELPERS ================= #

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(NONE\s*-+.*?$)", "", text, flags=re.I)
    text = re.sub(r"The text that belongs to.*$", "", text, flags=re.I)
    return text.strip()

def sentence_split(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text)

def semantic_dedup(sentences: List[str]) -> List[str]:
    if len(sentences) <= 1:
        return sentences

    embeds = EMBED_MODEL.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    keep, keep_embeds = [], []

    for i, s in enumerate(sentences):
        if not keep:
            keep.append(s)
            keep_embeds.append(embeds[i])
            continue

        sims = util.cos_sim(embeds[i], torch.stack(keep_embeds))[0]
        if torch.max(sims).item() < SEMANTIC_DUP_THRESHOLD:
            keep.append(s)
            keep_embeds.append(embeds[i])

    return keep

# ================= MAIN ================= #

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []

for entry in data:
    role_seen = set()   # ✅ role-level dedup ONLY
    final_chunks = []

    for c in entry["chunks"]:
        raw = c.get("text", "")
        if not raw:
            continue

        text = normalize_text(raw)
        if len(text) < 40:
            continue

        sentences = semantic_dedup(sentence_split(text))
        text = " ".join(sentences)

        h = hashlib.md5(text.encode()).hexdigest()
        if h in role_seen:
            continue
        role_seen.add(h)

        final_chunks.append({
            "chunk_type": c["chunk_type"],
            "text": text
        })

    if not final_chunks:
        continue

    cleaned.append({
        "company": entry["company"],
        "role_id": entry["role_id"],
        "role_name": entry["role_name"],
        "role_category": entry["role_category"],
        "confidence": entry.get("confidence", 0.6),

        # ✅ TRUST facts — DO NOT recompute
        "facts": entry.get("facts", {}),

        "chunks": final_chunks,
        "source_files": entry.get("source_files", {})
    })

# ================= SAVE ================= #

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print(f"✅ Cleaned roles: {len(cleaned)}")
print(f"📁 Saved to {OUTPUT_FILE}")
