# scripts/build_faiss_index.py

import os
import json
import numpy as np
import faiss

# ---------------- CONFIG ---------------- #

EMBEDDINGS_PATH = "output/embeddings/embeddings.npy"
METADATA_PATH = "output/embeddings/metadata.json"
INDEX_OUTPUT_PATH = "output/embeddings/faiss.index"

USE_GPU = True   # set False if faiss-gpu not installed

# ---------------- LOAD DATA ---------------- #

print("📥 Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

print("📄 Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

assert embeddings.shape[0] == len(metadata), "❌ Embeddings & metadata size mismatch"

num_vectors, dim = embeddings.shape
print(f"📐 Embeddings shape: {embeddings.shape}")

# ---------------- BUILD INDEX ---------------- #

print("⚙️ Building FAISS index (cosine similarity)...")

base_index = faiss.IndexFlatIP(dim)
index = faiss.IndexIDMap(base_index)

# Use stable integer IDs
ids = np.arange(num_vectors).astype("int64")

# ---------------- GPU OPTION ---------------- #

if USE_GPU:
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("🚀 Using GPU FAISS")
    except Exception as e:
        print(f"⚠️ GPU FAISS not available, using CPU\n{e}")

index.add_with_ids(embeddings, ids)
print(f"✅ Added {index.ntotal} vectors to index")

# ---------------- SAVE INDEX ---------------- #

os.makedirs(os.path.dirname(INDEX_OUTPUT_PATH), exist_ok=True)

# Always save CPU index
if isinstance(index, faiss.Index):
    try:
        index = faiss.index_gpu_to_cpu(index)
    except Exception:
        pass

faiss.write_index(index, INDEX_OUTPUT_PATH)
print(f"💾 FAISS index saved to: {INDEX_OUTPUT_PATH}")

# ---------------- SANITY CHECK ---------------- #

print("🔍 Running sanity check...")

query_vec = embeddings[0:1]
D, I = index.search(query_vec, 5)

print("Top matches:")
for rank, idx in enumerate(I[0]):
    meta = metadata[idx]
    print(
        f"{rank+1}. "
        f"{meta['company']} | "
        f"{meta['role_name']} | "
        f"{meta['chunk_type']}"
    )

print("✅ FAISS index build complete")
