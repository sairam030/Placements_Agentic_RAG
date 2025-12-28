import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #

FAISS_INDEX_PATH = "output/embeddings/faiss.index"
METADATA_PATH = "output/embeddings/metadata.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
TOP_K = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD ---------------- #

print("🚀 Loading embedding model...")
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)
print(f"✅ Model loaded on {DEVICE}")

print("📄 Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("📦 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print(f"✅ Ready | {len(metadata)} chunks indexed")

# ---------------- SEARCH FUNCTION ---------------- #

def semantic_search(query: str, top_k: int = TOP_K):
    """
    Semantic search over placement chunks.
    Returns agent-friendly structured results.
    """

    # Embed query (BGE expects 'query:' prefix)
    query_vec = model.encode(
        "query: " + query,
        normalize_embeddings=True
    ).reshape(1, -1)

    scores, indices = index.search(query_vec, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue  # skip invalid index

        item = metadata[idx]
        results.append({
            "score": round(float(score), 4),
            "company": item["company"],
            "role": item.get("role", ""),
            "category": item.get("category", ""),
            "chunk_type": item["chunk_type"],
            "text": item["text"]
        })


    return results

# ---------------- INTERACTIVE MODE ---------------- #

if __name__ == "__main__":
    print("\n💬 Semantic Search (Ctrl+C to exit)")
    while True:
        try:
            query = input("\nEnter query: ").strip()
            if not query:
                continue

            results = semantic_search(query)

            print("\n🎯 Top Matches:\n")
            for i, r in enumerate(results, 1):
                print(
                    f"{i}. "
                    f"[{r['company']} | {r['role_name']} | {r['chunk_type']}]"
                )
                print(f"   Score: {r['score']}")
                print(f"   {r['text']}\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting semantic search")
            break
