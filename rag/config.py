"""Configuration for RAG system."""

from pathlib import Path

# Base paths
BASE_DIR = Path("/data/shreyas25/sairam/placements_data/v_rag")
OUTPUT_DIR = BASE_DIR / "output"
RAG_DIR = BASE_DIR / "rag_index"

# Data files
FACTS_FILE = OUTPUT_DIR / "facts.json"
SEMANTIC_FILE = OUTPUT_DIR / "semantic.json"

# Index files
FAISS_INDEX_FILE = RAG_DIR / "semantic.faiss"
FAISS_METADATA_FILE = RAG_DIR / "semantic_metadata.json"
FACTS_INDEX_FILE = RAG_DIR / "facts_index.pkl"

# Embedding model - good balance of speed and quality
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative: Better quality, slower
# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Embedding settings
EMBEDDING_DIMENSION = 384  # For MiniLM, use 768 for BGE
EMBEDDING_BATCH_SIZE = 32

# Search settings
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# Ensure directories exist
RAG_DIR.mkdir(parents=True, exist_ok=True)
