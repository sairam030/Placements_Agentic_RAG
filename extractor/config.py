"""Configuration settings for the placement data extractor."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path("/data/shreyas25/sairam/placements_data")
PLACEMENTS_DIR = BASE_DIR / "Placements"
OUTPUT_DIR = BASE_DIR / "v_rag/output"
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw"

# =============================================================================
# LLM CONFIGURATION - Using high-quality models for best extraction
# =============================================================================

# Option 1: Qwen2.5-72B-Instruct (Excellent for structured extraction)
# LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Option 2: Llama-3.1-70B-Instruct (Great general purpose)
# LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

# Option 3: Mixtral-8x22B (Good balance of speed and quality)
# LLM_MODEL = "mistralai/Mixtral-8x22B-Instruct-v0.1"

# Option 4: Qwen2.5-32B-Instruct (Good quality, fits on single A100)
LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# Option 5: Smaller but fast - Qwen2.5-14B
# LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# LLM Backend settings
USE_VLLM = True
VLLM_GPU_MEMORY_UTILIZATION = 0.85  # Use more GPU for larger model
VLLM_TENSOR_PARALLEL_SIZE = 1  # Use 1 GPU, set to 2 for 70B+ models
DEVICE = "cuda:1"  # Use second A100 GPU

# Generation settings
LLM_TEMPERATURE = 0.1  # Low temperature for consistent extraction
LLM_MAX_TOKENS = 4096
LLM_TOP_P = 0.95

# =============================================================================
# OCR CONFIGURATION
# =============================================================================
OCR_BACKEND = "easyocr"
OCR_LANGUAGES = ["en"]
OCR_USE_GPU = True

# =============================================================================
# PROCESSING SETTINGS
# =============================================================================
MAX_CONTENT_LENGTH = 12000  # Max chars per LLM call
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 1  # Process one at a time for quality

# Semantic chunk types
CHUNK_TYPES = [
    "about_company",
    "roles_responsibilities", 
    "skills_required",
    "skills_optional",
    "interview_process",
    "eligibility_criteria",
    "compensation_benefits",
    "additional_info"
]

# Output files
RAW_EXTRACTED_OUTPUT = RAW_OUTPUT_DIR / "raw_extracted.json"
FACTS_OUTPUT = OUTPUT_DIR / "facts.json"
SEMANTIC_OUTPUT = OUTPUT_DIR / "semantic.json"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
