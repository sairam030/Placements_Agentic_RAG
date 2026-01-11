#!/usr/bin/env python3
"""Build FAISS and Facts indices from extracted data."""

import json
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import FACTS_FILE, SEMANTIC_FILE, RAG_DIR
from rag.semantic_index import SemanticIndex
from rag.facts_index import FactsIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_semantic_index():
    """Build FAISS index from semantic chunks."""
    logger.info("=" * 60)
    logger.info("BUILDING SEMANTIC INDEX (FAISS)")
    logger.info("=" * 60)
    
    # Load chunks
    if not SEMANTIC_FILE.exists():
        logger.error(f"Semantic file not found: {SEMANTIC_FILE}")
        return None
    
    with open(SEMANTIC_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} semantic chunks")
    
    # Build index
    index = SemanticIndex()
    index.build_index(chunks, save=True)
    
    # Print stats
    stats = index.get_stats()
    logger.info(f"\nSemantic Index Stats:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Total companies: {stats['total_companies']}")
    logger.info(f"  Chunks by type: {stats['chunks_by_type']}")
    
    return index


def build_facts_index():
    """Build facts index."""
    logger.info("\n" + "=" * 60)
    logger.info("BUILDING FACTS INDEX")
    logger.info("=" * 60)
    
    # Load and build
    index = FactsIndex()
    if not index.load_facts():
        logger.error("Failed to load facts")
        return None
    
    # Save index
    index.save()
    
    # Print stats
    stats = index.get_stats()
    logger.info(f"\nFacts Index Stats:")
    logger.info(f"  Total entries: {stats['total_entries']}")
    logger.info(f"  Total companies: {stats['total_companies']}")
    logger.info(f"  Avg stipend: {stats.get('avg_stipend', 'N/A')}")
    
    return index


def main():
    """Build all indices."""
    print("\n" + "=" * 70)
    print("BUILDING RAG INDICES")
    print("=" * 70 + "\n")
    
    # Build semantic index
    semantic_idx = build_semantic_index()
    
    # Build facts index
    facts_idx = build_facts_index()
    
    print("\n" + "=" * 70)
    print("INDEX BUILDING COMPLETE")
    print("=" * 70)
    print(f"\nIndex files saved to: {RAG_DIR}")
    print(f"  - semantic.faiss")
    print(f"  - semantic_metadata.json")
    print(f"  - facts_index.pkl")
    
    return semantic_idx, facts_idx


if __name__ == "__main__":
    main()
