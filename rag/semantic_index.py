"""FAISS-based semantic index for chunk retrieval."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticIndex:
    """FAISS-based semantic search index for placement chunks."""
    
    def __init__(self, embedding_model: str = None):
        from rag.config import (
            EMBEDDING_MODEL, EMBEDDING_DIMENSION,
            FAISS_INDEX_FILE, FAISS_METADATA_FILE
        )
        
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIMENSION
        self.index_file = FAISS_INDEX_FILE
        self.metadata_file = FAISS_METADATA_FILE
        
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedder = None
    
    def _load_embedder(self):
        """Load the sentence transformer model."""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)
            # Update dimension based on actual model
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts."""
        self._load_embedder()
        
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        return embeddings.astype('float32')
    
    def build_index(self, chunks: List[Dict[str, Any]], save: bool = True):
        """Build FAISS index from semantic chunks."""
        
        if not chunks:
            logger.error("No chunks provided")
            return
        
        logger.info(f"Building index from {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = []
        for chunk in chunks:
            # Create rich text for embedding (include context)
            text = f"{chunk.get('company', '')} - {chunk.get('role', '')}: {chunk.get('type', '')}\n{chunk.get('text', '')}"
            texts.append(text)
        
        # Generate embeddings
        embeddings = self._embed_texts(texts)
        
        # Create FAISS index
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add vectors
        self.index.add(embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
        # Store metadata
        self.metadata = []
        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "primary_key": chunk.get("primary_key", ""),
                "company": chunk.get("company", ""),
                "role": chunk.get("role", ""),
                "type": chunk.get("type", ""),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
            })
        
        if save:
            self.save()
    
    def save(self):
        """Save index and metadata to disk."""
        logger.info(f"Saving index to {self.index_file}")
        faiss.write_index(self.index, str(self.index_file))
        
        logger.info(f"Saving metadata to {self.metadata_file}")
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load(self) -> bool:
        """Load index and metadata from disk."""
        if not self.index_file.exists() or not self.metadata_file.exists():
            logger.warning("Index files not found")
            return False
        
        logger.info(f"Loading index from {self.index_file}")
        self.index = faiss.read_index(str(self.index_file))
        
        logger.info(f"Loading metadata from {self.metadata_file}")
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        return True
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_company: Optional[str] = None,
        filter_type: Optional[str] = None,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_company: Filter by company name
            filter_type: Filter by chunk type
            threshold: Minimum similarity score
        
        Returns:
            List of matching chunks with scores
        """
        if self.index is None:
            if not self.load():
                logger.error("No index available")
                return []
        
        # Embed query
        self._load_embedder()
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search (get more results if filtering)
        search_k = top_k * 5 if (filter_company or filter_type) else top_k
        scores, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))
        
        # Collect results with filtering
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            
            meta = self.metadata[idx]
            
            # Apply filters
            if filter_company and filter_company.lower() not in meta['company'].lower():
                continue
            if filter_type and filter_type != meta['type']:
                continue
            
            results.append({
                **meta,
                "score": float(score)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_type(
        self,
        query: str,
        chunk_type: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific chunk type."""
        return self.search(query, top_k=top_k, filter_type=chunk_type)
    
    def search_by_company(
        self,
        query: str,
        company: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific company."""
        return self.search(query, top_k=top_k, filter_company=company)
    
    def get_all_by_company(self, company: str) -> List[Dict[str, Any]]:
        """Get all chunks for a company."""
        return [m for m in self.metadata if company.lower() in m['company'].lower()]
    
    def get_all_by_type(self, chunk_type: str) -> List[Dict[str, Any]]:
        """Get all chunks of a specific type."""
        return [m for m in self.metadata if m['type'] == chunk_type]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.metadata:
            return {}
        
        companies = set(m['company'] for m in self.metadata)
        types = {}
        for m in self.metadata:
            t = m['type']
            types[t] = types.get(t, 0) + 1
        
        return {
            "total_chunks": len(self.metadata),
            "total_companies": len(companies),
            "companies": list(companies),
            "chunks_by_type": types,
            "index_size": self.index.ntotal if self.index else 0
        }
