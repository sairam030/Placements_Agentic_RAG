"""Main extraction pipeline for placement data - Two Phase Approach."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from extractor.config import (
    PLACEMENTS_DIR, OUTPUT_DIR, FACTS_OUTPUT, SEMANTIC_OUTPUT,
    USE_VLLM, LLM_MODEL, RAW_EXTRACTED_OUTPUT
)
from extractor.raw_extractor import RawDataExtractor
from extractor.llm_processor import LLMProcessor

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlacementDataExtractor:
    """Main class for extracting placement data using two-phase approach."""
    
    def __init__(self, placements_dir: Path = PLACEMENTS_DIR):
        self.placements_dir = placements_dir
        self.llm_processor = None
        self.facts_data: List[Dict[str, Any]] = []
        self.semantic_data: List[Dict[str, Any]] = []
        self.failed_entries: List[str] = []
    
    def initialize_llm(self):
        """Initialize the LLM processor."""
        logger.info("=" * 60)
        logger.info("INITIALIZING LLM")
        logger.info(f"Model: {LLM_MODEL}")
        logger.info("=" * 60)
        
        start_time = time.time()
        self.llm_processor = LLMProcessor(use_vllm=USE_VLLM, model_name=LLM_MODEL)
        elapsed = time.time() - start_time
        
        logger.info(f"LLM initialized in {elapsed:.1f} seconds")
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw extraction data."""
        if not RAW_EXTRACTED_OUTPUT.exists():
            logger.error(f"Raw data not found: {RAW_EXTRACTED_OUTPUT}")
            logger.error("Run Phase 1 first: python run_phase1_only.py")
            return []
        
        with open(RAW_EXTRACTED_OUTPUT, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} entries from raw extraction")
        return data
    
    def process_entry(self, entry: Dict[str, Any]) -> tuple:
        """Process a single entry to extract facts and chunks."""
        primary_key = entry['primary_key']
        
        # Skip entries with no content
        if entry['total_chars'] < 100:
            logger.warning(f"Skipping {primary_key}: insufficient content ({entry['total_chars']} chars)")
            return None, []
        
        try:
            # Extract facts
            facts = self.llm_processor.extract_facts(entry)
            
            # Extract semantic chunks
            chunks = self.llm_processor.extract_semantic_chunks(entry)
            
            return facts, chunks
            
        except Exception as e:
            logger.error(f"Error processing {primary_key}: {e}")
            self.failed_entries.append(primary_key)
            return None, []
    
    def run(self, skip_phase1: bool = True, resume_from: int = 0):
        """Run the extraction pipeline."""
        
        print("\n" + "=" * 70)
        print("PLACEMENT DATA EXTRACTION - PHASE 2: LLM PROCESSING")
        print("=" * 70 + "\n")
        
        # Load raw data
        raw_data = self.load_raw_data()
        if not raw_data:
            return
        
        # Filter entries with content
        valid_entries = [e for e in raw_data if e['total_chars'] >= 100]
        logger.info(f"Processing {len(valid_entries)} valid entries (skipping {len(raw_data) - len(valid_entries)} empty)")
        
        # Resume support
        if resume_from > 0:
            valid_entries = valid_entries[resume_from:]
            logger.info(f"Resuming from entry {resume_from}")
            
            # Load existing results
            if FACTS_OUTPUT.exists():
                with open(FACTS_OUTPUT, 'r') as f:
                    self.facts_data = json.load(f)
            if SEMANTIC_OUTPUT.exists():
                with open(SEMANTIC_OUTPUT, 'r') as f:
                    self.semantic_data = json.load(f)
        
        # Initialize LLM
        self.initialize_llm()
        
        # Process each entry
        print("\n" + "-" * 70)
        print("EXTRACTING FACTS AND SEMANTIC CHUNKS")
        print("-" * 70 + "\n")
        
        for idx, entry in enumerate(tqdm(valid_entries, desc="Processing")):
            try:
                facts, chunks = self.process_entry(entry)
                
                if facts:
                    self.facts_data.append(facts)
                
                if chunks:
                    self.semantic_data.extend(chunks)
                
                # Save intermediate results every 5 entries
                if (idx + 1) % 5 == 0:
                    self._save_intermediate()
                    logger.info(f"Progress: {idx + 1}/{len(valid_entries)} | Facts: {len(self.facts_data)} | Chunks: {len(self.semantic_data)}")
                    
            except KeyboardInterrupt:
                logger.warning(f"\nInterrupted at entry {idx + resume_from}")
                self._save_intermediate()
                print(f"\nResume with: python run_extractor.py --resume {idx + resume_from}")
                return
            except Exception as e:
                logger.error(f"Error on entry {idx}: {e}")
                continue
        
        # Save final results
        self.save_results()
        self._print_summary()
    
    def _save_intermediate(self):
        """Save intermediate results."""
        with open(FACTS_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(self.facts_data, f, indent=2, ensure_ascii=False)
        
        with open(SEMANTIC_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_data, f, indent=2, ensure_ascii=False)
    
    def save_results(self):
        """Save final results to JSON files."""
        logger.info(f"\nSaving {len(self.facts_data)} facts to {FACTS_OUTPUT}")
        with open(FACTS_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(self.facts_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saving {len(self.semantic_data)} chunks to {SEMANTIC_OUTPUT}")
        with open(SEMANTIC_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_data, f, indent=2, ensure_ascii=False)
        
        # Save summary
        self._save_summary()
    
    def _save_summary(self):
        """Save extraction summary."""
        chunks_by_type = {}
        for chunk in self.semantic_data:
            t = chunk["type"]
            chunks_by_type[t] = chunks_by_type.get(t, 0) + 1
        
        summary = {
            "model_used": LLM_MODEL,
            "total_facts": len(self.facts_data),
            "total_chunks": len(self.semantic_data),
            "chunks_by_type": chunks_by_type,
            "failed_entries": self.failed_entries,
            "companies": list(set(f.get("company_name", "") for f in self.facts_data))
        }
        
        summary_path = OUTPUT_DIR / "extraction_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _print_summary(self):
        """Print extraction summary."""
        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"\n✅ Facts extracted: {len(self.facts_data)}")
        print(f"✅ Semantic chunks: {len(self.semantic_data)}")
        
        if self.failed_entries:
            print(f"⚠️  Failed entries: {len(self.failed_entries)}")
        
        # Chunk breakdown
        chunks_by_type = {}
        for chunk in self.semantic_data:
            t = chunk["type"]
            chunks_by_type[t] = chunks_by_type.get(t, 0) + 1
        
        print("\n📊 Chunks by type:")
        for t, count in sorted(chunks_by_type.items()):
            print(f"   {t}: {count}")
        
        print(f"\n📁 Output files:")
        print(f"   Facts: {FACTS_OUTPUT}")
        print(f"   Chunks: {SEMANTIC_OUTPUT}")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract placement data using LLM")
    parser.add_argument("--skip-phase1", action="store_true", default=True,
                        help="Skip phase 1 (use existing raw data)")
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from entry number")
    args = parser.parse_args()
    
    extractor = PlacementDataExtractor()
    extractor.run(skip_phase1=args.skip_phase1, resume_from=args.resume)


if __name__ == "__main__":
    main()
