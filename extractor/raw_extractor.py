"""Phase 1: Extract raw text from all placement files."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

from extractor.config import PLACEMENTS_DIR, RAW_EXTRACTED_OUTPUT, RAW_OUTPUT_DIR
from extractor.directory_scanner import scan_placements_directory, PlacementEntry
from extractor.file_readers import read_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExtractedFile:
    """Represents extracted content from a single file."""
    file_name: str
    file_path: str
    file_type: str
    content: str
    char_count: int


@dataclass
class RawExtraction:
    """Raw extraction for a single placement entry."""
    primary_key: str
    company_name: str
    role_name: str  # None becomes "general"
    batch_year: str
    folder_path: str
    files: List[ExtractedFile]
    combined_text: str
    total_chars: int


class RawDataExtractor:
    """Extract raw text from all placement files without LLM processing."""
    
    def __init__(self, placements_dir: Path = PLACEMENTS_DIR):
        self.placements_dir = placements_dir
        self.extractions: List[RawExtraction] = []
    
    def extract_entry(self, entry: PlacementEntry) -> RawExtraction:
        """Extract raw text from all files in an entry."""
        extracted_files = []
        all_text_parts = []
        
        for file_path in entry.files:
            try:
                content = read_file(file_path)
                if content.strip():
                    ext_file = ExtractedFile(
                        file_name=file_path.name,
                        file_path=str(file_path),
                        file_type=file_path.suffix.lower(),
                        content=content,
                        char_count=len(content)
                    )
                    extracted_files.append(ext_file)
                    all_text_parts.append(f"[SOURCE: {file_path.name}]\n{content}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        combined_text = "\n\n---\n\n".join(all_text_parts)
        
        return RawExtraction(
            primary_key=entry.primary_key,
            company_name=entry.company_name,
            role_name=entry.role_name or "general",
            batch_year=entry.batch_year or "unknown",
            folder_path=str(entry.folder_path),
            files=extracted_files,
            combined_text=combined_text,
            total_chars=len(combined_text)
        )
    
    def run(self) -> List[Dict[str, Any]]:
        """Run raw extraction for all entries."""
        logger.info(f"Scanning: {self.placements_dir}")
        entries = scan_placements_directory(self.placements_dir)
        logger.info(f"Found {len(entries)} placement entries")
        
        for entry in tqdm(entries, desc="Extracting raw text"):
            try:
                extraction = self.extract_entry(entry)
                self.extractions.append(extraction)
            except Exception as e:
                logger.error(f"Error processing {entry.primary_key}: {e}")
        
        # Convert to dict for JSON serialization
        results = []
        for ext in self.extractions:
            ext_dict = {
                "primary_key": ext.primary_key,
                "company_name": ext.company_name,
                "role_name": ext.role_name,
                "batch_year": ext.batch_year,
                "folder_path": ext.folder_path,
                "total_chars": ext.total_chars,
                "combined_text": ext.combined_text,
                "files": [
                    {
                        "file_name": f.file_name,
                        "file_path": f.file_path,
                        "file_type": f.file_type,
                        "content": f.content,
                        "char_count": f.char_count
                    }
                    for f in ext.files
                ]
            }
            results.append(ext_dict)
        
        # Save raw extractions
        self.save_results(results)
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save raw extraction results."""
        RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(RAW_EXTRACTED_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} raw extractions to {RAW_EXTRACTED_OUTPUT}")
        
        # Also save a summary
        summary = {
            "total_entries": len(results),
            "total_files": sum(len(r["files"]) for r in results),
            "total_chars": sum(r["total_chars"] for r in results),
            "entries": [
                {
                    "primary_key": r["primary_key"],
                    "company": r["company_name"],
                    "role": r["role_name"],
                    "files_count": len(r["files"]),
                    "chars": r["total_chars"]
                }
                for r in results
            ]
        }
        
        summary_path = RAW_OUTPUT_DIR / "extraction_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_path}")


def run_raw_extraction():
    """Convenience function to run raw extraction."""
    extractor = RawDataExtractor()
    return extractor.run()


if __name__ == "__main__":
    run_raw_extraction()
