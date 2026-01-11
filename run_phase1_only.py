#!/usr/bin/env python3
"""Run only Phase 1: Raw text extraction."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Running Phase 1: Raw Text Extraction")
    print("=" * 50)
    
    # Check OCR availability
    from extractor.config import OCR_BACKEND, OCR_USE_GPU
    print(f"OCR Backend: {OCR_BACKEND}")
    print(f"OCR GPU: {OCR_USE_GPU}")
    
    # Pre-initialize OCR to see any errors upfront
    print("\nInitializing OCR...")
    from extractor.file_readers import get_ocr_reader
    reader = get_ocr_reader()
    
    if reader is None:
        print("WARNING: OCR initialization failed. Images will not be processed.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    else:
        print("OCR initialized successfully!")
    
    print("\n" + "=" * 50)
    print("Starting extraction...")
    print("=" * 50 + "\n")
    
    from extractor.raw_extractor import run_raw_extraction
    results = run_raw_extraction()
    
    print("\n" + "=" * 50)
    print(f"Extracted {len(results)} entries")
    print("Raw data saved. Run Phase 2 with GPU for LLM processing.")
    print("=" * 50)


if __name__ == "__main__":
    main()
