#!/usr/bin/env python3
"""Analyze raw extraction results to verify data quality."""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor.config import RAW_EXTRACTED_OUTPUT, PLACEMENTS_DIR


def load_raw_data():
    """Load the raw extracted data."""
    if not RAW_EXTRACTED_OUTPUT.exists():
        print(f"ERROR: Raw extraction file not found: {RAW_EXTRACTED_OUTPUT}")
        return None
    
    with open(RAW_EXTRACTED_OUTPUT, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_extraction(data):
    """Analyze extraction completeness and quality."""
    
    print("=" * 70)
    print("RAW EXTRACTION ANALYSIS REPORT")
    print("=" * 70)
    
    # Overall stats
    total_entries = len(data)
    total_files = sum(len(entry['files']) for entry in data)
    total_chars = sum(entry['total_chars'] for entry in data)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total placement entries: {total_entries}")
    print(f"   Total files processed: {total_files}")
    print(f"   Total characters extracted: {total_chars:,}")
    
    # File type breakdown
    file_types = defaultdict(lambda: {'count': 0, 'extracted': 0, 'empty': 0, 'chars': 0})
    
    for entry in data:
        for f in entry['files']:
            ext = f['file_type']
            file_types[ext]['count'] += 1
            file_types[ext]['chars'] += f['char_count']
            if f['char_count'] > 0:
                file_types[ext]['extracted'] += 1
            else:
                file_types[ext]['empty'] += 1
    
    print(f"\n📁 FILE TYPE BREAKDOWN:")
    print(f"   {'Type':<10} {'Total':<8} {'Extracted':<12} {'Empty':<8} {'Chars':<15}")
    print(f"   {'-'*10} {'-'*8} {'-'*12} {'-'*8} {'-'*15}")
    
    for ext, stats in sorted(file_types.items()):
        status = "✅" if stats['empty'] == 0 else "⚠️"
        print(f"   {ext:<10} {stats['count']:<8} {stats['extracted']:<12} {stats['empty']:<8} {stats['chars']:<15,} {status}")
    
    # Entries with issues
    print(f"\n⚠️  ENTRIES WITH POTENTIAL ISSUES:")
    
    empty_entries = []
    low_content_entries = []
    image_only_entries = []
    
    for entry in data:
        if entry['total_chars'] == 0:
            empty_entries.append(entry)
        elif entry['total_chars'] < 200:
            low_content_entries.append(entry)
        
        # Check if only images with no extraction
        non_image_content = sum(
            f['char_count'] for f in entry['files'] 
            if f['file_type'] not in ['.png', '.jpg', '.jpeg']
        )
        image_content = sum(
            f['char_count'] for f in entry['files'] 
            if f['file_type'] in ['.png', '.jpg', '.jpeg']
        )
        
        if non_image_content == 0 and image_content == 0:
            has_images = any(f['file_type'] in ['.png', '.jpg', '.jpeg'] for f in entry['files'])
            if has_images:
                image_only_entries.append(entry)
    
    if empty_entries:
        print(f"\n   🔴 EMPTY ENTRIES ({len(empty_entries)}):")
        for e in empty_entries:
            files = [f['file_name'] for f in e['files']]
            print(f"      - {e['primary_key']}: {files}")
    else:
        print(f"\n   ✅ No empty entries")
    
    if low_content_entries:
        print(f"\n   🟡 LOW CONTENT ENTRIES (<200 chars) ({len(low_content_entries)}):")
        for e in low_content_entries:
            print(f"      - {e['primary_key']}: {e['total_chars']} chars")
    else:
        print(f"\n   ✅ No low content entries")
    
    if image_only_entries:
        print(f"\n   🟠 IMAGE-ONLY ENTRIES WITH NO OCR ({len(image_only_entries)}):")
        for e in image_only_entries:
            images = [f['file_name'] for f in e['files'] if f['file_type'] in ['.png', '.jpg', '.jpeg']]
            print(f"      - {e['primary_key']}: {images}")
    
    # Sample content preview
    print(f"\n📝 SAMPLE CONTENT PREVIEW (first 3 entries with content):")
    print("-" * 70)
    
    shown = 0
    for entry in data:
        if entry['total_chars'] > 100 and shown < 3:
            print(f"\n   📌 {entry['primary_key']} ({entry['company_name']} - {entry['role_name']})")
            print(f"   Files: {len(entry['files'])}, Total chars: {entry['total_chars']:,}")
            
            # Show first file content preview
            for f in entry['files']:
                if f['char_count'] > 50:
                    preview = f['content'][:300].replace('\n', ' ')
                    print(f"   [{f['file_name']}]: {preview}...")
                    break
            shown += 1
    
    # Per-entry detailed breakdown
    print(f"\n📋 DETAILED ENTRY BREAKDOWN:")
    print("-" * 70)
    print(f"{'Entry':<40} {'Files':<6} {'Chars':<10} {'Status'}")
    print("-" * 70)
    
    for entry in data:
        status = "✅" if entry['total_chars'] > 200 else ("⚠️" if entry['total_chars'] > 0 else "❌")
        name = entry['primary_key'][:38]
        print(f"{name:<40} {len(entry['files']):<6} {entry['total_chars']:<10,} {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    good_entries = sum(1 for e in data if e['total_chars'] > 200)
    ok_entries = sum(1 for e in data if 0 < e['total_chars'] <= 200)
    bad_entries = sum(1 for e in data if e['total_chars'] == 0)
    
    print(f"   ✅ Good entries (>200 chars): {good_entries}")
    print(f"   ⚠️  OK entries (1-200 chars): {ok_entries}")
    print(f"   ❌ Empty entries: {bad_entries}")
    
    # Check if images were processed
    total_images = file_types.get('.png', {}).get('count', 0) + \
                   file_types.get('.jpg', {}).get('count', 0) + \
                   file_types.get('.jpeg', {}).get('count', 0)
    extracted_images = file_types.get('.png', {}).get('extracted', 0) + \
                       file_types.get('.jpg', {}).get('extracted', 0) + \
                       file_types.get('.jpeg', {}).get('extracted', 0)
    
    print(f"\n   📷 Image OCR: {extracted_images}/{total_images} images extracted")
    
    if extracted_images < total_images:
        print(f"   ⚠️  {total_images - extracted_images} images failed OCR - consider re-running with OCR enabled")
    
    return {
        'total_entries': total_entries,
        'good_entries': good_entries,
        'empty_entries': bad_entries,
        'image_ocr_success': extracted_images,
        'image_ocr_total': total_images
    }


def main():
    print("Loading raw extraction data...")
    data = load_raw_data()
    
    if data is None:
        return
    
    stats = analyze_extraction(data)
    
    # Decision point
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if stats['image_ocr_success'] < stats['image_ocr_total']:
        print("""
⚠️  Some images were not OCR'd. You have two options:

1. RE-RUN Phase 1 with OCR enabled:
   - Make sure easyocr is installed: pip install easyocr
   - Run: python run_phase1_only.py
   
2. PROCEED to Phase 2 anyway (if document files have enough content):
   - Run: python run_extractor.py --skip-phase1
""")
    else:
        print("""
✅ Extraction looks good! Ready for Phase 2.

To proceed with LLM-based extraction:
   python run_extractor.py --skip-phase1

This will use the LLM to extract:
   - Structured facts (company, role, eligibility, etc.)
   - Semantic chunks (about company, skills, etc.)
""")


if __name__ == "__main__":
    main()
