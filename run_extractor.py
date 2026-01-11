#!/usr/bin/env python3
"""Runner script for Phase 2: LLM-based extraction."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use second A100

from extractor.main_extractor import main

if __name__ == "__main__":
    main()
