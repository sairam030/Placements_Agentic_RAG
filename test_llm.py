#!/usr/bin/env python3
"""Test LLM loading and generation before full extraction."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    print("=" * 60)
    print("LLM TEST SCRIPT")
    print("=" * 60)
    
    from extractor.config import LLM_MODEL, USE_VLLM
    
    print(f"\nModel: {LLM_MODEL}")
    print(f"Backend: {'vLLM' if USE_VLLM else 'Transformers'}")
    
    # Check GPU
    import torch
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Try loading model
    print("\n" + "-" * 60)
    print("Loading LLM (this may take a few minutes)...")
    print("-" * 60)
    
    try:
        from extractor.llm_processor import LLMProcessor
        processor = LLMProcessor()
        
        # Test generation
        print("\n✅ Model loaded successfully!")
        print("\nTesting generation...")
        
        test_prompt = """Extract company name and job title from this text:
        
"Amazon is hiring Software Development Engineer Interns for 2025 batch."

Return JSON: {"company": "...", "job_title": "..."}"""
        
        response = processor.generate(test_prompt)
        print(f"\nResponse:\n{response}")
        
        print("\n" + "=" * 60)
        print("✅ LLM TEST PASSED - Ready for extraction!")
        print("=" * 60)
        print("\nRun full extraction with:")
        print("  python run_extractor.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry a smaller model by editing config.py:")
        print('  LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"')


if __name__ == "__main__":
    main()
