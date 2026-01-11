#!/usr/bin/env python3
"""Interactive testing with detailed debugging."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import create_agent


def main():
    print("\n🧪 INTERACTIVE DEBUG MODE")
    print("=" * 50)
    print("This shows detailed execution for each query.\n")
    
    agent = create_agent(use_llm=True)
    
    while True:
        query = input("\n🎓 Query (or 'quit'): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Run with verbose
        response = agent.query(query, verbose=True)
        
        print("\n" + "─" * 50)
        print("📝 FINAL RESPONSE:")
        print("─" * 50)
        print(response.answer)
        print("─" * 50)
        print(f"Confidence: {response.feedback.confidence_score:.0%}")
        print(f"Retries: {response.retries}")
        print(f"Intent: {response.plan.intent}")
        print(f"Companies: {response.plan.companies_mentioned}")


if __name__ == "__main__":
    main()
