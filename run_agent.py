#!/usr/bin/env python3
"""Run the LLM-powered placement agent."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use second GPU

from agent.orchestrator import create_agent


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            🎓 PLACEMENT RAG AGENT (LLM-Powered)                  ║
║                                                                  ║
║   This agent uses LLM for:                                       ║
║   • Query understanding & planning                               ║
║   • Result validation & critique                                 ║
║   • Natural response generation                                  ║
║                                                                  ║
║   Commands: 'quit' | 'verbose' | 'companies' | 'nollm'           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("🔄 Initializing LLM-powered agent (this may take a moment)...")
    
    use_llm = True
    agent = create_agent(use_llm=use_llm)
    
    companies = agent.get_companies()
    print(f"✅ Ready! {len(companies)} companies loaded.")
    print(f"📋 Sample: {', '.join(companies[:5])}...\n")
    
    verbose = False
    
    while True:
        try:
            query = input("🎓 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Good luck with your placements! 🎉")
                break
            
            if query.lower() == 'verbose':
                verbose = not verbose
                print(f"📢 Verbose mode: {'ON' if verbose else 'OFF'}\n")
                continue
            
            if query.lower() == 'companies':
                print(f"\n📋 Available companies ({len(companies)}):")
                for i, c in enumerate(companies, 1):
                    print(f"   {i}. {c}")
                print()
                continue
            
            if query.lower() == 'nollm':
                use_llm = not use_llm
                agent = create_agent(use_llm=use_llm)
                print(f"🤖 LLM mode: {'ON' if use_llm else 'OFF (rule-based)'}\n")
                continue
            
            # Process query
            response = agent.query(query, verbose=verbose)
            
            print("\n" + "─" * 60)
            print(response.answer)
            print("─" * 60)
            
            conf = response.feedback.confidence_score
            conf_icon = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
            print(f"{conf_icon} Confidence: {conf:.0%}")
            
            if response.retries > 0:
                print(f"🔄 Retries: {response.retries}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Good luck with your placements! 🎉")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
