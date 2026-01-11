#!/usr/bin/env python3
"""Test the placement agent with various queries."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import create_agent


def test_agent():
    """Test the agent with different query types."""
    
    print("\n" + "=" * 70)
    print("PLACEMENT AGENT TEST")
    print("=" * 70)
    
    # Create agent
    print("\nInitializing agent...")
    agent = create_agent()
    print(f"Agent ready! Known companies: {len(agent.planner.KNOWN_COMPANIES)}")
    
    # Test queries
    test_queries = [
        # Facts queries
        "What is the stipend offered by Dell?",
        "List all companies with stipend more than 40000",
        "Which companies are hiring in Bangalore?",
        
        # Semantic queries
        "Tell me about the interview process at Amazon",
        "What skills are required for machine learning roles?",
        "Describe the work culture at Google",
        
        # Comparison queries
        "Compare Dell and Amazon",
        "Which company offers the highest stipend?",
        
        # Hybrid/ambiguous queries
        "Tell me about Dell internship",
        "Best companies for data science",
        "What are my options for ML internship with good stipend?"
    ]
    
    for query in test_queries:
        print("\n" + "-" * 70)
        response = agent.query(query, verbose=True)
        
        print("\n📝 ANSWER:")
        print(response.answer)
        
        print(f"\n📊 Stats: Confidence={response.feedback.confidence_score:.0%}, Retries={response.retries}")
        
        input("\nPress Enter for next query...")


def interactive_mode():
    """Interactive query mode."""
    
    print("\n" + "=" * 70)
    print("PLACEMENT AGENT - INTERACTIVE MODE")
    print("=" * 70)
    
    agent = create_agent()
    print(f"\nAgent ready! Ask any question about placements.")
    print("Type 'quit' to exit, 'verbose' to toggle detailed output.\n")
    
    verbose = False
    
    while True:
        try:
            query = input("\n🎓 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            
            if query.lower() == 'verbose':
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            
            response = agent.query(query, verbose=verbose)
            
            print("\n🤖 Agent:")
            print(response.answer)
            
            if verbose:
                print(f"\n[Confidence: {response.feedback.confidence_score:.0%}]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test placement agent")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        test_agent()
