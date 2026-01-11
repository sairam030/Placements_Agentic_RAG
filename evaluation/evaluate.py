#!/usr/bin/env python3
"""Evaluate the placement agent on test queries."""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import create_agent
from evaluation.test_queries import TEST_QUERIES


def evaluate_response(query_info: Dict, response: str, confidence: float) -> Dict[str, Any]:
    """Evaluate a single response."""
    
    response_lower = response.lower()
    expected_terms = query_info["expected"]
    
    # Check if expected terms are in response
    found_terms = []
    missing_terms = []
    
    for term in expected_terms:
        if term.lower() in response_lower:
            found_terms.append(term)
        else:
            missing_terms.append(term)
    
    # Calculate scores
    term_coverage = len(found_terms) / len(expected_terms) if expected_terms else 1.0
    
    # Check for hallucination indicators
    hallucination_indicators = [
        "i don't have",
        "i cannot",
        "not available",
        "no information",
        "$",  # Dollar sign (should be INR)
    ]
    
    has_hallucination_risk = any(ind in response_lower for ind in hallucination_indicators[:4])
    has_wrong_currency = "$" in response and "INR" not in response
    
    # Overall quality score
    quality_score = (
        term_coverage * 0.4 +
        confidence * 0.3 +
        (0.2 if not has_hallucination_risk else 0.0) +
        (0.1 if not has_wrong_currency else 0.0)
    )
    
    return {
        "term_coverage": term_coverage,
        "found_terms": found_terms,
        "missing_terms": missing_terms,
        "confidence": confidence,
        "has_hallucination_risk": has_hallucination_risk,
        "has_wrong_currency": has_wrong_currency,
        "quality_score": quality_score,
        "passed": quality_score >= 0.6
    }


def run_evaluation(verbose: bool = True):
    """Run full evaluation."""
    
    print("\n" + "="*70)
    print("PLACEMENT AGENT EVALUATION")
    print("="*70)
    
    # Initialize agent
    print("\n🔄 Initializing agent...")
    agent = create_agent(use_llm=True)
    print("✅ Agent ready!\n")
    
    results = []
    total_time = 0
    
    for i, query_info in enumerate(TEST_QUERIES, 1):
        query = query_info["query"]
        
        print(f"\n{'─'*60}")
        print(f"Test {i}/{len(TEST_QUERIES)}: {query_info['category']}")
        print(f"Query: {query}")
        
        # Run query
        start_time = time.time()
        try:
            response = agent.query(query, verbose=False)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Evaluate
            evaluation = evaluate_response(
                query_info,
                response.answer,
                response.feedback.confidence_score
            )
            
            results.append({
                "query": query,
                "category": query_info["category"],
                "type": query_info["type"],
                "response_length": len(response.answer),
                "elapsed_time": elapsed,
                "retries": response.retries,
                **evaluation
            })
            
            # Print results
            status = "✅ PASS" if evaluation["passed"] else "❌ FAIL"
            print(f"Status: {status}")
            print(f"Quality Score: {evaluation['quality_score']:.2f}")
            print(f"Term Coverage: {evaluation['term_coverage']:.0%}")
            print(f"Confidence: {evaluation['confidence']:.0%}")
            print(f"Time: {elapsed:.2f}s")
            
            if evaluation["missing_terms"]:
                print(f"Missing: {evaluation['missing_terms']}")
            
            if verbose:
                print(f"\nResponse Preview:\n{response.answer[:300]}...")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "query": query,
                "category": query_info["category"],
                "error": str(e),
                "passed": False
            })
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    avg_quality = sum(r.get("quality_score", 0) for r in results) / total
    avg_confidence = sum(r.get("confidence", 0) for r in results) / total
    avg_time = total_time / total
    
    print(f"\n📊 Results:")
    print(f"   Passed: {passed}/{total} ({passed/total:.0%})")
    print(f"   Avg Quality Score: {avg_quality:.2f}")
    print(f"   Avg Confidence: {avg_confidence:.0%}")
    print(f"   Avg Response Time: {avg_time:.2f}s")
    print(f"   Total Time: {total_time:.2f}s")
    
    # By category
    print(f"\n📈 By Category:")
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1
    
    for cat, stats in categories.items():
        pct = stats["passed"] / stats["total"] * 100
        print(f"   {cat}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_path, f"eval_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {
                "passed": passed,
                "total": total,
                "pass_rate": passed/total,
                "avg_quality": avg_quality,
                "avg_confidence": avg_confidence,
                "avg_time": avg_time
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    run_evaluation(verbose=not args.quiet)
