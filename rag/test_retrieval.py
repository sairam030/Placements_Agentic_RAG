#!/usr/bin/env python3
"""Test retrieval quality from both indices."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.semantic_index import SemanticIndex
from rag.facts_index import FactsIndex


def test_semantic_search():
    """Test semantic search functionality."""
    print("\n" + "=" * 70)
    print("TESTING SEMANTIC SEARCH")
    print("=" * 70)
    
    index = SemanticIndex()
    if not index.load():
        print("❌ Failed to load semantic index. Run build_index.py first.")
        return
    
    # Test queries
    test_queries = [
        ("Python machine learning skills", None, None),
        ("interview process rounds", None, "interview_process"),
        ("company culture work environment", None, "about_company"),
        ("data science responsibilities", None, "roles_responsibilities"),
        ("Dell technologies", "Dell", None),
    ]
    
    for query, company_filter, type_filter in test_queries:
        print(f"\n🔍 Query: '{query}'")
        if company_filter:
            print(f"   Filter: company={company_filter}")
        if type_filter:
            print(f"   Filter: type={type_filter}")
        
        results = index.search(
            query,
            top_k=3,
            filter_company=company_filter,
            filter_type=type_filter
        )
        
        print(f"   Found {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"\n   [{i}] {r['company']} - {r['role']} ({r['type']})")
            print(f"       Score: {r['score']:.4f}")
            print(f"       Text: {r['text'][:150]}...")


def test_facts_queries():
    """Test facts-based queries."""
    print("\n" + "=" * 70)
    print("TESTING FACTS QUERIES")
    print("=" * 70)
    
    index = FactsIndex()
    if not index.load():
        print("❌ Failed to load facts index. Run build_index.py first.")
        return
    
    # Test 1: Get all companies
    print("\n📋 All Companies:")
    companies = index.get_all_companies()
    print(f"   {len(companies)} companies: {', '.join(companies[:10])}...")
    
    # Test 2: Get stipends
    print("\n💰 Stipend Information:")
    stipends = index.get_all_stipends()
    for s in stipends[:5]:
        print(f"   {s['company']} - {s['role_title']}: {s['stipend']}")
    
    # Test 3: Filter by stipend
    print("\n💰 Companies with stipend > 40000:")
    high_stipend = index.filter_by_stipend(min_amount=40000)
    for f in high_stipend[:5]:
        stipend = f.get('stipend_salary', {})
        if isinstance(stipend, dict):
            amt = stipend.get('amount', 'N/A')
        else:
            amt = stipend
        print(f"   {f['company_name']}: {amt}")
    
    # Test 4: Filter by location
    print("\n📍 Companies in Bangalore:")
    bangalore = index.filter_by_location("Bangalore")
    for f in bangalore[:5]:
        print(f"   {f['company_name']} - {f.get('role_title', 'N/A')}")
    
    # Test 5: Get by company
    print("\n🏢 Dell Technologies details:")
    dell = index.get_by_company("Dell")
    for d in dell:
        print(f"   Role: {d.get('role_title', 'N/A')}")
        print(f"   Stipend: {d.get('stipend_salary', 'N/A')}")
        print(f"   Location: {d.get('location', 'N/A')}")
    
    # Test 6: Search attribute
    print("\n🔍 Selection process across companies:")
    processes = index.search_attribute("selection_process")
    for p in processes[:5]:
        rounds = p.get('selection_process', [])
        if rounds:
            print(f"   {p['company']}: {len(rounds)} rounds")


def test_combined_queries():
    """Test queries that need both indices."""
    print("\n" + "=" * 70)
    print("TESTING COMBINED QUERIES")
    print("=" * 70)
    
    semantic = SemanticIndex()
    facts = FactsIndex()
    
    semantic.load()
    facts.load()
    
    # Query: Companies with high stipend that need Python
    print("\n🔍 Query: 'Companies with stipend > 40000 that need Python'")
    
    # Step 1: Get high stipend companies from facts
    high_stipend = facts.filter_by_stipend(min_amount=40000)
    high_stipend_companies = set(f['company_name'] for f in high_stipend)
    print(f"   Step 1: {len(high_stipend_companies)} companies with high stipend")
    
    # Step 2: Search for Python skills in those companies
    print(f"   Step 2: Searching for Python skills...")
    python_results = semantic.search("Python programming skills required", top_k=20)
    
    # Step 3: Intersection
    matching = []
    for r in python_results:
        if r['company'] in high_stipend_companies:
            matching.append(r)
    
    print(f"\n   ✅ Found {len(matching)} matching companies:")
    for m in matching[:5]:
        company_facts = facts.get_by_company(m['company'])
        stipend = "N/A"
        if company_facts:
            s = company_facts[0].get('stipend_salary', {})
            if isinstance(s, dict):
                stipend = s.get('amount', 'N/A')
        print(f"      - {m['company']} (Stipend: {stipend})")
        print(f"        Skills: {m['text'][:100]}...")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RAG RETRIEVAL QUALITY TEST")
    print("=" * 70)
    
    test_semantic_search()
    test_facts_queries()
    test_combined_queries()
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
