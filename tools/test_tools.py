#!/usr/bin/env python3
"""Test all tools functionality."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import FactsLookupTool, SemanticRAGTool, CompareCompaniesTool


def test_facts_tool():
    """Test Facts Lookup Tool."""
    print("\n" + "=" * 70)
    print("TESTING FACTS LOOKUP TOOL")
    print("=" * 70)
    
    tool = FactsLookupTool()
    
    # Test 1: Get all companies
    print("\n📋 Test 1: Get all companies")
    result = tool.execute(action="get_all_companies")
    print(f"   {result.message}")
    if result.success:
        companies = result.data.get("companies", [])[:5]
        print(f"   Sample: {companies}")
    
    # Test 2: Get company details
    print("\n🏢 Test 2: Get company details (Dell)")
    result = tool.execute(action="get_company_details", company="Dell")
    print(f"   {result.message}")
    if result.success:
        print(f"   Roles: {result.data.get('count', 0)}")
    
    # Test 3: Filter by stipend
    print("\n💰 Test 3: Filter by stipend > 40000")
    result = tool.execute(action="filter_by_stipend", min_value=40000)
    print(f"   {result.message}")
    if result.success:
        for r in result.data.get("results", [])[:3]:
            print(f"   - {r['company']}: {r['stipend']}")
    
    # Test 4: Filter by location
    print("\n📍 Test 4: Filter by location (Bangalore)")
    result = tool.execute(action="filter_by_location", location="Bangalore")
    print(f"   {result.message}")
    
    # Test 5: Get all stipends
    print("\n💵 Test 5: Get all stipends")
    result = tool.execute(action="get_all_stipends")
    print(f"   {result.message}")
    if result.success:
        for s in result.data.get("stipends", [])[:3]:
            print(f"   - {s['company']}: {s['stipend']}")


def test_semantic_tool():
    """Test Semantic RAG Tool."""
    print("\n" + "=" * 70)
    print("TESTING SEMANTIC RAG TOOL")
    print("=" * 70)
    
    tool = SemanticRAGTool()
    
    # Test 1: General search
    print("\n🔍 Test 1: Search 'Python machine learning'")
    result = tool.execute(query="Python machine learning skills")
    print(f"   {result.message}")
    if result.success:
        for r in result.data.get("results", [])[:2]:
            print(f"   - [{r['company']}] ({r['type']}): {r['content'][:100]}...")
    
    # Test 2: Search by type
    print("\n🔍 Test 2: Search interview process")
    result = tool.execute(
        query="interview rounds technical HR",
        search_type="interview_process"
    )
    print(f"   {result.message}")
    if result.success:
        for r in result.data.get("results", [])[:2]:
            print(f"   - [{r['company']}]: {r['content'][:100]}...")
    
    # Test 3: Search by company
    print("\n🔍 Test 3: Search Dell company info")
    result = tool.execute(
        query="company culture work environment",
        search_type="about_company",
        company="Dell"
    )
    print(f"   {result.message}")
    if result.success:
        for r in result.data.get("results", [])[:2]:
            print(f"   - {r['content'][:150]}...")
    
    # Test 4: Skills search
    print("\n🔍 Test 4: Search required skills")
    result = tool.search_skills("data science analytics")
    print(f"   {result.message}")


def test_compare_tool():
    """Test Company Comparison Tool."""
    print("\n" + "=" * 70)
    print("TESTING COMPARE COMPANIES TOOL")
    print("=" * 70)
    
    tool = CompareCompaniesTool()
    
    # Get some companies first
    facts_tool = FactsLookupTool()
    companies_result = facts_tool.execute(action="get_all_companies")
    if not companies_result.success:
        print("   Could not get companies list")
        return
    
    all_companies = companies_result.data.get("companies", [])
    test_companies = all_companies[:3] if len(all_companies) >= 3 else all_companies
    
    print(f"\n   Testing with companies: {test_companies}")
    
    # Test 1: Table comparison
    print("\n📊 Test 1: Table comparison")
    result = tool.execute(
        companies=test_companies,
        comparison_type="table"
    )
    print(f"   {result.message}")
    if result.success:
        print(result.data.get("table", "No table"))
    
    # Test 2: Ranking
    print("\n🏆 Test 2: Rank by stipend")
    result = tool.execute(
        companies=test_companies,
        comparison_type="ranking",
        rank_by="stipend"
    )
    print(f"   {result.message}")
    if result.success:
        for r in result.data.get("rankings", []):
            print(f"   #{r['rank']} {r['company']}: {r.get('stipend', 'N/A')}")
    
    # Test 3: Detailed comparison
    print("\n📝 Test 3: Detailed comparison")
    result = tool.execute(
        companies=test_companies[:2],
        comparison_type="detailed"
    )
    print(f"   {result.message}")
    
    # Test 4: Find best
    print("\n⭐ Test 4: Find best company")
    result = tool.execute(
        companies=test_companies,
        comparison_type="best_for",
        attributes=["stipend"]
    )
    print(f"   {result.message}")
    if result.success:
        print(f"   Best: {result.data.get('best_company')}")


def main():
    """Run all tool tests."""
    print("\n" + "=" * 70)
    print("TOOLS FUNCTIONALITY TEST")
    print("=" * 70)
    
    test_facts_tool()
    test_semantic_tool()
    test_compare_tool()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
