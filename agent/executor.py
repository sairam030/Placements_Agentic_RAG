"""Executor - handles aggregation queries differently (no enrichment needed)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from tools import FactsLookupTool, SemanticRAGTool, CompareCompaniesTool
from tools.base_tool import ToolResult
from agent.planner import QueryPlan, ToolType


@dataclass
class ExecutionResult:
    """Result from executing a query plan."""
    success: bool
    tool_results: List[ToolResult]
    enriched_results: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    
    def get_all_data(self) -> Dict[str, Any]:
        combined = {}
        for result in self.tool_results:
            if result.success and result.data:
                combined[result.tool_name] = result.data
        if self.enriched_results:
            combined["enriched"] = self.enriched_results
        return combined


class Executor:
    """Executes queries - skips enrichment for aggregation queries."""
    
    SEMANTIC_TOP_K = 3
    
    def __init__(self):
        self.facts_tool = FactsLookupTool()
        self.semantic_tool = SemanticRAGTool()
        self.compare_tool = CompareCompaniesTool()
    
    def execute(self, plan: QueryPlan) -> ExecutionResult:
        """Execute plan."""
        
        tool_results = []
        errors = []
        
        for tool_call in plan.tools_to_use:
            tool_type = tool_call.get("tool")
            params = tool_call.get("params", {})
            
            try:
                if tool_type == ToolType.FACTS_LOOKUP.value:
                    result = self._execute_facts(tool_call, plan)
                elif tool_type == ToolType.SEMANTIC_SEARCH.value:
                    result = self._execute_semantic(params)
                elif tool_type == ToolType.COMPARE_COMPANIES.value:
                    result = self._execute_compare(params, plan)
                elif tool_type == ToolType.HYBRID_SEARCH.value:
                    result = self._execute_hybrid(params, plan)
                else:
                    errors.append(f"Unknown tool: {tool_type}")
                    continue
                
                tool_results.append(result)
            except Exception as e:
                errors.append(f"Error: {str(e)}")
        
        # Only enrich for detailed queries, NOT for aggregation
        enriched = None
        if plan.needs_enrichment and plan.intent != "aggregation":
            enriched = self._enrich_comprehensive(tool_results, plan)
        
        return ExecutionResult(
            success=len(tool_results) > 0 and any(r.success for r in tool_results),
            tool_results=tool_results,
            enriched_results=enriched,
            errors=errors
        )
    
    def _execute_facts(self, tool_call: Dict[str, Any], plan: QueryPlan) -> ToolResult:
        """Execute facts lookup."""
        action = tool_call.get("action", "get_all_companies")
        params = tool_call.get("params", {})
        
        # For company-specific queries with no specific action
        if plan.companies_mentioned and action == "get_company_details":
            all_results = []
            for company in plan.companies_mentioned:
                result = self.facts_tool.execute(action="get_company_details", company=company)
                if result.success and result.data:
                    all_results.extend(result.data.get("roles", []))
            
            if all_results:
                return ToolResult(
                    success=True,
                    data={"roles": all_results, "count": len(all_results)},
                    message=f"Found {len(all_results)} roles",
                    tool_name="facts_lookup",
                    query=str(plan.companies_mentioned)
                )
        
        return self.facts_tool.execute(action=action, **params)
    
    def _execute_semantic(self, params: Dict[str, Any]) -> ToolResult:
        return self.semantic_tool.execute(**params)
    
    def _execute_compare(self, params: Dict[str, Any], plan: QueryPlan) -> ToolResult:
        return self.compare_tool.execute(**params)
    
    def _execute_hybrid(self, params: Dict[str, Any], plan: QueryPlan) -> ToolResult:
        """Execute hybrid search."""
        query = params.get("query", plan.original_query)
        companies = params.get("companies", plan.companies_mentioned)
        
        semantic_result = self.semantic_tool.execute(
            query=query, search_type="general", top_k=self.SEMANTIC_TOP_K
        )
        
        companies_found = set(companies) if companies else set()
        if semantic_result.success and semantic_result.data:
            for r in semantic_result.data.get("results", []):
                if r.get("company"):
                    companies_found.add(r.get("company"))
        
        facts_data = {}
        for company in companies_found:
            fact_result = self.facts_tool.execute(action="get_company_details", company=company)
            if fact_result.success and fact_result.data:
                facts_data[company] = fact_result.data
        
        return ToolResult(
            success=True,
            data={
                "semantic_results": semantic_result.data if semantic_result.success else {},
                "facts_data": facts_data,
                "companies": list(companies_found),
            },
            message=f"Found {len(companies_found)} companies",
            tool_name="hybrid_search",
            query=query
        )
    
    def _enrich_comprehensive(self, tool_results: List[ToolResult], plan: QueryPlan) -> Dict[str, Any]:
        """Comprehensive enrichment for detailed queries."""
        enriched = {}
        companies_to_enrich = set()
        
        if plan.companies_mentioned:
            companies_to_enrich.update(c.lower() for c in plan.companies_mentioned)
        
        for result in tool_results:
            if not result.success or not result.data:
                continue
            
            # Collect companies from results
            for r in result.data.get("results", result.data.get("semantic_results", {}).get("results", [])):
                if r.get("company"):
                    companies_to_enrich.add(r.get("company").lower())
            
            for role in result.data.get("roles", []):
                company = role.get("company", role.get("company_name", ""))
                if company:
                    companies_to_enrich.add(company.lower())
            
            for company in result.data.get("facts_data", {}).keys():
                companies_to_enrich.add(company.lower())
        
        # Enrich each company
        for company_lower in companies_to_enrich:
            if not company_lower:
                continue
            
            company_display = company_lower.title()
            enriched[company_display] = {"facts": [], "semantic": {}}
            
            # Get facts
            fact_result = self.facts_tool.execute(action="get_company_details", company=company_lower)
            if fact_result.success and fact_result.data:
                enriched[company_display]["facts"] = fact_result.data.get("roles", [])
            
            # Get semantic (top 3 per category)
            categories = {
                "skills_required": f"{company_lower} skills programming technical",
                "interview_process": f"{company_lower} interview selection rounds test",
                "roles_responsibilities": f"{company_lower} job responsibilities duties",
                "about_company": f"{company_lower} company culture",
            }
            
            for cat, query in categories.items():
                sem_result = self.semantic_tool.execute(
                    query=query, search_type=cat, company=company_lower, top_k=self.SEMANTIC_TOP_K
                )
                if sem_result.success and sem_result.data:
                    results = sem_result.data.get("results", [])
                    if results:
                        combined = "\n\n---\n\n".join([
                            r.get("content", r.get("text", "")) for r in results[:self.SEMANTIC_TOP_K]
                        ])
                        enriched[company_display]["semantic"][cat] = combined
        
        return enriched
