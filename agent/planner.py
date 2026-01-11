"""LLM-powered Planner - uses LLM to understand query and select tools."""

import re
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

from agent.llm_client import get_agent_llm


class ToolType(Enum):
    FACTS_LOOKUP = "facts_lookup"
    SEMANTIC_SEARCH = "semantic_search"
    COMPARE_COMPANIES = "compare_companies"
    HYBRID_SEARCH = "hybrid_search"


@dataclass
class QueryPlan:
    """Plan for executing a query."""
    original_query: str
    intent: str
    tools_to_use: List[Dict[str, Any]]
    is_comparison: bool = False
    companies_mentioned: List[str] = field(default_factory=list)
    attributes_requested: List[str] = field(default_factory=list)
    needs_enrichment: bool = True
    fallback_to_hybrid: bool = False
    reasoning: str = ""


class Planner:
    """LLM-powered query planner."""
    
    SYSTEM_PROMPT = """You are a query planning assistant for a placement/internship information system.

Available tools:
1. **facts_lookup** - For structured data queries:
   - Actions: get_all_companies, get_company_details, get_all_stipends, filter_by_stipend, filter_by_cgpa, filter_by_location, filter_by_branch, get_eligibility, get_selection_process
   - Use for: stipend amounts, CGPA requirements, locations, eligibility, counting/listing companies

2. **semantic_search** - For descriptive information:
   - Search types: about_company, roles_responsibilities, skills_required, interview_process
   - Use for: company culture, job descriptions, skills needed, interview details

3. **compare_companies** - For comparing 2+ companies
   - Use for: side-by-side comparison queries

4. **hybrid_search** - For queries needing BOTH facts AND descriptions
   - Use for: detailed company info, selection process details, comprehensive queries

Analyze the query and decide the best tool(s) to use."""

    def __init__(self, known_companies: List[str] = None, use_llm: bool = True):
        self.KNOWN_COMPANIES = set(c.lower() for c in (known_companies or []))
        self.use_llm = use_llm
        self.llm = get_agent_llm() if use_llm else None
    
    def analyze(self, query: str) -> QueryPlan:
        """Analyze query using LLM."""
        
        # Extract companies first (rule-based for accuracy)
        companies = self._extract_companies_fuzzy(query)
        
        if self.use_llm and self.llm:
            return self._analyze_with_llm(query, companies)
        else:
            return self._analyze_rule_based(query, companies)
    
    def _analyze_with_llm(self, query: str, detected_companies: List[str]) -> QueryPlan:
        """Use LLM to analyze query and create plan."""
        
        prompt = f"""Analyze this placement query and create an execution plan.

**Query:** "{query}"
**Detected companies:** {detected_companies}
**Known companies:** {list(self.KNOWN_COMPANIES)[:25]}

Return JSON:
```json
{{
    "intent": "aggregation|comparison|company_detail|general",
    "reasoning": "brief explanation",
    "companies": ["company1"],
    "attributes": ["stipend", "selection", "skills", "eligibility", "location"],
    "is_aggregation": true/false,
    "is_comparison": true/false,
    "tool": {{
        "name": "facts_lookup|semantic_search|compare_companies|hybrid_search",
        "action": "filter_by_location|get_company_details|etc (for facts_lookup)",
        "params": {{"location": "Bangalore", "company": "dell", "min_value": 40000}}
    }}
}}
```

RULES:
- For "how many/list/which companies" → facts_lookup with appropriate filter action
- For "company X details/selection/skills" → hybrid_search (needs both facts + semantic)
- For "compare X and Y" → compare_companies
- For aggregation (count, filter), set is_aggregation=true

Return only valid JSON:"""

        result = self.llm.generate_json(prompt, self.SYSTEM_PROMPT)
        
        if result:
            tool_config = result.get("tool", {})
            tool_name = tool_config.get("name", "hybrid_search")
            
            tools = [{
                "tool": tool_name,
                "action": tool_config.get("action"),
                "params": tool_config.get("params", {})
            }]
            
            # Ensure companies from LLM + detected are included
            companies = list(set(
                (result.get("companies") or []) + detected_companies
            ))
            
            return QueryPlan(
                original_query=query,
                intent=result.get("intent", "general"),
                tools_to_use=tools,
                is_comparison=result.get("is_comparison", False),
                companies_mentioned=companies,
                attributes_requested=result.get("attributes", []),
                needs_enrichment=not result.get("is_aggregation", False),
                fallback_to_hybrid=False,
                reasoning=result.get("reasoning", "LLM analysis")
            )
        
        # Fallback to rule-based
        return self._analyze_rule_based(query, detected_companies)
    
    def _analyze_rule_based(self, query: str, companies: List[str]) -> QueryPlan:
        """Fallback rule-based analysis."""
        query_lower = query.lower()
        
        # Aggregation detection
        is_aggregation = any(p in query_lower for p in ['how many', 'list all', 'which companies', 'count'])
        is_filter = any(p in query_lower for p in ['companies in', 'companies with', 'stipend more', 'cgpa less'])
        is_comparison = len(companies) >= 2 or any(kw in query_lower for kw in ['compare', 'vs', 'versus'])
        
        if is_aggregation or is_filter:
            return self._plan_aggregation(query_lower, companies)
        elif is_comparison:
            return self._plan_comparison(query, companies)
        elif companies:
            return self._plan_hybrid(query, companies)
        else:
            return self._plan_hybrid(query, [])
    
    def _plan_aggregation(self, query: str, companies: List[str]) -> QueryPlan:
        """Plan for aggregation queries."""
        action = "get_all_companies"
        params = {}
        
        # Location filter
        locations = {'bangalore': 'Bangalore', 'hyderabad': 'Hyderabad', 'chennai': 'Chennai', 
                    'mumbai': 'Mumbai', 'delhi': 'Delhi', 'pune': 'Pune', 'noida': 'Noida'}
        for loc_key, loc_val in locations.items():
            if loc_key in query:
                action = "filter_by_location"
                params["location"] = loc_val
                break
        
        # Stipend filter
        if 'stipend' in query:
            numbers = re.findall(r'\d+', query)
            if numbers and any(kw in query for kw in ['more', 'greater', 'above']):
                action = "filter_by_stipend"
                params["min_value"] = float(numbers[0])
        
        return QueryPlan(
            original_query=query,
            intent="aggregation",
            tools_to_use=[{"tool": "facts_lookup", "action": action, "params": params}],
            companies_mentioned=companies,
            needs_enrichment=False,
            reasoning="Aggregation query"
        )
    
    def _plan_comparison(self, query: str, companies: List[str]) -> QueryPlan:
        return QueryPlan(
            original_query=query,
            intent="comparison",
            tools_to_use=[{"tool": "compare_companies", "params": {"companies": companies, "comparison_type": "detailed"}}],
            is_comparison=True,
            companies_mentioned=companies,
            needs_enrichment=True,
            reasoning="Comparison query"
        )
    
    def _plan_hybrid(self, query: str, companies: List[str]) -> QueryPlan:
        return QueryPlan(
            original_query=query,
            intent="hybrid",
            tools_to_use=[{"tool": "hybrid_search", "params": {"query": query, "companies": companies, "top_k": 5}}],
            companies_mentioned=companies,
            needs_enrichment=True,
            reasoning="Hybrid query for comprehensive results"
        )
    
    def _extract_companies_fuzzy(self, query: str) -> List[str]:
        """Extract companies with fuzzy matching."""
        query_lower = query.lower()
        companies = [c for c in self.KNOWN_COMPANIES if c in query_lower]
        
        fuzzy = {'intell': 'intel', 'dell': 'dell', 'nvidia': 'nvidia', 'bosch': 'bosch', 'amazon': 'amazon'}
        for mis, correct in fuzzy.items():
            if mis in query_lower and correct in self.KNOWN_COMPANIES and correct not in companies:
                companies.append(correct)
        
        return list(set(companies))
