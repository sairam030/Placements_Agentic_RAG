"""LLM-powered Critic - validates results using LLM."""

from typing import Dict, List, Any
from dataclasses import dataclass

from agent.planner import QueryPlan
from agent.executor import ExecutionResult
from agent.llm_client import get_agent_llm


@dataclass
class CriticFeedback:
    """Feedback from the critic."""
    is_complete: bool
    is_relevant: bool
    missing_info: List[str]
    suggestions: List[str]
    confidence_score: float
    needs_retry: bool
    retry_suggestions: List[Dict[str, Any]]
    reasoning: str = ""


class Critic:
    """LLM-powered result validator."""
    
    SYSTEM_PROMPT = """You are a quality evaluator for a placement information system.
Evaluate whether the retrieved data adequately answers the user's question.
Be strict but fair - if the data contains the answer, it's complete."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = get_agent_llm() if use_llm else None
    
    def evaluate(self, plan: QueryPlan, result: ExecutionResult) -> CriticFeedback:
        """Evaluate results using LLM."""
        
        if self.use_llm and self.llm:
            return self._evaluate_with_llm(plan, result)
        else:
            return self._evaluate_rule_based(plan, result)
    
    def _evaluate_with_llm(self, plan: QueryPlan, result: ExecutionResult) -> CriticFeedback:
        """Use LLM to evaluate completeness and relevance."""
        
        # Summarize what we found
        summary = self._summarize_results(result)
        
        prompt = f"""Evaluate if the retrieved data answers the user's question.

**User Question:** "{plan.original_query}"
**Intent:** {plan.intent}
**Companies Asked:** {plan.companies_mentioned}
**Attributes Requested:** {plan.attributes_requested}

**Retrieved Data Summary:**
{summary}

Return JSON:
```json
{{
    "is_complete": true/false,
    "is_relevant": true/false,
    "confidence": 0.0-1.0,
    "missing_info": ["list of missing information"],
    "needs_retry": true/false,
    "reasoning": "brief explanation"
}}
```

RULES:
- is_complete=true if the data contains info to answer the question
- is_relevant=true if the data is about what was asked
- confidence: 0.9+ if complete, 0.6-0.9 if partial, <0.6 if poor
- needs_retry=true only if confidence < 0.4

Return only JSON:"""

        eval_result = self.llm.generate_json(prompt, self.SYSTEM_PROMPT)
        
        if eval_result:
            return CriticFeedback(
                is_complete=eval_result.get("is_complete", False),
                is_relevant=eval_result.get("is_relevant", False),
                missing_info=eval_result.get("missing_info", []),
                suggestions=[],
                confidence_score=eval_result.get("confidence", 0.5),
                needs_retry=eval_result.get("needs_retry", False),
                retry_suggestions=[{"tool": "hybrid_search"}] if eval_result.get("needs_retry") else [],
                reasoning=eval_result.get("reasoning", "")
            )
        
        return self._evaluate_rule_based(plan, result)
    
    def _summarize_results(self, result: ExecutionResult) -> str:
        """Create summary of results for LLM."""
        parts = []
        
        for tool_result in result.tool_results:
            if not tool_result.success:
                parts.append(f"[{tool_result.tool_name}] FAILED")
                continue
            
            data = tool_result.data or {}
            parts.append(f"[{tool_result.tool_name}] SUCCESS:")
            
            if "companies" in data:
                parts.append(f"  - Found {len(data['companies'])} companies")
            if "results" in data:
                parts.append(f"  - Found {len(data['results'])} results")
            if "roles" in data:
                parts.append(f"  - Found {len(data['roles'])} roles")
        
        if result.enriched_results:
            parts.append(f"[Enriched] Companies: {list(result.enriched_results.keys())}")
            for company, data in result.enriched_results.items():
                facts_count = len(data.get("facts", []))
                semantic_types = list(data.get("semantic", {}).keys())
                parts.append(f"  - {company}: {facts_count} roles, semantic: {semantic_types}")
        
        return "\n".join(parts) if parts else "No data retrieved"
    
    def _evaluate_rule_based(self, plan: QueryPlan, result: ExecutionResult) -> CriticFeedback:
        """Fallback rule-based evaluation."""
        has_data = any(r.success and r.data for r in result.tool_results) or bool(result.enriched_results)
        
        confidence = 0.8 if has_data else 0.2
        
        return CriticFeedback(
            is_complete=has_data,
            is_relevant=has_data,
            missing_info=[],
            suggestions=[],
            confidence_score=confidence,
            needs_retry=confidence < 0.4,
            retry_suggestions=[],
            reasoning="Rule-based evaluation"
        )
