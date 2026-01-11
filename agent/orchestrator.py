"""Main orchestrator - coordinates all LLM-powered agent components."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agent.planner import Planner, QueryPlan
from agent.executor import Executor, ExecutionResult
from agent.critic import Critic, CriticFeedback
from agent.synthesizer import Synthesizer
from agent.config import USE_LLM_PLANNER, USE_LLM_CRITIC, USE_LLM_SYNTHESIZER


@dataclass
class AgentResponse:
    """Final response from the agent."""
    answer: str
    plan: QueryPlan
    execution: ExecutionResult
    feedback: CriticFeedback
    retries: int = 0


class PlacementAgent:
    """
    LLM-powered placement query agent.
    
    Components:
    1. Planner (LLM) - Analyzes query, selects tools
    2. Executor - Runs tools, enriches results
    3. Critic (LLM) - Validates completeness
    4. Synthesizer (LLM) - Generates natural response
    """
    
    def __init__(self, known_companies: List[str] = None, use_llm: bool = True):
        self.planner = Planner(
            known_companies=known_companies, 
            use_llm=use_llm and USE_LLM_PLANNER
        )
        self.executor = Executor()
        self.critic = Critic(use_llm=use_llm and USE_LLM_CRITIC)
        self.synthesizer = Synthesizer(use_llm=use_llm and USE_LLM_SYNTHESIZER)
        
        self.max_retries = 2
        self.min_confidence = 0.4
    
    def query(self, user_query: str, verbose: bool = False) -> AgentResponse:
        """Process a user query and return response."""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎓 Query: {user_query}")
            print(f"{'='*60}")
        
        # Step 1: Plan (LLM-powered)
        plan = self.planner.analyze(user_query)
        
        if verbose:
            print(f"\n📋 **Plan:**")
            print(f"   Intent: {plan.intent}")
            print(f"   Tools: {[t.get('tool') for t in plan.tools_to_use]}")
            print(f"   Companies: {plan.companies_mentioned}")
            if plan.reasoning:
                print(f"   Reasoning: {plan.reasoning}")
        
        # Step 2: Execute
        result = self.executor.execute(plan)
        
        if verbose:
            print(f"\n⚡ **Execution:**")
            print(f"   Success: {result.success}")
            print(f"   Tools run: {len(result.tool_results)}")
            print(f"   Enriched: {bool(result.enriched_results)}")
        
        # Step 3: Critique (LLM-powered)
        feedback = self.critic.evaluate(plan, result)
        
        if verbose:
            print(f"\n🔍 **Critique:**")
            print(f"   Complete: {feedback.is_complete}")
            print(f"   Relevant: {feedback.is_relevant}")
            print(f"   Confidence: {feedback.confidence_score:.0%}")
            if feedback.reasoning:
                print(f"   Reasoning: {feedback.reasoning}")
        
        # Step 4: Retry if needed
        retries = 0
        while feedback.needs_retry and retries < self.max_retries:
            retries += 1
            if verbose:
                print(f"\n🔄 Retry {retries}...")
            
            plan = self._adjust_plan(plan, feedback)
            result = self.executor.execute(plan)
            feedback = self.critic.evaluate(plan, result)
        
        # Step 5: Synthesize (LLM-powered)
        answer = self.synthesizer.synthesize(plan, result, feedback)
        
        if verbose:
            print(f"\n{'='*60}")
            print("✅ Response generated!")
            print(f"{'='*60}\n")
        
        return AgentResponse(
            answer=answer,
            plan=plan,
            execution=result,
            feedback=feedback,
            retries=retries
        )
    
    def _adjust_plan(self, plan: QueryPlan, feedback: CriticFeedback) -> QueryPlan:
        """Adjust plan based on feedback."""
        if feedback.retry_suggestions:
            suggestion = feedback.retry_suggestions[0]
            tool = suggestion.get("tool", "hybrid_search")
            
            plan.tools_to_use = [{
                "tool": tool,
                "params": {
                    "query": plan.original_query,
                    "companies": plan.companies_mentioned,
                    "top_k": 10
                }
            }]
        else:
            # Default: hybrid search
            plan.tools_to_use = [{
                "tool": "hybrid_search",
                "params": {
                    "query": plan.original_query,
                    "companies": plan.companies_mentioned,
                    "top_k": 10
                }
            }]
            plan.fallback_to_hybrid = True
        
        return plan
    
    def get_companies(self) -> List[str]:
        """Get list of all available companies."""
        from tools import FactsLookupTool
        tool = FactsLookupTool()
        result = tool.execute(action="get_all_companies")
        if result.success and result.data:
            return result.data.get("companies", [])
        return []


def create_agent(use_llm: bool = True) -> PlacementAgent:
    """Create and initialize the placement agent."""
    from tools import FactsLookupTool
    
    facts_tool = FactsLookupTool()
    result = facts_tool.execute(action="get_all_companies")
    
    known_companies = []
    if result.success and result.data:
        known_companies = result.data.get("companies", [])
    
    return PlacementAgent(known_companies=known_companies, use_llm=use_llm)
