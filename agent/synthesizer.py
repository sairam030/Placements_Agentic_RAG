"""LLM-powered Synthesizer - uses LLM to generate natural responses."""

from typing import Dict, List, Any

from agent.planner import QueryPlan
from agent.executor import ExecutionResult
from agent.critic import CriticFeedback
from agent.llm_client import get_agent_llm


class Synthesizer:
    """LLM-powered response generator."""
    
    SYSTEM_PROMPT = """You are a helpful placement information assistant.

CRITICAL RULES:
1. ONLY use data provided - NEVER make up information
2. If data is missing, say "Not available in database"
3. Use INR (₹) for currency
4. Format clearly with sections and bullet points
5. Be concise but complete"""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = get_agent_llm() if use_llm else None
    
    def synthesize(self, plan: QueryPlan, result: ExecutionResult, feedback: CriticFeedback) -> str:
        """Generate response using LLM."""
        
        # For aggregation, use rule-based (more accurate for counts)
        if plan.intent == "aggregation":
            return self._synthesize_aggregation(plan, result)
        
        # Build context from enriched results
        enriched = result.enriched_results or {}
        
        if not enriched and not result.tool_results:
            return self._no_data_response(plan)
        
        if self.use_llm and self.llm:
            return self._synthesize_with_llm(plan, enriched, result)
        else:
            return self._synthesize_rule_based(plan, enriched)
    
    def _synthesize_with_llm(self, plan: QueryPlan, enriched: Dict[str, Any], result: ExecutionResult) -> str:
        """Use LLM to generate natural response."""
        
        context = self._build_context(enriched, result)
        
        prompt = f"""Answer the user's question using ONLY the provided data.

**USER QUESTION:** {plan.original_query}

**AVAILABLE DATA:**
{context}

**INSTRUCTIONS:**
1. Answer the specific question asked
2. ONLY use information from the data above - do NOT add anything
3. If asking about selection/interview process, use INTERVIEW/SELECTION section
4. If asking about skills, use SKILLS section
5. For stipend/salary, use FACTS section
6. Format with clear sections, bullet points, emojis for readability
7. If information is not in the data, say "Not available in database"

**YOUR RESPONSE:**"""

        response = self.llm.generate(prompt, self.SYSTEM_PROMPT)
        
        if response and len(response) > 50:
            return response
        
        return self._synthesize_rule_based(plan, enriched)
    
    def _build_context(self, enriched: Dict[str, Any], result: ExecutionResult) -> str:
        """Build comprehensive context for LLM."""
        parts = []
        
        for company, data in enriched.items():
            parts.append(f"\n{'='*50}")
            parts.append(f"## COMPANY: {company.upper()}")
            parts.append(f"{'='*50}")
            
            # Facts section
            facts = data.get("facts", [])
            if facts:
                parts.append("\n### FACTS (Structured Data):")
                for i, fact in enumerate(facts):
                    if isinstance(fact, dict):
                        role = fact.get('role', fact.get('role_title', 'N/A'))
                        parts.append(f"\n**Role {i+1}: {role}**")
                        
                        # Stipend
                        stipend = fact.get('stipend', fact.get('stipend_salary', {}))
                        if isinstance(stipend, dict) and stipend.get('amount'):
                            parts.append(f"- Stipend: ₹{stipend.get('amount')} {stipend.get('period', 'per month')}")
                        
                        # Location
                        loc = fact.get('location', [])
                        if loc:
                            parts.append(f"- Location: {', '.join(loc) if isinstance(loc, list) else loc}")
                        
                        # Duration
                        if fact.get('duration'):
                            parts.append(f"- Duration: {fact['duration']}")
                        
                        # Eligibility
                        elig = fact.get('eligibility', {})
                        if isinstance(elig, dict):
                            cgpa = elig.get('cgpa_pg') or elig.get('cgpa_ug')
                            if cgpa:
                                parts.append(f"- Min CGPA: {cgpa}")
                            branches = elig.get('branches', [])
                            if branches:
                                parts.append(f"- Eligible Branches: {', '.join(branches[:5])}")
            
            # Semantic sections
            semantic = data.get("semantic", {})
            
            if semantic.get("interview_process"):
                parts.append("\n### INTERVIEW/SELECTION PROCESS:")
                parts.append(semantic["interview_process"][:2000])
            
            if semantic.get("skills_required"):
                parts.append("\n### SKILLS REQUIRED:")
                parts.append(semantic["skills_required"][:1500])
            
            if semantic.get("roles_responsibilities"):
                parts.append("\n### JOB RESPONSIBILITIES:")
                parts.append(semantic["roles_responsibilities"][:1000])
            
            if semantic.get("about_company"):
                parts.append("\n### ABOUT COMPANY:")
                parts.append(semantic["about_company"][:800])
        
        return "\n".join(parts)
    
    def _synthesize_aggregation(self, plan: QueryPlan, result: ExecutionResult) -> str:
        """Rule-based synthesis for aggregation (accurate counts)."""
        parts = []
        
        for tool_result in result.tool_results:
            if not tool_result.success or not tool_result.data:
                continue
            
            data = tool_result.data
            
            if "companies" in data:
                count = len(data["companies"])
                parts.append(f"📊 **Found {count} companies:**\n")
                for i, c in enumerate(data["companies"][:25], 1):
                    parts.append(f"{i}. {c}")
                if count > 25:
                    parts.append(f"\n... and {count - 25} more")
            
            elif "results" in data:
                results = data["results"]
                count = len(results)
                location = data.get("location", "specified criteria")
                parts.append(f"📍 **{count} companies matching {location}:**\n")
                
                for r in results[:20]:
                    if isinstance(r, dict):
                        company = r.get('company', r.get('company_name', 'N/A'))
                        role = r.get('role', r.get('role_title', ''))
                        stipend = r.get('stipend', '')
                        
                        line = f"• **{company}**"
                        if role:
                            line += f" - {role}"
                        if isinstance(stipend, dict) and stipend.get('amount'):
                            line += f" | ₹{stipend['amount']}/month"
                        parts.append(line)
        
        return "\n".join(parts) if parts else self._no_data_response(plan)
    
    def _synthesize_rule_based(self, plan: QueryPlan, enriched: Dict[str, Any]) -> str:
        """Fallback rule-based synthesis."""
        parts = []
        query_lower = plan.original_query.lower()
        
        wants_selection = any(kw in query_lower for kw in ['selection', 'interview', 'process', 'rounds'])
        wants_skills = any(kw in query_lower for kw in ['skills', 'requirements'])
        
        for company, data in enriched.items():
            parts.append(f"\n## 🏢 {company.upper()}")
            
            facts = data.get("facts", [])
            semantic = data.get("semantic", {})
            
            if facts:
                parts.append("\n**Roles:**")
                for fact in facts:
                    if isinstance(fact, dict):
                        role = fact.get('role', fact.get('role_title', 'N/A'))
                        stipend = fact.get('stipend', fact.get('stipend_salary', {}))
                        if isinstance(stipend, dict) and stipend.get('amount'):
                            parts.append(f"- {role}: ₹{stipend['amount']}/month")
                        else:
                            parts.append(f"- {role}")
            
            if wants_selection and semantic.get("interview_process"):
                parts.append("\n### 🎯 Selection Process:")
                parts.append(semantic["interview_process"][:1500])
            
            if wants_skills and semantic.get("skills_required"):
                parts.append("\n### 🛠️ Skills:")
                parts.append(semantic["skills_required"][:1000])
        
        return "\n".join(parts) if parts else self._no_data_response(plan)
    
    def _no_data_response(self, plan: QueryPlan) -> str:
        return f"""❌ **No information found**

Query: "{plan.original_query}"

Try: Check spelling or use 'companies' to see available companies."""
