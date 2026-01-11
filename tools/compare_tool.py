"""Company Comparison Tool for comparing multiple companies."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
from tools.base_tool import BaseTool, ToolResult
from rag.facts_index import FactsIndex
from rag.semantic_index import SemanticIndex


class CompareCompaniesTool(BaseTool):
    """
    Tool for comparing multiple companies on various attributes.
    
    Use this tool for:
    - Comparing stipends across companies
    - Comparing eligibility criteria
    - Comparing selection processes
    - Side-by-side feature comparison
    - Finding best company based on criteria
    """
    
    name = "compare_companies"
    description = """Compare multiple companies on various attributes like stipend, 
    eligibility, location, skills required, interview rounds, etc. 
    Use for side-by-side comparisons and finding the best option."""
    
    def __init__(self):
        self.facts_index = FactsIndex()
        self.semantic_index = SemanticIndex()
        self._loaded = False
    
    def _ensure_loaded(self):
        """Ensure indices are loaded."""
        if not self._loaded:
            self.facts_index.load()
            self.semantic_index.load()
            self._loaded = True
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "companies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of company names to compare"
                },
                "attributes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Attributes to compare (stipend, cgpa, location, etc.)"
                },
                "comparison_type": {
                    "type": "string",
                    "enum": ["table", "detailed", "ranking", "best_for"],
                    "description": "Type of comparison output"
                },
                "rank_by": {
                    "type": "string",
                    "description": "Attribute to rank companies by"
                }
            },
            "required": ["companies"]
        }
    
    def get_capabilities(self) -> List[str]:
        return [
            "Compare stipends across multiple companies",
            "Compare eligibility requirements",
            "Compare selection process complexity",
            "Compare locations and work modes",
            "Rank companies by specific attribute",
            "Find best company for given criteria",
            "Generate comparison table",
            "Detailed side-by-side comparison"
        ]
    
    def execute(
        self,
        companies: List[str],
        attributes: List[str] = None,
        comparison_type: str = "table",
        rank_by: str = None,
        **kwargs
    ) -> ToolResult:
        """Execute a company comparison."""
        
        self._ensure_loaded()
        
        if not companies or len(companies) < 2:
            return ToolResult(
                success=False,
                data=None,
                message="At least 2 companies required for comparison",
                tool_name=self.name,
                query=str(companies)
            )
        
        try:
            if comparison_type == "table":
                return self._compare_table(companies, attributes)
            elif comparison_type == "detailed":
                return self._compare_detailed(companies)
            elif comparison_type == "ranking":
                return self._compare_ranking(companies, rank_by)
            elif comparison_type == "best_for":
                return self._find_best(companies, attributes)
            else:
                return self._compare_table(companies, attributes)
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Comparison error: {str(e)}",
                tool_name=self.name,
                query=str(companies)
            )
    
    def _compare_table(self, companies: List[str], attributes: List[str] = None) -> ToolResult:
        """Generate comparison table."""
        
        if attributes is None:
            attributes = ["stipend", "cgpa", "location", "duration", "num_rounds", "work_mode"]
        
        comparison = {}
        found_companies = []
        
        for company in companies:
            facts = self.facts_index.get_by_company(company)
            if not facts:
                continue
            
            found_companies.append(company)
            
            # Use first role if multiple
            fact = facts[0]
            
            company_data = {
                "role": fact.get("role_title") or fact.get("role_name", "N/A")
            }
            
            for attr in attributes:
                if attr == "stipend":
                    stipend = fact.get("stipend_salary", {})
                    if isinstance(stipend, dict):
                        company_data[attr] = stipend.get("amount", "N/A")
                    else:
                        company_data[attr] = str(stipend) if stipend else "N/A"
                
                elif attr == "cgpa":
                    elig = fact.get("eligibility", {})
                    if isinstance(elig, dict):
                        company_data[attr] = elig.get("cgpa_pg") or elig.get("cgpa_ug", "N/A")
                    else:
                        company_data[attr] = "N/A"
                
                elif attr == "location":
                    loc = fact.get("location", [])
                    if isinstance(loc, list):
                        company_data[attr] = ", ".join(loc) if loc else "N/A"
                    else:
                        company_data[attr] = str(loc) if loc else "N/A"
                
                elif attr == "num_rounds":
                    process = fact.get("selection_process", [])
                    company_data[attr] = len(process) if isinstance(process, list) else "N/A"
                
                elif attr == "duration":
                    company_data[attr] = fact.get("duration", "N/A")
                
                elif attr == "work_mode":
                    company_data[attr] = fact.get("work_mode", "N/A")
                
                else:
                    company_data[attr] = fact.get(attr, "N/A")
            
            comparison[company] = company_data
        
        # Format as table string
        table_str = self._format_table(comparison, attributes)
        
        return ToolResult(
            success=True,
            data={
                "comparison": comparison,
                "companies": found_companies,
                "attributes": attributes,
                "table": table_str
            },
            message=f"Compared {len(found_companies)} companies on {len(attributes)} attributes",
            tool_name=self.name,
            query=f"compare:{','.join(companies)}"
        )
    
    def _format_table(self, comparison: Dict, attributes: List[str]) -> str:
        """Format comparison as ASCII table."""
        if not comparison:
            return "No data to compare"
        
        # Headers
        companies = list(comparison.keys())
        header = ["Attribute"] + companies
        
        # Rows
        rows = []
        rows.append(["Role"] + [comparison[c].get("role", "N/A") for c in companies])
        
        for attr in attributes:
            row = [attr.replace("_", " ").title()]
            for company in companies:
                val = comparison[company].get(attr, "N/A")
                row.append(str(val) if val else "N/A")
            rows.append(row)
        
        # Calculate column widths
        widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
        
        # Build table
        lines = []
        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        
        lines.append(separator)
        lines.append("|" + "|".join(f" {header[i]:<{widths[i]}} " for i in range(len(header))) + "|")
        lines.append(separator)
        
        for row in rows:
            lines.append("|" + "|".join(f" {str(row[i]):<{widths[i]}} " for i in range(len(row))) + "|")
        
        lines.append(separator)
        
        return "\n".join(lines)
    
    def _compare_detailed(self, companies: List[str]) -> ToolResult:
        """Generate detailed comparison with semantic info."""
        
        detailed = {}
        
        for company in companies:
            # Get facts
            facts = self.facts_index.get_by_company(company)
            
            # Get semantic chunks
            chunks = self.semantic_index.get_all_by_company(company)
            
            if not facts and not chunks:
                continue
            
            company_info = {
                "facts": {},
                "about": "",
                "skills_required": "",
                "interview_process": ""
            }
            
            # Extract facts
            if facts:
                f = facts[0]
                company_info["facts"] = {
                    "role": f.get("role_title") or f.get("role_name"),
                    "stipend": f.get("stipend_salary"),
                    "location": f.get("location"),
                    "eligibility": f.get("eligibility"),
                    "selection_process": f.get("selection_process")
                }
            
            # Extract semantic info
            for chunk in chunks:
                chunk_type = chunk.get("type", "")
                text = chunk.get("text", "")
                
                if chunk_type == "about_company" and not company_info["about"]:
                    company_info["about"] = text[:500]
                elif chunk_type == "skills_required" and not company_info["skills_required"]:
                    company_info["skills_required"] = text[:500]
                elif chunk_type == "interview_process" and not company_info["interview_process"]:
                    company_info["interview_process"] = text[:500]
            
            detailed[company] = company_info
        
        return ToolResult(
            success=True,
            data={
                "detailed_comparison": detailed,
                "companies": list(detailed.keys())
            },
            message=f"Generated detailed comparison for {len(detailed)} companies",
            tool_name=self.name,
            query=f"detailed:{','.join(companies)}"
        )
    
    def _compare_ranking(self, companies: List[str], rank_by: str = "stipend") -> ToolResult:
        """Rank companies by specific attribute."""
        
        rankings = []
        
        for company in companies:
            facts = self.facts_index.get_by_company(company)
            if not facts:
                continue
            
            fact = facts[0]
            value = None
            
            if rank_by == "stipend":
                stipend = fact.get("stipend_salary", {})
                if isinstance(stipend, dict):
                    value = self._parse_number(stipend.get("amount"))
                else:
                    value = self._parse_number(stipend)
            
            elif rank_by == "cgpa":
                elig = fact.get("eligibility", {})
                if isinstance(elig, dict):
                    value = self._parse_number(elig.get("cgpa_pg") or elig.get("cgpa_ug"))
            
            elif rank_by == "rounds" or rank_by == "num_rounds":
                process = fact.get("selection_process", [])
                value = len(process) if isinstance(process, list) else None
            
            if value is not None:
                rankings.append({
                    "company": company,
                    "role": fact.get("role_title") or fact.get("role_name"),
                    rank_by: value
                })
        
        # Sort (descending for stipend, ascending for cgpa/rounds)
        reverse = rank_by == "stipend"
        rankings.sort(key=lambda x: x.get(rank_by, 0) or 0, reverse=reverse)
        
        # Add rank
        for i, r in enumerate(rankings, 1):
            r["rank"] = i
        
        return ToolResult(
            success=True,
            data={
                "rankings": rankings,
                "ranked_by": rank_by,
                "order": "descending" if reverse else "ascending"
            },
            message=f"Ranked {len(rankings)} companies by {rank_by}",
            tool_name=self.name,
            query=f"rank:{rank_by}"
        )
    
    def _find_best(self, companies: List[str], criteria: List[str] = None) -> ToolResult:
        """Find best company based on criteria."""
        
        if criteria is None:
            criteria = ["stipend"]
        
        scores = {}
        
        for company in companies:
            facts = self.facts_index.get_by_company(company)
            if not facts:
                continue
            
            fact = facts[0]
            score = 0
            details = {}
            
            for criterion in criteria:
                if criterion == "stipend":
                    stipend = fact.get("stipend_salary", {})
                    if isinstance(stipend, dict):
                        val = self._parse_number(stipend.get("amount"))
                    else:
                        val = self._parse_number(stipend)
                    
                    if val:
                        score += val / 10000  # Normalize
                        details["stipend"] = val
                
                elif criterion == "low_cgpa":
                    elig = fact.get("eligibility", {})
                    if isinstance(elig, dict):
                        cgpa = self._parse_number(elig.get("cgpa_pg") or elig.get("cgpa_ug"))
                        if cgpa:
                            score += (10 - cgpa)  # Lower is better
                            details["cgpa"] = cgpa
                
                elif criterion == "few_rounds":
                    process = fact.get("selection_process", [])
                    rounds = len(process) if isinstance(process, list) else 0
                    if rounds > 0:
                        score += (10 - rounds)  # Fewer is better
                        details["rounds"] = rounds
            
            scores[company] = {"score": score, "details": details}
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        best = ranked[0] if ranked else None
        
        return ToolResult(
            success=True,
            data={
                "best_company": best[0] if best else None,
                "best_score": best[1] if best else None,
                "all_scores": dict(ranked),
                "criteria": criteria
            },
            message=f"Best company: {best[0] if best else 'N/A'} based on {', '.join(criteria)}",
            tool_name=self.name,
            query=f"best_for:{','.join(criteria)}"
        )
    
    def _parse_number(self, value: Any) -> Optional[float]:
        """Parse numeric value."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        
        import re
        numbers = re.findall(r'[\d.]+', str(value).replace(',', ''))
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        return None
