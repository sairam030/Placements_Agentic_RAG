"""Facts Lookup Tool for structured attribute queries."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
from tools.base_tool import BaseTool, ToolResult
from rag.facts_index import FactsIndex


class FactsLookupTool(BaseTool):
    """
    Tool for querying structured placement facts.
    
    Use this tool for:
    - Getting specific attribute values (stipend, CGPA, location, duration)
    - Filtering companies by criteria (stipend > X, location = Y)
    - Getting eligibility requirements
    - Getting selection process details
    - Listing all companies or roles
    """
    
    name = "facts_lookup"
    description = """Lookup structured placement facts like stipend, eligibility criteria, 
    location, duration, selection process rounds, CGPA requirements, and apply deadlines. 
    Use for exact value queries and filtering by numeric/categorical criteria."""
    
    def __init__(self):
        self.index = FactsIndex()
        self._loaded = False
    
    def _ensure_loaded(self):
        """Ensure index is loaded."""
        if not self._loaded:
            self._loaded = self.index.load()
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_all_companies",
                        "get_company_details", 
                        "get_all_stipends",
                        "filter_by_stipend",
                        "filter_by_cgpa",
                        "filter_by_location",
                        "filter_by_branch",
                        "get_attribute",
                        "get_eligibility",
                        "get_selection_process"
                    ],
                    "description": "The action to perform"
                },
                "company": {
                    "type": "string",
                    "description": "Company name to query"
                },
                "attribute": {
                    "type": "string",
                    "description": "Attribute to retrieve (stipend, location, cgpa, etc.)"
                },
                "min_value": {
                    "type": "number",
                    "description": "Minimum value for filtering"
                },
                "max_value": {
                    "type": "number",
                    "description": "Maximum value for filtering"
                },
                "location": {
                    "type": "string",
                    "description": "Location to filter by"
                },
                "branch": {
                    "type": "string",
                    "description": "Branch/specialization to filter by"
                }
            },
            "required": ["action"]
        }
    
    def get_capabilities(self) -> List[str]:
        return [
            "Get list of all companies",
            "Get company details (all facts)",
            "Get stipend for specific company or all companies",
            "Filter companies by stipend range",
            "Filter companies by CGPA requirement",
            "Filter companies by location",
            "Filter companies by eligible branches",
            "Get eligibility criteria for a company",
            "Get selection process for a company",
            "Get any specific attribute across companies"
        ]
    
    def execute(
        self,
        action: str,
        company: str = None,
        attribute: str = None,
        min_value: float = None,
        max_value: float = None,
        location: str = None,
        branch: str = None,
        **kwargs
    ) -> ToolResult:
        """Execute a facts lookup query."""
        
        self._ensure_loaded()
        
        if not self._loaded:
            return ToolResult(
                success=False,
                data=None,
                message="Failed to load facts index",
                tool_name=self.name,
                query=action
            )
        
        try:
            # Route to appropriate method
            if action == "get_all_companies":
                return self._get_all_companies()
            
            elif action == "get_company_details":
                return self._get_company_details(company)
            
            elif action == "get_all_stipends":
                return self._get_all_stipends()
            
            elif action == "filter_by_stipend":
                return self._filter_by_stipend(min_value, max_value)
            
            elif action == "filter_by_cgpa":
                return self._filter_by_cgpa(max_value)
            
            elif action == "filter_by_location":
                return self._filter_by_location(location)
            
            elif action == "filter_by_branch":
                return self._filter_by_branch(branch)
            
            elif action == "get_attribute":
                return self._get_attribute(attribute, company)
            
            elif action == "get_eligibility":
                return self._get_eligibility(company)
            
            elif action == "get_selection_process":
                return self._get_selection_process(company)
            
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Unknown action: {action}",
                    tool_name=self.name,
                    query=action
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Error: {str(e)}",
                tool_name=self.name,
                query=action
            )
    
    def _get_all_companies(self) -> ToolResult:
        """Get list of all companies."""
        companies = self.index.get_all_companies()
        return ToolResult(
            success=True,
            data={"companies": companies, "count": len(companies)},
            message=f"Found {len(companies)} companies",
            tool_name=self.name,
            query="get_all_companies"
        )
    
    def _get_company_details(self, company: str) -> ToolResult:
        """Get all facts for a company."""
        if not company:
            return ToolResult(
                success=False,
                data=None,
                message="Company name required",
                tool_name=self.name,
                query="get_company_details"
            )
        
        facts = self.index.get_by_company(company)
        if not facts:
            return ToolResult(
                success=False,
                data=None,
                message=f"No data found for company: {company}",
                tool_name=self.name,
                query=f"get_company_details:{company}"
            )
        
        # Format the output
        formatted = []
        for f in facts:
            formatted.append({
                "company": f.get("company_name"),
                "role": f.get("role_title") or f.get("role_name"),
                "stipend": f.get("stipend_salary"),
                "duration": f.get("duration"),
                "location": f.get("location"),
                "work_mode": f.get("work_mode"),
                "eligibility": f.get("eligibility"),
                "selection_process": f.get("selection_process"),
                "apply_before": f.get("apply_before")
            })
        
        return ToolResult(
            success=True,
            data={"company": company, "roles": formatted, "count": len(formatted)},
            message=f"Found {len(formatted)} roles for {company}",
            tool_name=self.name,
            query=f"get_company_details:{company}"
        )
    
    def _get_all_stipends(self) -> ToolResult:
        """Get stipend info for all companies."""
        stipends = self.index.get_all_stipends()
        
        # Format nicely
        formatted = []
        for s in stipends:
            stipend_val = s.get("stipend", "N/A")
            if isinstance(stipend_val, dict):
                amount = stipend_val.get("amount", "N/A")
                currency = stipend_val.get("currency", "INR")
                period = stipend_val.get("period", "per month")
                stipend_str = f"{amount} {currency} {period}"
            else:
                stipend_str = str(stipend_val) if stipend_val else "N/A"
            
            formatted.append({
                "company": s.get("company"),
                "role": s.get("role_title") or s.get("role"),
                "stipend": stipend_str
            })
        
        return ToolResult(
            success=True,
            data={"stipends": formatted, "count": len(formatted)},
            message=f"Retrieved stipend info for {len(formatted)} roles",
            tool_name=self.name,
            query="get_all_stipends"
        )
    
    def _filter_by_stipend(self, min_val: float = None, max_val: float = None) -> ToolResult:
        """Filter companies by stipend range."""
        facts = self.index.filter_by_stipend(min_amount=min_val, max_amount=max_val)
        
        formatted = []
        for f in facts:
            stipend = f.get("stipend_salary", {})
            if isinstance(stipend, dict):
                amount = stipend.get("amount", "N/A")
            else:
                amount = stipend
            
            formatted.append({
                "company": f.get("company_name"),
                "role": f.get("role_title") or f.get("role_name"),
                "stipend": amount,
                "location": f.get("location")
            })
        
        criteria = []
        if min_val:
            criteria.append(f"min={min_val}")
        if max_val:
            criteria.append(f"max={max_val}")
        
        return ToolResult(
            success=True,
            data={"results": formatted, "count": len(formatted)},
            message=f"Found {len(formatted)} roles with stipend {', '.join(criteria)}",
            tool_name=self.name,
            query=f"filter_by_stipend:{','.join(criteria)}"
        )
    
    def _filter_by_cgpa(self, max_cgpa: float) -> ToolResult:
        """Filter companies by maximum CGPA requirement."""
        if max_cgpa is None:
            return ToolResult(
                success=False,
                data=None,
                message="max_value (CGPA) required",
                tool_name=self.name,
                query="filter_by_cgpa"
            )
        
        facts = self.index.filter_by_cgpa(max_cgpa_required=max_cgpa)
        
        formatted = []
        for f in facts:
            elig = f.get("eligibility", {})
            cgpa = elig.get("cgpa_pg") or elig.get("cgpa_ug") if isinstance(elig, dict) else None
            
            formatted.append({
                "company": f.get("company_name"),
                "role": f.get("role_title") or f.get("role_name"),
                "cgpa_required": cgpa,
                "stipend": f.get("stipend_salary")
            })
        
        return ToolResult(
            success=True,
            data={"results": formatted, "count": len(formatted)},
            message=f"Found {len(formatted)} roles with CGPA requirement <= {max_cgpa}",
            tool_name=self.name,
            query=f"filter_by_cgpa:<={max_cgpa}"
        )
    
    def _filter_by_location(self, location: str) -> ToolResult:
        """Filter companies by location."""
        if not location:
            return ToolResult(
                success=False,
                data=None,
                message="Location required",
                tool_name=self.name,
                query="filter_by_location"
            )
        
        facts = self.index.filter_by_location(location)
        
        formatted = []
        for f in facts:
            formatted.append({
                "company": f.get("company_name"),
                "role": f.get("role_title") or f.get("role_name"),
                "location": f.get("location"),
                "stipend": f.get("stipend_salary")
            })
        
        return ToolResult(
            success=True,
            data={"results": formatted, "count": len(formatted), "location": location},
            message=f"Found {len(formatted)} roles in {location}",
            tool_name=self.name,
            query=f"filter_by_location:{location}"
        )
    
    def _filter_by_branch(self, branch: str) -> ToolResult:
        """Filter companies by eligible branch."""
        if not branch:
            return ToolResult(
                success=False,
                data=None,
                message="Branch required",
                tool_name=self.name,
                query="filter_by_branch"
            )
        
        facts = self.index.filter_by_branch(branch)
        
        formatted = []
        for f in facts:
            elig = f.get("eligibility", {})
            branches = elig.get("branches", []) if isinstance(elig, dict) else []
            
            formatted.append({
                "company": f.get("company_name"),
                "role": f.get("role_title") or f.get("role_name"),
                "eligible_branches": branches,
                "stipend": f.get("stipend_salary")
            })
        
        return ToolResult(
            success=True,
            data={"results": formatted, "count": len(formatted), "branch": branch},
            message=f"Found {len(formatted)} roles for {branch} students",
            tool_name=self.name,
            query=f"filter_by_branch:{branch}"
        )
    
    def _get_attribute(self, attribute: str, company: str = None) -> ToolResult:
        """Get specific attribute for company/all companies."""
        if not attribute:
            return ToolResult(
                success=False,
                data=None,
                message="Attribute name required",
                tool_name=self.name,
                query="get_attribute"
            )
        
        companies = [company] if company else None
        results = self.index.search_attribute(attribute, companies=companies)
        
        return ToolResult(
            success=True,
            data={"attribute": attribute, "results": results, "count": len(results)},
            message=f"Retrieved '{attribute}' for {len(results)} roles",
            tool_name=self.name,
            query=f"get_attribute:{attribute}"
        )
    
    def _get_eligibility(self, company: str = None) -> ToolResult:
        """Get eligibility criteria."""
        if company:
            facts = self.index.get_by_company(company)
        else:
            facts = self.index.facts
        
        formatted = []
        for f in facts:
            elig = f.get("eligibility", {})
            formatted.append({
                "company": f.get("company_name"),
                "role": f.get("role_title") or f.get("role_name"),
                "eligibility": elig
            })
        
        return ToolResult(
            success=True,
            data={"results": formatted, "count": len(formatted)},
            message=f"Retrieved eligibility for {len(formatted)} roles",
            tool_name=self.name,
            query=f"get_eligibility:{company or 'all'}"
        )
    
    def _get_selection_process(self, company: str = None) -> ToolResult:
        """Get selection process details."""
        if company:
            facts = self.index.get_by_company(company)
        else:
            facts = self.index.facts
        
        formatted = []
        for f in facts:
            process = f.get("selection_process", [])
            formatted.append({
                "company": f.get("company_name"),
                "role": f.get("role_title") or f.get("role_name"),
                "selection_process": process,
                "num_rounds": len(process) if isinstance(process, list) else 0
            })
        
        return ToolResult(
            success=True,
            data={"results": formatted, "count": len(formatted)},
            message=f"Retrieved selection process for {len(formatted)} roles",
            tool_name=self.name,
            query=f"get_selection_process:{company or 'all'}"
        )
