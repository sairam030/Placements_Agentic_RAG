"""Semantic RAG Tool for contextual/descriptive queries."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
from tools.base_tool import BaseTool, ToolResult
from rag.semantic_index import SemanticIndex


class SemanticRAGTool(BaseTool):
    """
    Tool for semantic search over placement document chunks.
    
    Use this tool for:
    - Company descriptions and culture
    - Job roles and responsibilities
    - Required and optional skills
    - Interview process details
    - Any descriptive/contextual information
    """
    
    name = "semantic_search"
    description = """Search for descriptive information about companies, roles, 
    skills required, interview processes, company culture, responsibilities, etc. 
    Use for contextual queries that need understanding of text content."""
    
    def __init__(self):
        self.index = SemanticIndex()
        self._loaded = False
    
    def _ensure_loaded(self):
        """Ensure index is loaded."""
        if not self._loaded:
            self._loaded = self.index.load()
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "search_type": {
                    "type": "string",
                    "enum": [
                        "general",
                        "about_company",
                        "roles_responsibilities",
                        "skills_required",
                        "skills_optional",
                        "interview_process",
                        "eligibility_criteria",
                        "compensation_benefits",
                        "additional_info"
                    ],
                    "description": "Type of information to search for"
                },
                "company": {
                    "type": "string",
                    "description": "Filter results by company name"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    def get_capabilities(self) -> List[str]:
        return [
            "Search for company information and culture",
            "Find job roles and responsibilities",
            "Search for required skills for positions",
            "Find optional/preferred skills",
            "Get interview process details",
            "Search eligibility criteria descriptions",
            "Find compensation and benefits info",
            "General semantic search across all content",
            "Filter search by specific company",
            "Filter search by content type"
        ]
    
    def execute(
        self,
        query: str,
        search_type: str = "general",
        company: str = None,
        top_k: int = 5,
        **kwargs
    ) -> ToolResult:
        """Execute a semantic search query."""
        
        self._ensure_loaded()
        
        if not self._loaded:
            return ToolResult(
                success=False,
                data=None,
                message="Failed to load semantic index",
                tool_name=self.name,
                query=query
            )
        
        if not query:
            return ToolResult(
                success=False,
                data=None,
                message="Query is required",
                tool_name=self.name,
                query=""
            )
        
        try:
            # Determine filter type
            type_filter = None if search_type == "general" else search_type
            
            # Perform search
            results = self.index.search(
                query=query,
                top_k=top_k,
                filter_company=company,
                filter_type=type_filter,
                threshold=0.2
            )
            
            if not results:
                return ToolResult(
                    success=True,
                    data={"results": [], "count": 0},
                    message=f"No relevant results found for: {query}",
                    tool_name=self.name,
                    query=query
                )
            
            # Format results
            formatted = []
            for r in results:
                formatted.append({
                    "company": r.get("company"),
                    "role": r.get("role"),
                    "type": r.get("type"),
                    "content": r.get("text"),
                    "relevance_score": round(r.get("score", 0), 4),
                    "source": r.get("source")
                })
            
            # Build context string for easy consumption
            context_parts = []
            for r in formatted:
                context_parts.append(
                    f"[{r['company']} - {r['role']}] ({r['type']})\n{r['content']}"
                )
            context = "\n\n---\n\n".join(context_parts)
            
            return ToolResult(
                success=True,
                data={
                    "results": formatted,
                    "count": len(formatted),
                    "context": context,
                    "query": query,
                    "filters": {
                        "type": search_type,
                        "company": company
                    }
                },
                message=f"Found {len(formatted)} relevant results",
                tool_name=self.name,
                query=query
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Search error: {str(e)}",
                tool_name=self.name,
                query=query
            )
    
    def search_skills(self, query: str, company: str = None, top_k: int = 5) -> ToolResult:
        """Convenience method to search for skills."""
        # Search both required and optional skills
        required = self.execute(
            query=query,
            search_type="skills_required",
            company=company,
            top_k=top_k
        )
        
        optional = self.execute(
            query=query,
            search_type="skills_optional",
            company=company,
            top_k=top_k
        )
        
        # Combine results
        all_results = []
        if required.success and required.data:
            all_results.extend(required.data.get("results", []))
        if optional.success and optional.data:
            all_results.extend(optional.data.get("results", []))
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return ToolResult(
            success=True,
            data={
                "results": all_results[:top_k],
                "count": len(all_results[:top_k]),
                "query": query
            },
            message=f"Found {len(all_results[:top_k])} skill-related results",
            tool_name=self.name,
            query=f"skills:{query}"
        )
    
    def search_interview_process(self, company: str = None, top_k: int = 5) -> ToolResult:
        """Convenience method to search interview processes."""
        query = "interview selection process rounds technical HR"
        return self.execute(
            query=query,
            search_type="interview_process",
            company=company,
            top_k=top_k
        )
    
    def search_company_info(self, company: str, top_k: int = 5) -> ToolResult:
        """Convenience method to get all info about a company."""
        query = f"{company} company culture work environment products services"
        return self.execute(
            query=query,
            search_type="about_company",
            company=company,
            top_k=top_k
        )
    
    def get_all_chunks_for_company(self, company: str) -> ToolResult:
        """Get all semantic chunks for a specific company."""
        self._ensure_loaded()
        
        chunks = self.index.get_all_by_company(company)
        
        if not chunks:
            return ToolResult(
                success=False,
                data=None,
                message=f"No information found for company: {company}",
                tool_name=self.name,
                query=f"all_chunks:{company}"
            )
        
        # Group by type
        by_type = {}
        for chunk in chunks:
            t = chunk.get("type", "other")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append({
                "role": chunk.get("role"),
                "content": chunk.get("text"),
                "source": chunk.get("source")
            })
        
        return ToolResult(
            success=True,
            data={
                "company": company,
                "chunks_by_type": by_type,
                "total_chunks": len(chunks)
            },
            message=f"Found {len(chunks)} chunks for {company}",
            tool_name=self.name,
            query=f"all_chunks:{company}"
        )
