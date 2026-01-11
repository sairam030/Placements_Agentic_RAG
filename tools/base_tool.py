"""Base class for all tools."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any
    message: str
    tool_name: str
    query: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "tool_name": self.tool_name,
            "query": self.query
        }
    
    def __str__(self) -> str:
        if self.success:
            return f"[{self.tool_name}] ✅ {self.message}\nData: {self.data}"
        return f"[{self.tool_name}] ❌ {self.message}"


class BaseTool(ABC):
    """Base class for all RAG tools."""
    
    name: str = "base_tool"
    description: str = "Base tool"
    
    @abstractmethod
    def execute(self, query: str, **kwargs) -> ToolResult:
        """Execute the tool with given query."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this tool provides."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }
    
    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema."""
        pass
