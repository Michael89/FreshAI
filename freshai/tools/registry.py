"""Tool registry for managing investigation tools."""

import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all investigation tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool parameter schema."""
        pass


class ToolRegistry:
    """Registry for managing investigation tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool_name: str, tool: BaseTool) -> None:
        """Register a new tool."""
        self.tools[tool_name] = tool
        logger.info(f"Registered tool: {tool_name}")
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
    
    async def use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Use a registered tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found. Available tools: {list(self.tools.keys())}")
        
        tool = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name}")
        
        try:
            result = await tool.execute(parameters)
            return {
                "tool_name": tool_name,
                "status": "success",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "tool_name": tool_name,
                "status": "error",
                "error": str(e)
            }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.get_schema()
        }
    
    def get_all_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered tools."""
        return {
            name: self.get_tool_info(name) 
            for name in self.tools.keys()
        }