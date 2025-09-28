"""Model Context Protocol (MCP) support for FreshAI."""

from .client import MCPClient, MCPTool
from .config import MCPConfig

__all__ = ["MCPClient", "MCPTool", "MCPConfig"]