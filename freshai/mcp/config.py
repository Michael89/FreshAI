"""MCP configuration models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    
    name: str = Field(description="Name of the MCP server")
    command: str = Field(description="Command to start the MCP server")
    arguments: List[str] = Field(default_factory=list, description="Arguments for the server command")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_directory: Optional[Path] = Field(default=None, description="Working directory for server")
    enabled: bool = Field(default=True, description="Whether this server is enabled")
    timeout: int = Field(default=30, description="Connection timeout in seconds")


class MCPConfig(BaseModel):
    """Configuration for MCP servers."""
    
    enable_mcp: bool = Field(default=True, description="Enable MCP server support")
    servers: Dict[str, MCPServerConfig] = Field(default_factory=dict, description="MCP server configurations")
    default_timeout: int = Field(default=30, description="Default timeout for MCP operations")
    max_concurrent_servers: int = Field(default=5, description="Maximum concurrent MCP servers")
    
    @classmethod
    def load_from_env(cls) -> "MCPConfig":
        """Load MCP configuration from environment variables."""
        enable_mcp = os.getenv("FRESHAI_ENABLE_MCP", "true").lower() == "true"
        default_timeout = int(os.getenv("FRESHAI_MCP_DEFAULT_TIMEOUT", "30"))
        max_concurrent = int(os.getenv("FRESHAI_MCP_MAX_CONCURRENT_SERVERS", "5"))
        
        # Load default servers from environment
        servers = {}
        
        # Add built-in filesystem server if enabled
        if os.getenv("FRESHAI_MCP_FILESYSTEM_ENABLED", "true").lower() == "true":
            servers["filesystem"] = MCPServerConfig(
                name="filesystem",
                command="python",
                arguments=["-m", "mcp_servers.filesystem"],
                enabled=True
            )
        
        # Add built-in web search server if enabled
        if os.getenv("FRESHAI_MCP_WEB_SEARCH_ENABLED", "false").lower() == "true":
            servers["web_search"] = MCPServerConfig(
                name="web_search",
                command="python",
                arguments=["-m", "mcp_servers.web_search"],
                enabled=True
            )
        
        return cls(
            enable_mcp=enable_mcp,
            servers=servers,
            default_timeout=default_timeout,
            max_concurrent_servers=max_concurrent
        )