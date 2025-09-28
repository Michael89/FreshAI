"""MCP client implementation for connecting to MCP servers."""

import asyncio
import json
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional, AsyncIterator
from pathlib import Path

from ..tools.registry import BaseTool
from .config import MCPServerConfig


logger = logging.getLogger(__name__)


class MCPServerProcess:
    """Manages a single MCP server process."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.connected = False
        
    async def start(self) -> bool:
        """Start the MCP server process."""
        try:
            # Build command
            cmd = [self.config.command] + self.config.arguments
            
            # Set up environment
            env = dict(os.environ) if 'os' in globals() else {}
            env.update(self.config.environment)
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=self.config.working_directory
            )
            
            # Wait a moment to check if process started successfully
            await asyncio.sleep(0.1)
            
            if self.process.poll() is None:
                self.connected = True
                logger.info(f"Started MCP server: {self.config.name}")
                return True
            else:
                logger.error(f"Failed to start MCP server: {self.config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting MCP server {self.config.name}: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
                self.connected = False
                logger.info(f"Stopped MCP server: {self.config.name}")
            except Exception as e:
                logger.error(f"Error stopping MCP server {self.config.name}: {e}")
    
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server and get response."""
        if not self.connected or not self.process:
            raise RuntimeError(f"MCP server {self.config.name} is not connected")
        
        try:
            # Send request
            request_json = json.dumps(request) + '\n'
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response (simplified - in real MCP we'd need proper JSON-RPC handling)
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                raise RuntimeError("No response from MCP server")
                
        except Exception as e:
            logger.error(f"Error communicating with MCP server {self.config.name}: {e}")
            raise


class MCPTool(BaseTool):
    """Wrapper for MCP server tools to integrate with FreshAI tool registry."""
    
    def __init__(self, server_name: str, tool_name: str, tool_schema: Dict[str, Any], server_process: MCPServerProcess):
        self.server_name = server_name
        self.tool_name = tool_name
        self.tool_schema = tool_schema
        self.server_process = server_process
        
        # Extract description from schema
        description = tool_schema.get("description", f"MCP tool {tool_name} from {server_name}")
        
        super().__init__(f"{server_name}_{tool_name}", description)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MCP tool."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": self.tool_name,
                "arguments": parameters
            }
        }
        
        try:
            response = await self.server_process.send_request(request)
            
            # Extract result from JSON-RPC response
            if "result" in response:
                return {
                    "server": self.server_name,
                    "tool": self.tool_name,
                    "success": True,
                    "data": response["result"]
                }
            elif "error" in response:
                return {
                    "server": self.server_name,
                    "tool": self.tool_name,
                    "success": False,
                    "error": response["error"]
                }
            else:
                return {
                    "server": self.server_name,
                    "tool": self.tool_name,
                    "success": False,
                    "error": "Invalid response format"
                }
                
        except Exception as e:
            logger.error(f"Error executing MCP tool {self.name}: {e}")
            return {
                "server": self.server_name,
                "tool": self.tool_name,
                "success": False,
                "error": str(e)
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool parameter schema."""
        return self.tool_schema


class MCPClient:
    """Client for managing MCP servers and their tools."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerProcess] = {}
        self.tools: Dict[str, MCPTool] = {}
        
    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and start an MCP server."""
        if not config.enabled:
            logger.info(f"MCP server {config.name} is disabled")
            return False
            
        try:
            server_process = MCPServerProcess(config)
            success = await server_process.start()
            
            if success:
                self.servers[config.name] = server_process
                await self._discover_tools(config.name, server_process)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to add MCP server {config.name}: {e}")
            return False
    
    async def _discover_tools(self, server_name: str, server_process: MCPServerProcess) -> None:
        """Discover available tools from an MCP server."""
        try:
            # Request tools list (simplified JSON-RPC)
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            response = await server_process.send_request(request)
            
            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                
                for tool_info in tools:
                    tool_name = tool_info["name"]
                    tool_schema = tool_info.get("inputSchema", {})
                    
                    mcp_tool = MCPTool(server_name, tool_name, tool_schema, server_process)
                    tool_key = f"{server_name}_{tool_name}"
                    self.tools[tool_key] = mcp_tool
                    
                    logger.info(f"Discovered MCP tool: {tool_key}")
            
        except Exception as e:
            logger.error(f"Failed to discover tools from {server_name}: {e}")
    
    async def stop_all_servers(self) -> None:
        """Stop all MCP servers."""
        for server_name, server_process in self.servers.items():
            await server_process.stop()
        
        self.servers.clear()
        self.tools.clear()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        return list(self.tools.keys())
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get an MCP tool by name."""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, MCPTool]:
        """Get all MCP tools."""
        return self.tools.copy()