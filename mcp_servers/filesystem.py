#!/usr/bin/env python3
"""
Simple MCP server that provides filesystem operations.
This is a basic implementation for demonstration purposes.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List


class FilesystemMCPServer:
    """Simple MCP server for filesystem operations."""
    
    def __init__(self):
        self.tools = {
            "list_files": {
                "name": "list_files",
                "description": "List files and directories in a given path",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to list files from"
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": "Include hidden files",
                            "default": False
                        }
                    },
                    "required": ["path"]
                }
            },
            "read_file": {
                "name": "read_file",
                "description": "Read contents of a text file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding",
                            "default": "utf-8"
                        }
                    },
                    "required": ["path"]
                }
            },
            "get_file_info": {
                "name": "get_file_info",
                "description": "Get information about a file or directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to get information about"
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    
    def handle_tools_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": list(self.tools.values())
            }
        }
    
    def handle_tools_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "list_files":
                result = self._list_files(arguments)
            elif tool_name == "read_file":
                result = self._read_file(arguments)
            elif tool_name == "get_file_info":
                result = self._get_file_info(arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                }
            }
    
    def _list_files(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List files in a directory."""
        path = Path(arguments["path"])
        include_hidden = arguments.get("include_hidden", False)
        
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        files = []
        directories = []
        
        for item in path.iterdir():
            if not include_hidden and item.name.startswith('.'):
                continue
            
            item_info = {
                "name": item.name,
                "path": str(item),
                "size": item.stat().st_size if item.is_file() else None,
                "modified": item.stat().st_mtime
            }
            
            if item.is_file():
                files.append(item_info)
            elif item.is_dir():
                directories.append(item_info)
        
        return {
            "path": str(path),
            "files": files,
            "directories": directories,
            "total_files": len(files),
            "total_directories": len(directories)
        }
    
    def _read_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Read contents of a file."""
        path = Path(arguments["path"])
        encoding = arguments.get("encoding", "utf-8")
        
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        try:
            content = path.read_text(encoding=encoding)
            return {
                "path": str(path),
                "content": content,
                "encoding": encoding,
                "size": len(content),
                "lines": len(content.splitlines())
            }
        except UnicodeDecodeError as e:
            raise ValueError(f"Cannot decode file with encoding {encoding}: {e}")
    
    def _get_file_info(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about a file or directory."""
        path = Path(arguments["path"])
        
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        stat = path.stat()
        
        info = {
            "path": str(path),
            "name": path.name,
            "type": "file" if path.is_file() else "directory",
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:],
            "is_readable": os.access(path, os.R_OK),
            "is_writable": os.access(path, os.W_OK),
            "is_executable": os.access(path, os.X_OK)
        }
        
        if path.is_file():
            # Add file-specific info
            info["extension"] = path.suffix
            try:
                # Try to detect if it's a text file
                with open(path, 'rb') as f:
                    sample = f.read(1024)
                    info["is_text"] = all(byte < 128 or byte in [10, 13] for byte in sample)
            except:
                info["is_text"] = False
        
        return info
    
    def run(self):
        """Run the MCP server."""
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    method = request.get("method")
                    
                    if method == "tools/list":
                        response = self.handle_tools_list(request)
                    elif method == "tools/call":
                        response = self.handle_tools_call(request)
                    else:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32601,
                                "message": f"Unknown method: {method}"
                            }
                        }
                    
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {e}"
                        }
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)


if __name__ == "__main__":
    server = FilesystemMCPServer()
    server.run()