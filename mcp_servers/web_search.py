#!/usr/bin/env python3
"""
Simple MCP server that provides web search operations.
This is a basic implementation using requests for demonstration purposes.
"""

import json
import sys
import re
from typing import Dict, Any, List
from urllib.parse import quote_plus


class WebSearchMCPServer:
    """Simple MCP server for web search operations."""
    
    def __init__(self):
        self.tools = {
            "search_web": {
                "name": "search_web",
                "description": "Search the web for information (simplified version)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            "extract_urls": {
                "name": "extract_urls",
                "description": "Extract URLs from text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract URLs from"
                        }
                    },
                    "required": ["text"]
                }
            },
            "validate_url": {
                "name": "validate_url",
                "description": "Validate if a URL is properly formatted",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to validate"
                        }
                    },
                    "required": ["url"]
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
            if tool_name == "search_web":
                result = self._search_web(arguments)
            elif tool_name == "extract_urls":
                result = self._extract_urls(arguments)
            elif tool_name == "validate_url":
                result = self._validate_url(arguments)
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
    
    def _search_web(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Search the web (simplified mock implementation)."""
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        
        # This is a simplified mock implementation
        # In a real implementation, you would use a search API like Google, Bing, etc.
        mock_results = [
            {
                "title": f"Result 1 for '{query}'",
                "url": f"https://example.com/result1?q={quote_plus(query)}",
                "snippet": f"This is a mock search result for the query '{query}'. In a real implementation, this would contain actual search results from a search engine API.",
                "domain": "example.com"
            },
            {
                "title": f"Result 2 for '{query}'",
                "url": f"https://example.org/result2?search={quote_plus(query)}",
                "snippet": f"Another mock result that demonstrates how search results would be formatted and returned for the query '{query}'.",
                "domain": "example.org"
            },
            {
                "title": f"Result 3 for '{query}'",
                "url": f"https://demo.com/page?q={quote_plus(query)}",
                "snippet": f"A third example result showing various aspects of search functionality for '{query}'. Real implementations would fetch from actual search engines.",
                "domain": "demo.com"
            }
        ]
        
        # Limit results
        results = mock_results[:max_results]
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "note": "This is a mock implementation. In production, integrate with real search APIs like Google Custom Search, Bing Search API, etc."
        }
    
    def _extract_urls(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract URLs from text."""
        text = arguments["text"]
        
        # Regular expression to find URLs
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        urls = url_pattern.findall(text)
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        return {
            "text_length": len(text),
            "urls_found": unique_urls,
            "total_urls": len(unique_urls),
            "unique_domains": list(set(
                url.split('/')[2] for url in unique_urls if '/' in url
            ))
        }
    
    def _validate_url(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate URL format."""
        url = arguments["url"]
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        is_valid = bool(url_pattern.match(url))
        
        analysis = {
            "url": url,
            "is_valid": is_valid,
            "protocol": None,
            "domain": None,
            "path": None,
            "has_query": False,
            "has_fragment": False
        }
        
        if is_valid:
            parts = url.split('://', 1)
            if len(parts) == 2:
                analysis["protocol"] = parts[0]
                remaining = parts[1]
                
                # Extract domain
                domain_path = remaining.split('/', 1)
                analysis["domain"] = domain_path[0]
                
                if len(domain_path) > 1:
                    analysis["path"] = '/' + domain_path[1]
                
                # Check for query and fragment
                analysis["has_query"] = '?' in url
                analysis["has_fragment"] = '#' in url
        
        return analysis
    
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
    server = WebSearchMCPServer()
    server.run()