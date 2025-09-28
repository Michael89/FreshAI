#!/usr/bin/env python3
"""
Example demonstrating MCP server integration with FreshAI.

This example shows how to:
1. Configure MCP servers
2. Start and connect to MCP servers
3. Use MCP tools through FreshAI's tool registry
4. Handle MCP server lifecycle
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from freshai.config import Config
from freshai.core import FreshAICore
from freshai.mcp import MCPClient, MCPConfig
from freshai.mcp.config import MCPServerConfig


async def demo_filesystem_mcp_server():
    """Demonstrate filesystem MCP server integration."""
    print("=== Filesystem MCP Server Demo ===")
    
    # Create a temporary directory with sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Create sample files
        sample_files = {
            "case_notes.txt": "Investigation notes for Case #001\nSuspect: John Doe",
            "evidence_list.txt": "Evidence items:\n1. Digital camera\n2. USB drive\n3. Phone records",
            "interview_transcript.txt": "Interview with witness on 2024-01-15..."
        }
        
        for filename, content in sample_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            print(f"  Created: {filename}")
        
        # Create subdirectory
        subdir = Path(temp_dir) / "evidence_photos"
        subdir.mkdir()
        (subdir / "photo1.txt").write_text("Mock photo file")
        print(f"  Created subdirectory with files")
        
        # Configure MCP filesystem server
        fs_config = MCPServerConfig(
            name="filesystem",
            command="python",
            arguments=["-m", "mcp_servers.filesystem"],
            enabled=True
        )
        
        # Create MCP client and add server
        mcp_client = MCPClient()
        
        print("\nStarting filesystem MCP server...")
        success = await mcp_client.add_server(fs_config)
        
        if success:
            print("✓ Filesystem server started successfully")
            
            # List available tools
            tools = mcp_client.get_available_tools()
            print(f"Available tools: {tools}")
            
            # Test filesystem tools
            if "filesystem_list_files" in tools:
                list_tool = mcp_client.get_tool("filesystem_list_files")
                if list_tool:
                    print(f"\nListing files in {temp_dir}:")
                    result = await list_tool.execute({"path": temp_dir})
                    if result["success"]:
                        data = result["data"]
                        print(f"  Files: {len(data['files'])}")
                        print(f"  Directories: {len(data['directories'])}")
                        for file_info in data["files"]:
                            print(f"    - {file_info['name']} ({file_info['size']} bytes)")
            
            if "filesystem_read_file" in tools:
                read_tool = mcp_client.get_tool("filesystem_read_file")
                if read_tool:
                    print(f"\nReading case_notes.txt:")
                    result = await read_tool.execute({
                        "path": str(Path(temp_dir) / "case_notes.txt")
                    })
                    if result["success"]:
                        data = result["data"]
                        print(f"  Content: {data['content']}")
                        print(f"  Lines: {data['lines']}")
            
        else:
            print("❌ Failed to start filesystem server")
        
        # Cleanup
        await mcp_client.stop_all_servers()
        print("\n✓ Filesystem server stopped")


async def demo_with_freshai_core():
    """Demonstrate MCP integration with FreshAI Core."""
    print("\n=== FreshAI Core MCP Integration Demo ===")
    
    # Create custom config with MCP enabled
    config = Config()
    config.mcp.enable_mcp = True
    
    # Add filesystem server to config
    config.mcp.servers["filesystem"] = MCPServerConfig(
        name="filesystem",
        command="python",
        arguments=["-m", "mcp_servers.filesystem"],
        enabled=True
    )
    
    # Create FreshAI Core with custom config
    core = FreshAICore(config)
    
    try:
        print("Initializing FreshAI Core with MCP support...")
        await core.initialize()
        
        # List all available tools (including MCP tools)
        all_tools = core.get_available_tools()
        print(f"Total available tools: {len(all_tools)}")
        
        # Separate MCP tools from regular tools
        mcp_tools = [tool for tool in all_tools if "_" in tool and any(
            tool.startswith(f"{server}_") for server in config.mcp.servers.keys()
        )]
        regular_tools = [tool for tool in all_tools if tool not in mcp_tools]
        
        print(f"Regular tools: {regular_tools}")
        print(f"MCP tools: {mcp_tools}")
        
        # Test using MCP tools through FreshAI Core
        if mcp_tools:
            print(f"\nTesting MCP tool integration...")
            
            # Create a temporary file to test with
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write("Sample evidence file for MCP testing")
                tmp_path = tmp.name
            
            try:
                # Try to use a filesystem tool if available
                fs_tools = [tool for tool in mcp_tools if tool.startswith("filesystem_")]
                if fs_tools:
                    tool_name = fs_tools[0]  # Use first available filesystem tool
                    print(f"Using MCP tool: {tool_name}")
                    
                    if "get_file_info" in tool_name:
                        result = await core.use_tool(tool_name, {"path": tmp_path})
                        print(f"Tool result: {result}")
                    
            finally:
                Path(tmp_path).unlink()
        
    except Exception as e:
        print(f"Error in FreshAI Core integration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await core.cleanup()
        print("✓ FreshAI Core cleanup completed")


async def demo_web_search_server():
    """Demonstrate web search MCP server (mock implementation)."""
    print("\n=== Web Search MCP Server Demo ===")
    
    # Configure web search server
    ws_config = MCPServerConfig(
        name="web_search",
        command="python",
        arguments=["-m", "mcp_servers.web_search"],
        enabled=True
    )
    
    mcp_client = MCPClient()
    
    print("Starting web search MCP server...")
    success = await mcp_client.add_server(ws_config)
    
    if success:
        print("✓ Web search server started successfully")
        
        # Test URL extraction
        if "web_search_extract_urls" in mcp_client.get_available_tools():
            extract_tool = mcp_client.get_tool("web_search_extract_urls")
            if extract_tool:
                test_text = """
                Check these websites for more information:
                https://example.com/investigation
                http://evidence.org/analysis
                Visit https://crime-lab.gov for forensic details
                """
                
                print(f"\nExtracting URLs from text...")
                result = await extract_tool.execute({"text": test_text})
                if result["success"]:
                    data = result["data"]
                    print(f"Found {data['total_urls']} URLs:")
                    for url in data["urls_found"]:
                        print(f"  - {url}")
        
        # Test URL validation
        if "web_search_validate_url" in mcp_client.get_available_tools():
            validate_tool = mcp_client.get_tool("web_search_validate_url")
            if validate_tool:
                test_urls = [
                    "https://example.com",
                    "http://invalid-url",
                    "not-a-url-at-all"
                ]
                
                print(f"\nValidating URLs...")
                for url in test_urls:
                    result = await validate_tool.execute({"url": url})
                    if result["success"]:
                        data = result["data"]
                        status = "✓" if data["is_valid"] else "❌"
                        print(f"  {status} {url} - Valid: {data['is_valid']}")
    
    else:
        print("❌ Failed to start web search server")
    
    await mcp_client.stop_all_servers()
    print("✓ Web search server stopped")


async def main():
    """Run all MCP demos."""
    print("FreshAI MCP Integration Examples")
    print("=" * 50)
    
    try:
        # Demo individual MCP servers
        await demo_filesystem_mcp_server()
        await demo_web_search_server()
        
        # Demo integration with FreshAI Core
        await demo_with_freshai_core()
        
        print("\n" + "=" * 50)
        print("✅ All MCP demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())