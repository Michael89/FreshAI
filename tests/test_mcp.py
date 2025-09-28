"""Tests for MCP functionality in FreshAI."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mcp_config_imports():
    """Test that MCP modules can be imported."""
    try:
        from freshai.mcp import MCPClient, MCPTool, MCPConfig
        from freshai.mcp.config import MCPServerConfig
        assert MCPClient is not None
        assert MCPTool is not None
        assert MCPConfig is not None
        assert MCPServerConfig is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import MCP modules: {e}")


def test_mcp_config_creation():
    """Test MCP configuration creation."""
    from freshai.mcp.config import MCPConfig, MCPServerConfig
    
    # Test default config
    config = MCPConfig()
    assert config.enable_mcp is True
    assert config.default_timeout == 30
    assert config.max_concurrent_servers == 5
    assert isinstance(config.servers, dict)
    
    # Test server config
    server_config = MCPServerConfig(
        name="test_server",
        command="python",
        arguments=["-m", "test_module"],
    )
    assert server_config.name == "test_server"
    assert server_config.command == "python"
    assert server_config.arguments == ["-m", "test_module"]
    assert server_config.enabled is True


def test_mcp_config_from_env():
    """Test loading MCP configuration from environment."""
    from freshai.mcp.config import MCPConfig
    import os
    
    # Set test environment variables
    old_values = {}
    test_env = {
        "FRESHAI_ENABLE_MCP": "true",
        "FRESHAI_MCP_DEFAULT_TIMEOUT": "45",
        "FRESHAI_MCP_MAX_CONCURRENT_SERVERS": "3",
        "FRESHAI_MCP_FILESYSTEM_ENABLED": "true",
        "FRESHAI_MCP_WEB_SEARCH_ENABLED": "false"
    }
    
    for key, value in test_env.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        config = MCPConfig.load_from_env()
        assert config.enable_mcp is True
        assert config.default_timeout == 45
        assert config.max_concurrent_servers == 3
        assert "filesystem" in config.servers
        assert "web_search" not in config.servers or not config.servers["web_search"].enabled
    finally:
        # Restore environment
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def test_main_config_includes_mcp():
    """Test that main config includes MCP configuration."""
    from freshai.config import Config
    
    config = Config()
    assert hasattr(config, 'mcp')
    assert config.mcp is not None
    assert hasattr(config.mcp, 'enable_mcp')


async def test_mcp_client_creation():
    """Test MCP client creation."""
    from freshai.mcp import MCPClient
    
    client = MCPClient()
    assert client is not None
    assert len(client.servers) == 0
    assert len(client.tools) == 0
    assert client.get_available_tools() == []


async def test_mcp_tool_creation():
    """Test MCP tool creation."""
    from freshai.mcp import MCPTool
    from freshai.mcp.client import MCPServerProcess
    from freshai.mcp.config import MCPServerConfig
    
    # Create a mock server process
    server_config = MCPServerConfig(
        name="test_server",
        command="python",
        arguments=["-m", "test"]
    )
    
    server_process = MCPServerProcess(server_config)
    
    tool_schema = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Test input"
            }
        },
        "required": ["input"]
    }
    
    tool = MCPTool("test_server", "test_tool", tool_schema, server_process)
    
    assert tool.server_name == "test_server"
    assert tool.tool_name == "test_tool"
    assert tool.name == "test_server_test_tool"
    assert tool.get_schema() == tool_schema


async def test_filesystem_mcp_server():
    """Test the filesystem MCP server."""
    import subprocess
    import json
    import tempfile
    import os
    
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello, World!")
        
        test_dir = Path(temp_dir) / "subdir"
        test_dir.mkdir()
        
        # Start the filesystem MCP server
        server_path = Path(__file__).parent.parent / "mcp_servers" / "filesystem.py"
        
        try:
            process = subprocess.Popen(
                [sys.executable, str(server_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Test tools/list
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            response_line = process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                assert response["jsonrpc"] == "2.0"
                assert "result" in response
                assert "tools" in response["result"]
                tools = response["result"]["tools"]
                tool_names = [tool["name"] for tool in tools]
                assert "list_files" in tool_names
                assert "read_file" in tool_names
                assert "get_file_info" in tool_names
            
            # Test list_files
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "list_files",
                    "arguments": {"path": temp_dir}
                }
            }
            
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            response_line = process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                assert response["jsonrpc"] == "2.0"
                assert "result" in response
                result = response["result"]
                assert "files" in result
                assert "directories" in result
                assert len(result["files"]) >= 1  # Should have test.txt
                assert len(result["directories"]) >= 1  # Should have subdir
                
        finally:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()


async def test_web_search_mcp_server():
    """Test the web search MCP server."""
    import subprocess
    import json
    
    server_path = Path(__file__).parent.parent / "mcp_servers" / "web_search.py"
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Test tools/list
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            assert response["jsonrpc"] == "2.0"
            assert "result" in response
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]
            assert "search_web" in tool_names
            assert "extract_urls" in tool_names
            assert "validate_url" in tool_names
        
        # Test extract_urls
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "extract_urls",
                "arguments": {
                    "text": "Visit https://example.com and http://test.org for more info"
                }
            }
        }
        
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            assert response["jsonrpc"] == "2.0"
            assert "result" in response
            result = response["result"]
            assert "urls_found" in result
            assert len(result["urls_found"]) == 2
            assert "https://example.com" in result["urls_found"]
            assert "http://test.org" in result["urls_found"]
            
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()


async def test_database_mcp_server():
    """Test the database MCP server."""
    import subprocess
    import json
    import tempfile
    
    server_path = Path(__file__).parent.parent / "mcp_servers" / "database.py"
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Test tools/list
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            assert response["jsonrpc"] == "2.0"
            assert "result" in response
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]
            assert "execute_query" in tool_names
            assert "list_tables" in tool_names
            assert "describe_table" in tool_names
            assert "create_sample_database" in tool_names
        
        # Test create_sample_database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "create_sample_database",
                    "arguments": {"database_path": db_path}
                }
            }
            
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            response_line = process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                assert response["jsonrpc"] == "2.0"
                assert "result" in response
                result = response["result"]
                assert result["success"] is True
                assert "tables_created" in result
                assert len(result["tables_created"]) > 0
        finally:
            Path(db_path).unlink(missing_ok=True)
            
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()


def run_async_tests():
    """Run all async tests."""
    async def run_all():
        await test_mcp_client_creation()
        await test_mcp_tool_creation()
        await test_filesystem_mcp_server()
        await test_web_search_mcp_server()
        await test_database_mcp_server()
    
    asyncio.run(run_all())


if __name__ == "__main__":
    # Run basic tests
    test_mcp_config_imports()
    print("✓ MCP config imports working")
    
    test_mcp_config_creation()
    print("✓ MCP config creation working")
    
    test_mcp_config_from_env()
    print("✓ MCP config from environment working")
    
    test_main_config_includes_mcp()
    print("✓ Main config includes MCP")
    
    # Run async tests
    run_async_tests()
    print("✓ MCP client and server tests working")
    
    print("\n✅ All MCP tests passed!")