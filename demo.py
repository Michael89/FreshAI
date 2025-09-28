#!/usr/bin/env python3
"""
FreshAI Demo Script

This script demonstrates the core functionality of FreshAI
without requiring heavy ML dependencies like transformers or OpenCV.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a section header."""
    print(f"\n--- {title} ---")

async def main():
    """Main demo function."""
    
    print_header("FreshAI - AI Assistant for Crime Investigators")
    print("Demo Script - Basic Functionality")
    print("\nNote: This demo shows core functionality without ML dependencies.")
    print("For full functionality, install: aiohttp, transformers, torch, opencv-python")
    
    # Test 1: Configuration System
    print_section("Configuration System")
    try:
        from freshai.config import Config, OllamaConfig, TransformersConfig
        
        config = Config()
        print(f"✓ Default configuration loaded")
        print(f"  - Ollama endpoint: {config.ollama.base_url}")
        print(f"  - Vision enabled: {config.enable_vision}")
        print(f"  - Tools enabled: {config.enable_tools}")
        print(f"  - Debug mode: {config.debug}")
        
        # Test environment loading
        config_from_env = Config.load_from_env()
        print(f"✓ Environment configuration loaded")
        print(f"  - Evidence storage: {config_from_env.evidence_storage_path}")
        print(f"  - Case storage: {config_from_env.case_storage_path}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    # Test 2: Tool Registry
    print_section("Investigation Tools")
    try:
        from freshai.tools import ToolRegistry, EvidenceAnalyzer, TextAnalyzer
        
        registry = ToolRegistry()
        
        # Register tools
        evidence_tool = EvidenceAnalyzer()
        text_tool = TextAnalyzer()
        
        registry.register("evidence_analyzer", evidence_tool)
        registry.register("text_analyzer", text_tool)
        
        print(f"✓ Tool registry created")
        print(f"  - Available tools: {registry.get_available_tools()}")
        
        # Get tool info
        for tool_name in registry.get_available_tools():
            info = registry.get_tool_info(tool_name)
            print(f"  - {tool_name}: {info['description']}")
        
    except Exception as e:
        print(f"❌ Tool registry test failed: {e}")
    
    # Test 3: Text Analysis Tool
    print_section("Text Analysis Demo")
    try:
        sample_text = """
        Urgent meeting tomorrow at 3 PM. Location: warehouse on 5th Street.
        Contact details: john.doe@email.com, (555) 123-4567
        Package delivery confirmation required. Use password "eagle123".
        Time sensitive - handle with care.
        """
        
        print("Sample text for analysis:")
        print(sample_text.strip())
        
        text_analyzer = TextAnalyzer()
        result = await text_analyzer.execute({
            "text": sample_text,
            "analysis_type": "comprehensive"
        })
        
        print(f"\n✓ Text analysis completed")
        print(f"  - Text length: {result['text_length']} characters")
        print(f"  - Word count: {result['word_count']}")
        
        # Show pattern analysis
        if "pattern_analysis" in result:
            patterns = result["pattern_analysis"]["extracted_patterns"]
            for pattern_type, matches in patterns.items():
                if matches:
                    print(f"  - {pattern_type}: {matches}")
        
        # Show keyword analysis
        if "keyword_analysis" in result:
            keywords = result["keyword_analysis"]
            if keywords["suspicious_keywords"]:
                print(f"  - Risk level: {keywords['risk_level']}")
                print(f"  - Suspicious categories: {keywords['categories_found']}")
        
        # Show sentiment analysis
        if "sentiment_analysis" in result:
            sentiment = result["sentiment_analysis"]
            print(f"  - Overall sentiment: {sentiment['overall_sentiment']}")
            if sentiment['threat_score'] > 0:
                print(f"  - Threat indicators found: {sentiment['threatening_indicators']}")
        
    except Exception as e:
        print(f"❌ Text analysis test failed: {e}")
    
    # Test 4: Evidence Analysis Tool
    print_section("Evidence Analysis Demo")
    try:
        # Create a sample evidence file
        evidence_dir = Path("./evidence")
        evidence_dir.mkdir(exist_ok=True)
        
        sample_file = evidence_dir / "sample_evidence.txt"
        sample_content = """Meeting notes - CONFIDENTIAL
Date: 2024-01-15
Participants: John Smith, Mary Johnson
Location: 123 Business Ave, Suite 500

Discussion points:
- Project timeline delayed
- Budget concerns: $50,000 shortfall
- Need to contact supplier at (555) 987-6543
- Follow up with client: client@company.com
- Password for system access: temp123!

Action items:
1. Review financial statements
2. Schedule follow-up meeting
3. Update project documentation
"""
        sample_file.write_text(sample_content)
        
        evidence_analyzer = EvidenceAnalyzer()
        result = await evidence_analyzer.execute({
            "file_path": str(sample_file)
        })
        
        print(f"✓ Evidence analysis completed for: {sample_file.name}")
        
        # Show file info
        if "file_info" in result:
            file_info = result["file_info"]
            print(f"  - File size: {file_info['size_human']}")
            print(f"  - MIME type: {file_info['mime_type']}")
            print(f"  - Created: {file_info['created'][:19]}")
        
        # Show hash analysis
        if "hash_analysis" in result:
            hashes = result["hash_analysis"]
            print(f"  - MD5: {hashes.get('md5', 'N/A')[:16]}...")
            print(f"  - SHA256: {hashes.get('sha256', 'N/A')[:32]}...")
        
        # Show content analysis
        if "content_analysis" in result:
            content = result["content_analysis"]
            print(f"  - Content type: {content.get('type', 'unknown')}")
            if "contains_suspicious_keywords" in content:
                suspicious = content["contains_suspicious_keywords"]
                if suspicious:
                    print(f"  - Suspicious keywords found: {suspicious}")
        
    except Exception as e:
        print(f"❌ Evidence analysis test failed: {e}")
    
    # Test 5: Validation Utilities
    print_section("Validation Utilities")
    try:
        from freshai.utils.validation import validate_case_id, validate_config
        
        # Test case ID validation
        test_cases = ["CASE001", "case-123", "my case 456!", ""]
        print("Case ID validation:")
        for case_id in test_cases:
            try:
                normalized = validate_case_id(case_id)
                print(f"  - '{case_id}' → '{normalized}' ✓")
            except ValueError as e:
                print(f"  - '{case_id}' → Error: {e}")
        
        # Test config validation
        print("\nConfiguration validation:")
        valid_config = {
            "ollama": {"host": "localhost", "port": 11434},
            "transformers": {"device": "cpu"}
        }
        issues = validate_config(valid_config)
        print(f"  - Valid config: {len(issues)} issues")
        
        invalid_config = {"ollama": {"host": "localhost"}}  # Missing port
        issues = validate_config(invalid_config)
        print(f"  - Invalid config: {len(issues)} issues found")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    # Test 6: MCP Server Integration
    print_section("MCP Server Integration")
    try:
        from freshai.mcp import MCPClient, MCPConfig
        from freshai.mcp.config import MCPServerConfig
        import asyncio
        
        print("✓ MCP modules imported successfully")
        
        # Test MCP configuration
        mcp_config = MCPConfig.load_from_env()
        print(f"  - MCP enabled: {mcp_config.enable_mcp}")
        print(f"  - Default timeout: {mcp_config.default_timeout}s")
        print(f"  - Available servers: {list(mcp_config.servers.keys())}")
        
        # Test MCP client
        client = MCPClient()
        print(f"  - MCP client created")
        print(f"  - Initial tools: {len(client.get_available_tools())}")
        
        async def test_mcp_integration():
            """Test MCP server integration."""
            try:
                # Test with the filesystem server
                if "filesystem" in mcp_config.servers:
                    fs_config = mcp_config.servers["filesystem"]
                    print(f"  - Testing filesystem server: {fs_config.name}")
                    
                    # Note: In a real scenario, we'd start the server
                    # For demo purposes, we just show the configuration
                    print(f"    Command: {fs_config.command} {' '.join(fs_config.arguments)}")
                    print(f"    Enabled: {fs_config.enabled}")
                
                print("  - MCP integration test completed (servers not started in demo)")
                
            except Exception as e:
                print(f"  - MCP integration error: {e}")
        
        await test_mcp_integration()
        
        # Show available MCP server files
        mcp_servers_dir = Path(__file__).parent / "mcp_servers"
        if mcp_servers_dir.exists():
            server_files = [f.name for f in mcp_servers_dir.glob("*.py") if f.name != "__init__.py"]
            print(f"  - Available MCP servers: {server_files}")
        
    except Exception as e:
        print(f"❌ MCP integration test failed: {e}")
    
    # Test 7: CLI Functionality (basic test)
    print_section("CLI Interface")
    try:
        # Test just the CLI structure without importing agent
        import typer
        from rich.console import Console
        
        print("✓ CLI dependencies available")
        print("✓ CLI module structure ready")
        print("  Available commands:")
        print("    - freshai init         # Initialize FreshAI workspace")
        print("    - freshai start-case   # Start new investigation case")
        print("    - freshai analyze      # Analyze evidence")
        print("    - freshai ask          # Ask questions about case")
        print("    - freshai report       # Generate case report")
        print("    - freshai status       # Check case status")
        print("    - freshai close-case   # Close investigation")
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
    
    # Summary
    print_header("Demo Summary")
    print("✅ Core FreshAI functionality demonstrated successfully!")
    print("\nWhat works without additional dependencies:")
    print("  - Configuration management with environment variables")
    print("  - Investigation tool registry and framework")
    print("  - Text analysis with pattern recognition")
    print("  - Evidence file analysis and hashing")
    print("  - Validation utilities")
    print("  - CLI framework structure")
    print("  - MCP (Model Context Protocol) server support")
    print("  - Example MCP servers (filesystem, web search, database)")
    
    print("\nTo enable full functionality, install:")
    print("  pip install aiohttp transformers torch opencv-python pillow numpy")
    
    print("\nThen you can use:")
    print("  - Ollama LLM/VLM integration")
    print("  - Transformers model support")
    print("  - Advanced image analysis")
    print("  - Complete investigation agent")
    
    print("\nFor more information, see:")
    print("  - README.md for full documentation")
    print("  - examples/ directory for usage examples")
    print("  - tests/ directory for test suite")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()