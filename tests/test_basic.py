"""Basic tests for FreshAI core functionality."""

import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_imports():
    """Test that config modules can be imported."""
    from freshai.config import Config, OllamaConfig, TransformersConfig
    
    # Test basic config creation
    config = Config()
    assert config is not None
    assert hasattr(config, 'ollama')
    assert hasattr(config, 'transformers')
    
    # Test Ollama config
    ollama_config = OllamaConfig()
    assert ollama_config.host == "localhost"
    assert ollama_config.port == 11434
    assert ollama_config.base_url == "http://localhost:11434"


def test_tools_imports():
    """Test that tool modules can be imported."""
    from freshai.tools import ToolRegistry, EvidenceAnalyzer, TextAnalyzer
    from freshai.tools.registry import BaseTool
    
    # Try to import ImageAnalyzer if available
    try:
        from freshai.tools import ImageAnalyzer
        has_image_analyzer = True
    except ImportError:
        has_image_analyzer = False
    
    # Test tool registry
    registry = ToolRegistry()
    assert registry is not None
    assert len(registry.get_available_tools()) == 0
    
    # Test tool registration
    text_analyzer = TextAnalyzer()
    registry.register("text_analyzer", text_analyzer)
    
    tools = registry.get_available_tools()
    assert "text_analyzer" in tools
    assert len(tools) == 1


def test_text_analyzer_basic():
    """Test basic text analyzer functionality."""
    from freshai.tools.text import TextAnalyzer
    
    analyzer = TextAnalyzer()
    assert analyzer.name == "text_analyzer"
    assert "text content" in analyzer.description.lower()
    
    # Test schema
    schema = analyzer.get_schema()
    assert "properties" in schema
    assert "text" in schema["properties"] or "file_path" in schema["properties"]


def test_evidence_analyzer_basic():
    """Test basic evidence analyzer functionality."""
    from freshai.tools.evidence import EvidenceAnalyzer
    
    analyzer = EvidenceAnalyzer()
    assert analyzer.name == "evidence_analyzer"
    assert "evidence" in analyzer.description.lower()
    
    # Test schema
    schema = analyzer.get_schema()
    assert "properties" in schema
    assert "file_path" in schema["properties"]


def test_config_validation():
    """Test configuration validation."""
    from freshai.utils.validation import validate_config
    
    # Test valid config
    valid_config = {
        "ollama": {"host": "localhost", "port": 11434},
        "transformers": {"device": "cpu"}
    }
    
    issues = validate_config(valid_config)
    assert len(issues) == 0
    
    # Test invalid config
    invalid_config = {
        "ollama": {"host": "localhost"}  # Missing port
    }
    
    issues = validate_config(invalid_config)
    assert len(issues) > 0
    assert any("port" in issue.lower() for issue in issues)


def test_case_id_validation():
    """Test case ID validation."""
    from freshai.utils.validation import validate_case_id
    
    # Valid case IDs
    assert validate_case_id("CASE001") == "CASE001"
    assert validate_case_id("case-001") == "case-001"
    assert validate_case_id("case_001") == "case_001"
    
    # Invalid case IDs
    try:
        validate_case_id("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Normalized case ID
    normalized = validate_case_id("case with spaces!")
    assert " " not in normalized
    assert "!" not in normalized


def test_text_analyzer_execution():
    """Test text analyzer execution with sample data."""
    import asyncio
    from freshai.tools.text import TextAnalyzer
    
    analyzer = TextAnalyzer()
    
    # Sample text for analysis
    sample_text = "Call me at (555) 123-4567 tomorrow. Meet at 123 Main Street."
    
    async def run_test():
        result = await analyzer.execute({
            "text": sample_text,
            "analysis_type": "patterns"
        })
        
        assert "pattern_analysis" in result
        patterns = result["pattern_analysis"]["extracted_patterns"]
        
        # Should find phone number
        assert "phone_numbers" in patterns
        assert len(patterns["phone_numbers"]) > 0
        
        # Should find address
        assert "addresses" in patterns
        assert len(patterns["addresses"]) > 0
    
    asyncio.run(run_test())


if __name__ == "__main__":
    # Run basic tests
    test_config_imports()
    print("✓ Config imports working")
    
    test_tools_imports()
    print("✓ Tools imports working")
    
    test_text_analyzer_basic()
    print("✓ Text analyzer basic functionality working")
    
    test_evidence_analyzer_basic()
    print("✓ Evidence analyzer basic functionality working")
    
    test_config_validation()
    print("✓ Config validation working")
    
    test_case_id_validation()
    print("✓ Case ID validation working")
    
    test_text_analyzer_execution()
    print("✓ Text analyzer execution working")
    
    print("\n✅ All basic tests passed!")