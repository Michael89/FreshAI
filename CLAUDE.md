# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreshAI is an AI-powered investigation assistant designed for crime investigators. It combines Large Language Models (LLM) and Vision Language Models (VLM) to analyze evidence, answer questions about cases, and generate professional reports. The system integrates with Ollama for local inference and supports Hugging Face Transformers.

## Architecture

FreshAI follows a modular architecture with these core components:

- **`freshai/core/`** - Central orchestrator that coordinates all system components
- **`freshai/agent/`** - Investigation agent that manages case workflows
- **`freshai/llm/`** - Language model integrations (Ollama, Transformers)
- **`freshai/vlm/`** - Vision language model integrations
- **`freshai/tools/`** - Investigation tools for evidence analysis
- **`freshai/config/`** - Configuration management and validation
- **`freshai/mcp/`** - Model Context Protocol (MCP) server integration
- **`freshai/utils/`** - Utility functions and validation
- **`mcp_servers/`** - Example MCP servers (filesystem, web search, database)
- **`examples/`** - Usage examples and demonstrations
- **`tests/`** - Test suite

### Key Design Patterns

1. **Async-first architecture**: All core operations use async/await for non-blocking execution
2. **Tool registry system**: Pluggable tools registered via `ToolRegistry` class
3. **Configuration-driven**: Environment variables and `.env` files control behavior
4. **MCP integration**: External tools and services accessible via Model Context Protocol
5. **Graceful degradation**: Core functionality works without heavy ML dependencies

## Development Commands

### Installation and Setup

```bash
# Install uv (if not already installed)
pip install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (recommended)
uv sync

# Install with development dependencies
uv sync --all-extras

# Initialize workspace
uv run freshai init [directory]
```

### Testing

```bash
# Run basic tests (no ML dependencies required)
uv run python tests/test_basic.py

# Run full test suite
uv run pytest

# Run specific test modules
uv run pytest tests/test_mcp.py
uv run pytest tests/test_basic.py

# Run with coverage
uv run pytest --cov=freshai
```

### Code Quality

```bash
# Format code
uv run black freshai/

# Lint code
uv run ruff check freshai/

# Type checking
uv run mypy freshai/
```

### Running the Application

```bash
# Demo script (basic functionality without ML dependencies)
uv run python demo.py

# CLI commands
uv run freshai --help
uv run freshai init ./workspace
uv run freshai start-case CASE001 --description "Test case"
uv run freshai analyze CASE001 evidence/file.txt
uv run freshai ask CASE001 "What evidence was found?"
uv run freshai report CASE001
```

## Key Configuration

### Environment Variables

The system is configured through `.env` files or environment variables:

```env
# Core settings
FRESHAI_DEBUG=false
FRESHAI_LOG_LEVEL=INFO
FRESHAI_ENABLE_VISION=true
FRESHAI_ENABLE_TOOLS=true

# Ollama integration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_DEFAULT_LLM=llama2
OLLAMA_DEFAULT_VLM=llava

# MCP settings
FRESHAI_ENABLE_MCP=true
FRESHAI_MCP_FILESYSTEM_ENABLED=true

# Storage paths
FRESHAI_EVIDENCE_STORAGE_PATH=./evidence
FRESHAI_CASE_STORAGE_PATH=./cases
```

### Configuration Classes

- `Config` - Main configuration class that loads from environment
- `OllamaConfig` - Ollama server connection settings
- `TransformersConfig` - Hugging Face model configurations
- `MCPConfig` - Model Context Protocol server settings

## Important Development Guidelines

### Dependency Management

The codebase supports graceful degradation - core functionality works without heavy ML dependencies:

- **Always works**: Configuration, tools, text analysis, evidence hashing, CLI framework, MCP support
- **Requires ML dependencies**: Ollama integration, Transformers models, image analysis, full agent functionality

When adding new features, check if dependencies are available and handle missing imports gracefully.

### Tool Development

Create new investigation tools by extending `BaseTool`:

```python
from freshai.tools.registry import BaseTool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(name="my_tool", description="Tool description")

    async def execute(self, parameters):
        # Tool implementation
        return {"result": "output"}

    def get_schema(self):
        return {"type": "object", "properties": {...}}
```

Register tools in the `FreshAICore._initialize_tools()` method.

### MCP Server Integration

MCP servers extend functionality through external processes. Example servers are in `mcp_servers/`:

- `filesystem.py` - File system operations
- `web_search.py` - Web search capabilities
- `database.py` - SQLite database access

MCP servers communicate via JSON-RPC over stdin/stdout.

### Case Management

Cases follow this workflow:
1. `start_case()` - Initialize new investigation
2. `analyze_evidence()` - Process evidence files
3. `ask_question()` - Query case data with LLM
4. `generate_case_report()` - Create final report
5. `close_case()` - Finalize investigation

### Error Handling

The system uses structured error handling:
- Configuration errors: Validation with helpful messages
- Missing dependencies: Graceful degradation with user guidance
- Tool failures: Error results rather than exceptions
- MCP server issues: Automatic retries and fallbacks

## Testing Strategy

### Test Structure

- `test_basic.py` - Tests core functionality without ML dependencies
- `test_mcp.py` - Tests MCP server integration
- Test files use `pytest` and `pytest-asyncio` for async testing

### Running Tests

Tests are designed to run without external dependencies. Use `python tests/test_basic.py` for quick validation or `pytest` for full suite.

### Test Coverage

Focus on testing:
- Configuration loading and validation
- Tool registration and execution
- Case workflow management
- MCP server communication
- CLI command structure

## Common Development Tasks

### Adding New LLM/VLM Support

1. Create integration in `freshai/llm/` or `freshai/vlm/`
2. Extend base model classes from `freshai/core/base.py`
3. Add configuration options to appropriate config classes
4. Register models in `FreshAICore._initialize_*_models()`

### Creating MCP Servers

1. Implement JSON-RPC server in `mcp_servers/`
2. Add server configuration to `MCPConfig`
3. Test server integration with `MCPClient`

### Extending Evidence Analysis

1. Add analysis logic to existing tools in `freshai/tools/`
2. Or create new tools following the `BaseTool` pattern
3. Register tools in the tool registry
4. Update CLI commands to expose new functionality