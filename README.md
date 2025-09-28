# FreshAI - AI Assistant for Crime Investigators

FreshAI is a comprehensive AI platform designed to assist crime investigators with their work by combining Large Language Models (LLM) and Vision Language Models (VLM) capabilities. The system integrates with Ollama server for local inference and supports Transformers library for open source models.

## Features

- **Multi-modal AI Analysis**: Supports both text and vision analysis for comprehensive evidence examination
- **Ollama Integration**: Seamless integration with local Ollama server for LLM and VLM inference
- **Transformers Support**: Built-in support for Hugging Face Transformers models
- **Investigation Tools**: Specialized tools for evidence analysis, image forensics, and text pattern recognition
- **Case Management**: Complete case lifecycle management with evidence tracking and reporting
- **CLI Interface**: User-friendly command-line interface for investigators
- **Professional Reporting**: Automated generation of investigation reports

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Michael89/FreshAI.git
cd FreshAI

# Install dependencies with uv (recommended)
uv sync

# Or install in development mode with pip
pip install -e .
```

### Prerequisites

1. **Ollama Server** (recommended for best performance):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull recommended models
   ollama pull llama2
   ollama pull llava
   ```

2. **Python 3.9+** and **uv** (install with `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Initialize FreshAI

```bash
# Initialize FreshAI in current directory
uv run freshai init

# Or initialize in specific directory
uv run freshai init /path/to/investigation/workspace
```

This creates the necessary directories and configuration files:
- `evidence/` - Store evidence files
- `cases/` - Store case data and reports
- `config/` - Configuration files
- `.env` - Environment configuration

## Usage

### 1. Start a New Case

```bash
uv run freshai start-case CASE001 --description "Investigation of digital evidence"
```

### 2. Analyze Evidence

```bash
# Analyze an image
uv run freshai analyze CASE001 /path/to/evidence/photo.jpg --type image

# Analyze text document
uv run freshai analyze CASE001 /path/to/evidence/chat_log.txt --type text

# Auto-detect evidence type
uv run freshai analyze CASE001 /path/to/evidence/document.pdf
```

### 3. Ask Questions About the Case

```bash
uv run freshai ask CASE001 "What suspicious elements were found in the analyzed images?"
```

### 4. Generate Investigation Report

```bash
# Generate and display report
uv run freshai report CASE001

# Save report to file
uv run freshai report CASE001 --output /path/to/reports/case001_final.json
```

### 5. Check Case Status

```bash
# View case details
uv run freshai status CASE001

# List all active cases
uv run freshai list-cases
```

### 6. Close Case

```bash
uv run freshai close-case CASE001
```

## Python API Usage

```python
import asyncio
from freshai import InvestigatorAgent, Config

async def main():
    # Initialize agent
    config = Config.load_from_env()
    agent = InvestigatorAgent(config)
    await agent.initialize()
    
    # Start new case
    case_info = await agent.start_case("CASE001", "Digital evidence investigation")
    
    # Analyze evidence
    analysis = await agent.analyze_evidence("CASE001", "/path/to/evidence/image.jpg")
    print(f"Analysis complete: {analysis['evidence_type']}")
    
    # Ask questions
    response = await agent.ask_question("CASE001", "What objects are visible in the image?")
    print(f"Answer: {response['answer']}")
    
    # Generate report
    report = await agent.generate_case_report("CASE001")
    print(f"Report generated: {len(report['report_content'])} characters")
    
    # Cleanup
    await agent.cleanup()

# Run the example
asyncio.run(main())
```

## Configuration

FreshAI can be configured through environment variables or a `.env` file:

```env
# General Settings
FRESHAI_DEBUG=false
FRESHAI_LOG_LEVEL=INFO
FRESHAI_ENABLE_VISION=true
FRESHAI_ENABLE_TOOLS=true

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_DEFAULT_LLM=llama2
OLLAMA_DEFAULT_VLM=llava

# Storage Paths
FRESHAI_EVIDENCE_STORAGE_PATH=./evidence
FRESHAI_CASE_STORAGE_PATH=./cases

# Transformers Configuration
TRANSFORMERS_CACHE_DIR=./models
TRANSFORMERS_DEVICE=auto
```

## Available Models

### Ollama Models (Recommended)
- **LLM**: llama2, codellama, mistral, phi
- **VLM**: llava, bakllava

### Transformers Models
- **Text**: bert-base-uncased, distilbert-base-uncased
- **Vision**: openai/clip-vit-base-patch32, facebook/deit-base-distilled-patch16-224
- **Multimodal**: Salesforce/blip-image-captioning-base

## Investigation Tools

FreshAI includes specialized tools for criminal investigations:

### Evidence Analyzer
- File integrity verification (MD5, SHA1, SHA256 hashes)
- Metadata extraction
- Content analysis based on file type
- Forensic metadata preservation

### Image Analyzer
- Image quality assessment
- Object detection (basic)
- Forensic analysis and tampering detection
- EXIF metadata extraction

### Text Analyzer
- Pattern extraction (phone numbers, emails, addresses, etc.)
- Suspicious keyword detection
- Entity extraction (names, locations, organizations)
- Sentiment and linguistic analysis

## MCP Server Support

FreshAI supports the Model Context Protocol (MCP) for extending functionality through external servers. MCP allows you to connect various specialized tools and services to enhance investigation capabilities.

### Built-in MCP Servers

FreshAI includes several example MCP servers:

#### Filesystem Server
Provides secure file system operations:
- List files and directories
- Read file contents safely
- Get detailed file information
- Access metadata and permissions

#### Web Search Server
Offers web-related utilities:
- Extract URLs from text
- Validate URL formats
- Mock web search functionality (extensible to real search APIs)

#### Database Server
Enables database operations:
- Execute SQL queries on SQLite databases
- List tables and describe schemas
- Create sample investigation databases
- Secure parameter binding

### MCP Configuration

Configure MCP servers through environment variables:

```env
# MCP Settings
FRESHAI_ENABLE_MCP=true
FRESHAI_MCP_DEFAULT_TIMEOUT=30
FRESHAI_MCP_MAX_CONCURRENT_SERVERS=5

# Enable/disable specific servers
FRESHAI_MCP_FILESYSTEM_ENABLED=true
FRESHAI_MCP_WEB_SEARCH_ENABLED=false
```

### Using MCP Tools

MCP tools integrate seamlessly with FreshAI's tool registry:

```python
from freshai.core import FreshAICore
import asyncio

async def use_mcp_tools():
    core = FreshAICore()
    await core.initialize()
    
    # List all available tools (including MCP tools)
    tools = core.get_available_tools()
    print("Available tools:", tools)
    
    # Use filesystem MCP tool
    if "filesystem_list_files" in tools:
        result = await core.use_tool("filesystem_list_files", {
            "path": "/path/to/evidence"
        })
        print("Files found:", result)
    
    await core.cleanup()

asyncio.run(use_mcp_tools())
```

### Creating Custom MCP Servers

Create your own MCP servers for specialized tools:

```python
#!/usr/bin/env python3
"""Custom MCP server example."""

import json
import sys

class CustomMCPServer:
    def __init__(self):
        self.tools = {
            "my_tool": {
                "name": "my_tool",
                "description": "My custom investigation tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input parameter"}
                    },
                    "required": ["input"]
                }
            }
        }
    
    def handle_tools_list(self, request):
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"tools": list(self.tools.values())}
        }
    
    def handle_tools_call(self, request):
        # Implement your tool logic here
        params = request.get("params", {})
        # ... tool implementation ...
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {...}}
    
    def run(self):
        for line in sys.stdin:
            request = json.loads(line.strip())
            method = request.get("method")
            
            if method == "tools/list":
                response = self.handle_tools_list(request)
            elif method == "tools/call":
                response = self.handle_tools_call(request)
            
            print(json.dumps(response))
            sys.stdout.flush()

if __name__ == "__main__":
    CustomMCPServer().run()
```

## Architecture

```
FreshAI/
├── freshai/
│   ├── core/           # Core system orchestrator
│   ├── agent/          # Investigation agent
│   ├── llm/            # Language model integrations
│   ├── vlm/            # Vision language model integrations
│   ├── tools/          # Investigation tools
│   ├── config/         # Configuration management
│   └── utils/          # Utility functions
├── examples/           # Usage examples
└── tests/             # Test suite
```

## Development

### Setting up Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/Michael89/FreshAI.git
cd FreshAI

# Install with development dependencies using uv
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black freshai/
uv run ruff check freshai/

# Type checking
uv run mypy freshai/
```

### Adding New Tools

```python
from freshai.tools.registry import BaseTool

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Description of what the tool does"
        )
    
    async def execute(self, parameters):
        # Tool implementation
        return {"result": "tool output"}
    
    def get_schema(self):
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param1"]
        }
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for the Cellebrite Hackathon
- Powered by Ollama and Hugging Face Transformers
- Designed for law enforcement and investigation professionals

## Support

For questions, issues, or contributions, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Join our community discussions

---

**Note**: This tool is designed to assist investigations but should not replace proper forensic procedures and human expertise. Always follow your organization's protocols and legal requirements.
