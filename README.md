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

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
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

2. **Python 3.8+** with the required dependencies

### Initialize FreshAI

```bash
# Initialize FreshAI in current directory
freshai init

# Or initialize in specific directory
freshai init /path/to/investigation/workspace
```

This creates the necessary directories and configuration files:
- `evidence/` - Store evidence files
- `cases/` - Store case data and reports
- `config/` - Configuration files
- `.env` - Environment configuration

## Usage

### 1. Start a New Case

```bash
freshai start-case CASE001 --description "Investigation of digital evidence"
```

### 2. Analyze Evidence

```bash
# Analyze an image
freshai analyze CASE001 /path/to/evidence/photo.jpg --type image

# Analyze text document
freshai analyze CASE001 /path/to/evidence/chat_log.txt --type text

# Auto-detect evidence type
freshai analyze CASE001 /path/to/evidence/document.pdf
```

### 3. Ask Questions About the Case

```bash
freshai ask CASE001 "What suspicious elements were found in the analyzed images?"
```

### 4. Generate Investigation Report

```bash
# Generate and display report
freshai report CASE001

# Save report to file
freshai report CASE001 --output /path/to/reports/case001_final.json
```

### 5. Check Case Status

```bash
# View case details
freshai status CASE001

# List all active cases
freshai list-cases
```

### 6. Close Case

```bash
freshai close-case CASE001
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
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black freshai/
ruff check freshai/

# Type checking
mypy freshai/
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
