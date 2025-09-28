# FreshAI Installation Guide

This guide will help you install and set up FreshAI for crime investigation assistance.

## System Requirements

- Python 3.8 or higher
- 4GB+ RAM (more recommended for ML models)
- Disk space: 2GB+ (more for model storage)

## Installation Options

### Option 1: Quick Start (Basic Functionality)

For basic functionality without heavy ML dependencies:

```bash
# Clone the repository
git clone https://github.com/Michael89/FreshAI.git
cd FreshAI

# Install minimal dependencies
pip install pydantic python-dotenv typer[all] rich pytest

# Run the demo
python demo.py
```

This provides:
- Configuration management
- Investigation tools (text analysis, evidence analysis)
- CLI framework
- Basic validation and utilities

### Option 2: Full Installation (Recommended)

For complete functionality including LLM/VLM capabilities:

```bash
# Clone the repository
git clone https://github.com/Michael89/FreshAI.git
cd FreshAI

# Install all dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

This includes:
- All basic functionality
- Ollama integration for local LLM/VLM
- Transformers library support
- Advanced image analysis with OpenCV
- Complete investigation agent

### Option 3: Development Installation

For development and contributing:

```bash
# Clone the repository
git clone https://github.com/Michael89/FreshAI.git
cd FreshAI

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black freshai/
ruff check freshai/
mypy freshai/
```

## External Dependencies

### Ollama (Recommended for LLM/VLM)

Install Ollama for local AI model inference:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull llama2      # General LLM
ollama pull llava       # Vision-Language Model
ollama pull codellama   # Code analysis
ollama pull mistral     # Alternative LLM
```

### GPU Support (Optional)

For GPU acceleration with transformers:

```bash
# For NVIDIA GPUs with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Configuration

### Initialize FreshAI

```bash
# Initialize in current directory
python -m freshai init

# Or specify directory
python -m freshai init /path/to/investigation/workspace
```

This creates:
- `evidence/` - Evidence storage directory
- `cases/` - Case data and reports
- `config/` - Configuration files
- `.env` - Environment configuration

### Environment Configuration

Edit the `.env` file to customize settings:

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

## Verification

### Test Basic Installation

```bash
# Run the demo script
python demo.py

# Run basic tests
python tests/test_basic.py

# Or use pytest
pytest tests/
```

### Test Full Installation

```bash
# Test CLI
python -m freshai --help

# Test with sample case
python -m freshai init ./test_workspace
cd test_workspace
python -m freshai start-case TEST001 --description "Test case"
echo "Sample evidence text" > evidence/sample.txt
python -m freshai analyze TEST001 evidence/sample.txt
python -m freshai ask TEST001 "What evidence was found?"
python -m freshai report TEST001
```

### Test Ollama Integration

```bash
# Ensure Ollama is running
ollama serve

# Test in another terminal
python examples/basic_usage.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Missing dependencies
   pip install -r requirements.txt
   
   # Path issues
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Ollama Connection Issues**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

3. **GPU/CUDA Issues**
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Permission Issues**
   ```bash
   # Ensure directories are writable
   chmod -R 755 evidence/ cases/
   
   # Or use user installation
   pip install --user -r requirements.txt
   ```

### Performance Optimization

1. **Memory Usage**
   - Use smaller models for limited RAM
   - Set `TRANSFORMERS_DEVICE=cpu` for CPU-only inference
   - Enable model quantization in config

2. **Storage Optimization**
   - Set `TRANSFORMERS_CACHE_DIR` to external storage
   - Use model pruning for production deployments
   - Clean old case data regularly

3. **Network Optimization**
   - Use local Ollama instance for privacy
   - Configure proxy settings if needed
   - Cache model downloads

## Next Steps

After successful installation:

1. **Read the Documentation**
   - See `README.md` for full feature overview
   - Check `examples/` for usage examples
   - Review API documentation in code

2. **Start Your First Investigation**
   ```bash
   python -m freshai start-case CASE001 --description "My first case"
   ```

3. **Explore Advanced Features**
   - Custom tool development
   - Model fine-tuning
   - Integration with existing systems

4. **Join the Community**
   - Report issues on GitHub
   - Contribute improvements
   - Share use cases and feedback

## Support

For help and support:
- Check the troubleshooting section above
- Search existing GitHub issues
- Create a new issue with detailed information
- Join community discussions

Remember: FreshAI is designed to assist investigations but should complement, not replace, proper forensic procedures and human expertise.