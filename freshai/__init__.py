"""
FreshAI - AI Assistant for Crime Investigators

A comprehensive AI platform that combines Large Language Models (LLM) and 
Vision Language Models (VLM) to assist crime investigators with their work.

Features:
- Ollama server integration for local LLM/VLM inference
- Transformers library support for open source models
- Tool system for evidence analysis
- Vision capabilities for image and video analysis
"""

__version__ = "0.1.0"
__author__ = "FreshAI Team"

# Import config first as it has fewer dependencies
from .config import Config

# Try to import core components, but handle missing dependencies gracefully
try:
    from .core import FreshAICore
    from .agent import InvestigatorAgent
    __all__ = ["InvestigatorAgent", "FreshAICore", "Config"]
except ImportError as e:
    # If optional dependencies are missing, only export Config
    import warnings
    warnings.warn(f"Some FreshAI components unavailable due to missing dependencies: {e}")
    __all__ = ["Config"]