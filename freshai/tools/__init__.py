"""Investigation tools for FreshAI."""

from .registry import ToolRegistry
from .evidence import EvidenceAnalyzer
from .text import TextAnalyzer

# Import ImageAnalyzer only if OpenCV is available
try:
    from .image import ImageAnalyzer
    __all__ = ["ToolRegistry", "EvidenceAnalyzer", "ImageAnalyzer", "TextAnalyzer"]
except ImportError:
    import warnings
    warnings.warn("ImageAnalyzer not available - OpenCV not found")
    __all__ = ["ToolRegistry", "EvidenceAnalyzer", "TextAnalyzer"]