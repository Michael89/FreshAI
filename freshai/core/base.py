"""Base classes and interfaces for FreshAI models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel


class ModelResponse(PydanticBaseModel):
    """Standard response format for all models."""
    
    content: str
    model_name: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = {}
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None


class BaseModel(ABC):
    """Base class for all AI models in FreshAI."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model for inference."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from text prompt."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup model resources."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized


class VisionModel(BaseModel):
    """Base class for vision-capable models."""
    
    @abstractmethod
    async def generate_from_image(
        self, 
        image_path: str, 
        prompt: str = "", 
        **kwargs
    ) -> ModelResponse:
        """Generate response from image input."""
        pass
    
    @abstractmethod
    async def analyze_image(self, image_path: str, **kwargs) -> ModelResponse:
        """Analyze image and return description."""
        pass


class ToolCapableModel(BaseModel):
    """Base class for models that can use tools."""
    
    @abstractmethod
    async def call_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        **kwargs
    ) -> ModelResponse:
        """Call a specific tool with given parameters."""
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        pass