"""Model configuration classes."""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Base model configuration."""
    
    name: str = Field(description="Model name")
    enabled: bool = Field(default=True, description="Whether model is enabled")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")


class OllamaConfig(BaseModel):
    """Configuration for Ollama server integration."""
    
    host: str = Field(default="localhost", description="Ollama server host")
    port: int = Field(default=11434, description="Ollama server port")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # Default models
    default_llm: str = Field(default="llama2", description="Default LLM model")
    default_vlm: str = Field(default="llava", description="Default VLM model")
    
    # Available models
    available_llms: List[str] = Field(
        default_factory=lambda: ["llama2", "codellama", "mistral", "phi"],
        description="Available LLM models"
    )
    available_vlms: List[str] = Field(
        default_factory=lambda: ["llava", "bakllava"],
        description="Available VLM models"
    )
    
    @property
    def base_url(self) -> str:
        """Get the base URL for Ollama API."""
        return f"http://{self.host}:{self.port}"
    
    @classmethod
    def load_from_env(cls) -> "OllamaConfig":
        """Load Ollama configuration from environment variables."""
        return cls(
            host=os.getenv("OLLAMA_HOST", "localhost"),
            port=int(os.getenv("OLLAMA_PORT", "11434")),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),
            default_llm=os.getenv("OLLAMA_DEFAULT_LLM", "llama2"),
            default_vlm=os.getenv("OLLAMA_DEFAULT_VLM", "llava"),
        )


class TransformersConfig(BaseModel):
    """Configuration for Transformers library models."""
    
    cache_dir: Optional[str] = Field(default=None, description="Model cache directory")
    device: str = Field(default="auto", description="Device to use for inference")
    
    # Text models
    text_models: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "bert-base": ModelConfig(
                name="bert-base-uncased",
                parameters={"max_length": 512}
            ),
            "distilbert": ModelConfig(
                name="distilbert-base-uncased",
                parameters={"max_length": 512}
            ),
        },
        description="Available text models"
    )
    
    # Vision models
    vision_models: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "clip": ModelConfig(
                name="openai/clip-vit-base-patch32",
                parameters={}
            ),
            "deit": ModelConfig(
                name="facebook/deit-base-distilled-patch16-224",
                parameters={}
            ),
        },
        description="Available vision models"
    )
    
    # Multimodal models
    multimodal_models: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "blip": ModelConfig(
                name="Salesforce/blip-image-captioning-base",
                parameters={}
            ),
        },
        description="Available multimodal models"
    )
    
    @classmethod
    def load_from_env(cls) -> "TransformersConfig":
        """Load Transformers configuration from environment variables."""
        return cls(
            cache_dir=os.getenv("TRANSFORMERS_CACHE_DIR"),
            device=os.getenv("TRANSFORMERS_DEVICE", "auto"),
        )