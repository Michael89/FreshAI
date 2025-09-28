"""Main configuration class for FreshAI."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .models import ModelConfig, OllamaConfig, TransformersConfig


class Config(BaseModel):
    """Main configuration for FreshAI."""
    
    # General settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Model configurations
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    transformers: TransformersConfig = Field(default_factory=TransformersConfig)
    
    # Tools and features
    enable_vision: bool = Field(default=True, description="Enable vision capabilities")
    enable_tools: bool = Field(default=True, description="Enable tool usage")
    max_context_length: int = Field(default=8192, description="Maximum context length")
    
    # Investigation specific settings
    evidence_storage_path: Path = Field(
        default=Path("./evidence"), 
        description="Path to store evidence files"
    )
    case_storage_path: Path = Field(
        default=Path("./cases"), 
        description="Path to store case files"
    )
    
    class Config:
        env_prefix = "FRESHAI_"
        
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables and .env file."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
            
        # Create directories if they don't exist
        evidence_path = Path(os.getenv("FRESHAI_EVIDENCE_STORAGE_PATH", "./evidence"))
        case_path = Path(os.getenv("FRESHAI_CASE_STORAGE_PATH", "./cases"))
        
        evidence_path.mkdir(exist_ok=True)
        case_path.mkdir(exist_ok=True)
        
        return cls(
            debug=os.getenv("FRESHAI_DEBUG", "false").lower() == "true",
            log_level=os.getenv("FRESHAI_LOG_LEVEL", "INFO"),
            enable_vision=os.getenv("FRESHAI_ENABLE_VISION", "true").lower() == "true",
            enable_tools=os.getenv("FRESHAI_ENABLE_TOOLS", "true").lower() == "true",
            max_context_length=int(os.getenv("FRESHAI_MAX_CONTEXT_LENGTH", "8192")),
            evidence_storage_path=evidence_path,
            case_storage_path=case_path,
            ollama=OllamaConfig.load_from_env(),
            transformers=TransformersConfig.load_from_env(),
        )