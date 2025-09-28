"""Core FreshAI system orchestrator."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..config import Config
from ..llm import OllamaLLM, TransformersLLM
from ..vlm import OllamaVLM, TransformersVLM
from ..tools import ToolRegistry
from ..mcp import MCPClient
from .base import BaseModel, ModelResponse


logger = logging.getLogger(__name__)


class FreshAICore:
    """Core orchestrator for FreshAI system."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load_from_env()
        
        # Model instances
        self.llm_models: Dict[str, BaseModel] = {}
        self.vlm_models: Dict[str, BaseModel] = {}
        
        # Tool registry
        self.tool_registry = ToolRegistry()
        
        # MCP client
        self.mcp_client = MCPClient()
        
        self._initialized = False
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def initialize(self) -> None:
        """Initialize the FreshAI core system."""
        if self._initialized:
            return
            
        logger.info("Initializing FreshAI Core...")
        
        try:
            # Initialize Ollama models
            await self._initialize_ollama_models()
            
            # Initialize Transformers models
            await self._initialize_transformers_models()
            
            # Initialize tools
            await self._initialize_tools()
            
            # Initialize MCP servers
            await self._initialize_mcp_servers()
            
            self._initialized = True
            logger.info("FreshAI Core initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FreshAI Core: {e}")
            raise
    
    async def _initialize_ollama_models(self) -> None:
        """Initialize Ollama-based models."""
        try:
            # Initialize default LLM
            llm = OllamaLLM(
                model_name=self.config.ollama.default_llm,
                config=self.config.ollama.dict()
            )
            await llm.initialize()
            self.llm_models["ollama_default"] = llm
            
            # Initialize default VLM if vision is enabled
            if self.config.enable_vision:
                vlm = OllamaVLM(
                    model_name=self.config.ollama.default_vlm,
                    config=self.config.ollama.dict()
                )
                await vlm.initialize()
                self.vlm_models["ollama_default"] = vlm
                
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama models: {e}")
    
    async def _initialize_transformers_models(self) -> None:
        """Initialize Transformers-based models."""
        try:
            # Initialize a basic text model
            if self.config.transformers.text_models:
                model_key = next(iter(self.config.transformers.text_models))
                model_config = self.config.transformers.text_models[model_key]
                
                llm = TransformersLLM(
                    model_name=model_config.name,
                    config=model_config.dict()
                )
                await llm.initialize()
                self.llm_models["transformers_default"] = llm
            
            # Initialize vision model if enabled
            if self.config.enable_vision and self.config.transformers.vision_models:
                model_key = next(iter(self.config.transformers.vision_models))
                model_config = self.config.transformers.vision_models[model_key]
                
                vlm = TransformersVLM(
                    model_name=model_config.name,
                    config=model_config.dict()
                )
                await vlm.initialize()
                self.vlm_models["transformers_default"] = vlm
                
        except Exception as e:
            logger.warning(f"Failed to initialize Transformers models: {e}")
    
    async def _initialize_tools(self) -> None:
        """Initialize investigation tools."""
        if not self.config.enable_tools:
            return
            
        # Register default tools for crime investigation
        from ..tools.evidence import EvidenceAnalyzer
        from ..tools.image import ImageAnalyzer
        from ..tools.text import TextAnalyzer
        
        self.tool_registry.register("evidence_analyzer", EvidenceAnalyzer())
        
        if self.config.enable_vision:
            self.tool_registry.register("image_analyzer", ImageAnalyzer())
            
        self.tool_registry.register("text_analyzer", TextAnalyzer())
    
    async def _initialize_mcp_servers(self) -> None:
        """Initialize MCP servers and register their tools."""
        if not self.config.mcp.enable_mcp:
            logger.info("MCP support is disabled")
            return
        
        logger.info("Initializing MCP servers...")
        
        for server_name, server_config in self.config.mcp.servers.items():
            if not server_config.enabled:
                logger.info(f"MCP server {server_name} is disabled")
                continue
            
            try:
                success = await self.mcp_client.add_server(server_config)
                if success:
                    logger.info(f"Successfully initialized MCP server: {server_name}")
                    
                    # Register MCP tools with the tool registry
                    mcp_tools = self.mcp_client.get_all_tools()
                    for tool_name, mcp_tool in mcp_tools.items():
                        if tool_name.startswith(f"{server_name}_"):
                            self.tool_registry.register(tool_name, mcp_tool)
                            logger.info(f"Registered MCP tool: {tool_name}")
                else:
                    logger.warning(f"Failed to initialize MCP server: {server_name}")
                    
            except Exception as e:
                logger.error(f"Error initializing MCP server {server_name}: {e}")
    
    
    async def generate_response(
        self, 
        prompt: str, 
        model_type: str = "llm",
        model_name: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using specified model."""
        if not self._initialized:
            await self.initialize()
        
        if model_type == "llm":
            models = self.llm_models
            default_model = "ollama_default"
        elif model_type == "vlm":
            models = self.vlm_models
            default_model = "ollama_default"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_name = model_name or default_model
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not available")
        
        model = models[model_name]
        return await model.generate(prompt, **kwargs)
    
    async def analyze_image(
        self, 
        image_path: str, 
        prompt: str = "",
        model_name: Optional[str] = None
    ) -> ModelResponse:
        """Analyze image using vision model."""
        if not self.config.enable_vision:
            raise ValueError("Vision capabilities are disabled")
        
        if not self._initialized:
            await self.initialize()
        
        model_name = model_name or "ollama_default"
        
        if model_name not in self.vlm_models:
            raise ValueError(f"Vision model {model_name} not available")
        
        model = self.vlm_models[model_name]
        
        if hasattr(model, 'generate_from_image'):
            return await model.generate_from_image(image_path, prompt)
        else:
            return await model.analyze_image(image_path)
    
    async def use_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use an investigation tool."""
        if not self.config.enable_tools:
            raise ValueError("Tools are disabled")
        
        return await self.tool_registry.use_tool(tool_name, parameters)
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models."""
        return {
            "llm": list(self.llm_models.keys()),
            "vlm": list(self.vlm_models.keys())
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.tool_registry.get_available_tools()
    
    async def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up FreshAI Core...")
        
        # Stop MCP servers
        await self.mcp_client.stop_all_servers()
        
        # Cleanup models
        for model in self.llm_models.values():
            await model.cleanup()
        
        for model in self.vlm_models.values():
            await model.cleanup()
        
        self._initialized = False
        logger.info("FreshAI Core cleanup completed")