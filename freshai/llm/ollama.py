"""Ollama LLM integration."""

import json
import time
import logging
from typing import Any, Dict, Optional
import aiohttp

from ..core.base import BaseModel, ModelResponse, ToolCapableModel


logger = logging.getLogger(__name__)


class OllamaLLM(ToolCapableModel):
    """Ollama Language Model integration."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.base_url = config.get("base_url", "http://localhost:11434") if config else "http://localhost:11434"
        self.timeout = config.get("timeout", 300) if config else 300
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> None:
        """Initialize the Ollama connection."""
        if self._initialized:
            return
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # Test connection and model availability
        try:
            await self._check_model_availability()
            self._initialized = True
            logger.info(f"Ollama LLM {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise
    
    async def _check_model_availability(self) -> None:
        """Check if the model is available on Ollama server."""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [model["name"] for model in data.get("models", [])]
                    
                    if self.model_name not in available_models:
                        logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                        # Try to pull the model
                        await self._pull_model()
                else:
                    raise Exception(f"Failed to connect to Ollama server: {response.status}")
        except Exception as e:
            raise Exception(f"Unable to check model availability: {e}")
    
    async def _pull_model(self) -> None:
        """Pull the model if it's not available."""
        logger.info(f"Pulling model {self.model_name}...")
        
        payload = {"name": self.model_name}
        
        async with self.session.post(
            f"{self.base_url}/api/pull",
            json=payload
        ) as response:
            if response.status == 200:
                # Stream the pull progress
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            if data.get("status"):
                                logger.debug(f"Pull progress: {data['status']}")
                        except json.JSONDecodeError:
                            continue
                logger.info(f"Model {self.model_name} pulled successfully")
            else:
                raise Exception(f"Failed to pull model: {response.status}")
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from text prompt."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "num_predict": kwargs.get("max_tokens", 512),
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    processing_time = time.time() - start_time
                    
                    return ModelResponse(
                        content=data.get("response", ""),
                        model_name=self.model_name,
                        tokens_used=data.get("eval_count"),
                        processing_time=processing_time,
                        metadata={
                            "total_duration": data.get("total_duration"),
                            "eval_duration": data.get("eval_duration"),
                            "prompt_eval_count": data.get("prompt_eval_count"),
                        }
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def call_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        **kwargs
    ) -> ModelResponse:
        """Call a tool using function calling capabilities."""
        # For now, implement basic tool calling through prompt engineering
        tool_prompt = f"""
You are an AI assistant with access to tools. Use the following tool:

Tool: {tool_name}
Parameters: {json.dumps(parameters)}

Please analyze the request and provide a response based on using this tool.
"""
        
        return await self.generate(tool_prompt, **kwargs)
    
    def get_available_tools(self) -> list[str]:
        """Get list of available tools (placeholder implementation)."""
        return ["evidence_analyzer", "image_analyzer", "text_analyzer"]
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
        self._initialized = False