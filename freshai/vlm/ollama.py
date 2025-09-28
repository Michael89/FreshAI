"""Ollama VLM integration."""

import base64
import json
import time
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import aiohttp

from ..core.base import VisionModel, ModelResponse


logger = logging.getLogger(__name__)


class OllamaVLM(VisionModel):
    """Ollama Vision Language Model integration."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.base_url = config.get("base_url", "http://localhost:11434") if config else "http://localhost:11434"
        self.timeout = config.get("timeout", 300) if config else 300
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> None:
        """Initialize the Ollama VLM connection."""
        if self._initialized:
            return
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        try:
            await self._check_model_availability()
            self._initialized = True
            logger.info(f"Ollama VLM {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama VLM: {e}")
            raise
    
    async def _check_model_availability(self) -> None:
        """Check if the VLM model is available."""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [model["name"] for model in data.get("models", [])]
                    
                    if self.model_name not in available_models:
                        logger.warning(f"VLM Model {self.model_name} not found. Available: {available_models}")
                        await self._pull_model()
                else:
                    raise Exception(f"Failed to connect to Ollama server: {response.status}")
        except Exception as e:
            raise Exception(f"Unable to check VLM model availability: {e}")
    
    async def _pull_model(self) -> None:
        """Pull the VLM model if not available."""
        logger.info(f"Pulling VLM model {self.model_name}...")
        
        payload = {"name": self.model_name}
        
        async with self.session.post(
            f"{self.base_url}/api/pull",
            json=payload
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            if data.get("status"):
                                logger.debug(f"VLM Pull progress: {data['status']}")
                        except json.JSONDecodeError:
                            continue
                logger.info(f"VLM Model {self.model_name} pulled successfully")
            else:
                raise Exception(f"Failed to pull VLM model: {response.status}")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text response (fallback for VLM)."""
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
                        metadata={"mode": "text_only"}
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama VLM API error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error generating VLM response: {e}")
            raise
    
    async def generate_from_image(
        self, 
        image_path: str, 
        prompt: str = "What do you see in this image?", 
        **kwargs
    ) -> ModelResponse:
        """Generate response from image and text prompt."""
        if not self._initialized:
            await self.initialize()
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        start_time = time.time()
        
        # Encode the image
        image_b64 = self._encode_image_to_base64(image_path)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
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
                            "image_path": image_path,
                            "prompt": prompt,
                            "mode": "vision",
                            "total_duration": data.get("total_duration"),
                            "eval_duration": data.get("eval_duration"),
                        }
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama VLM API error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error generating vision response: {e}")
            raise
    
    async def analyze_image(self, image_path: str, **kwargs) -> ModelResponse:
        """Analyze image and provide detailed description for crime investigation."""
        investigation_prompt = """
        Analyze this image carefully for potential evidence or items of interest in a criminal investigation. 
        Please describe:
        1. All visible objects, people, and their positions
        2. Any potential evidence (weapons, drugs, documents, etc.)
        3. Environmental details (location type, lighting, time indicators)
        4. Any anomalies or suspicious elements
        5. Quality and clarity of the image for forensic purposes
        
        Be detailed and objective in your analysis.
        """
        
        return await self.generate_from_image(image_path, investigation_prompt, **kwargs)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
        self._initialized = False