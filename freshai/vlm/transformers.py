"""Transformers VLM integration."""

import time
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration

from ..core.base import VisionModel, ModelResponse


logger = logging.getLogger(__name__)


class TransformersVLM(VisionModel):
    """Transformers-based Vision Language Model integration."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.device = config.get("device", "auto") if config else "auto"
        
        self.processor = None
        self.model = None
        self.model_type = None
        
    async def initialize(self) -> None:
        """Initialize the Transformers VLM."""
        if self._initialized:
            return
            
        try:
            logger.info(f"Loading Transformers VLM: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load different model types based on model name
            if "blip" in self.model_name.lower():
                self.processor = BlipProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.config.get("cache_dir")
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.config.get("cache_dir"),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self.model_type = "blip"
            else:
                # Try generic approach
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        cache_dir=self.config.get("cache_dir")
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.config.get("cache_dir"),
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32
                    )
                    self.model_type = "auto"
                except Exception as e:
                    logger.error(f"Failed to load with AutoProcessor: {e}")
                    raise
            
            # Move model to device
            if device != "cpu":
                self.model = self.model.to(device)
            
            self._initialized = True
            logger.info(f"Transformers VLM {self.model_name} initialized successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers VLM: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise Exception(f"Failed to load image {image_path}: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text response (fallback for VLM)."""
        # For VLMs, text-only generation might not be supported
        # Return a helpful message
        return ModelResponse(
            content="This is a vision model. Please use generate_from_image() or analyze_image() methods.",
            model_name=self.model_name,
            metadata={"mode": "text_fallback"}
        )
    
    async def generate_from_image(
        self, 
        image_path: str, 
        prompt: str = "What do you see in this image?", 
        **kwargs
    ) -> ModelResponse:
        """Generate response from image and text prompt."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            image = self._load_image(image_path)
            
            if self.model_type == "blip":
                # BLIP-specific processing
                if prompt:
                    # Conditional generation with prompt
                    inputs = self.processor(image, prompt, return_tensors="pt")
                else:
                    # Unconditional caption generation
                    inputs = self.processor(image, return_tensors="pt")
                
                if torch.cuda.is_available() and self.model.device.type == "cuda":
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=kwargs.get("max_tokens", 150),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True,
                    )
                
                response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # Generic auto processor approach
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                if torch.cuda.is_available() and self.model.device.type == "cuda":
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=kwargs.get("max_tokens", 150),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True,
                    )
                
                response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return ModelResponse(
                content=response_text,
                model_name=self.model_name,
                tokens_used=len(outputs[0]) if 'outputs' in locals() else None,
                processing_time=processing_time,
                metadata={
                    "image_path": image_path,
                    "prompt": prompt,
                    "mode": "vision",
                    "model_type": self.model_type,
                    "device": str(self.model.device)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating vision response: {e}")
            raise
    
    async def analyze_image(self, image_path: str, **kwargs) -> ModelResponse:
        """Analyze image for crime investigation purposes."""
        investigation_prompt = """
        Describe this image in detail, focusing on:
        - Objects and people visible
        - Potential evidence or items of interest
        - Environmental context and setting
        - Any notable features for investigation purposes
        """
        
        # For BLIP models, we might need to use unconditional captioning
        if self.model_type == "blip":
            return await self.generate_from_image(image_path, "", **kwargs)
        else:
            return await self.generate_from_image(image_path, investigation_prompt, **kwargs)
    
    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._initialized = False