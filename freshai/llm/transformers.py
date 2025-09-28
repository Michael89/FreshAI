"""Transformers LLM integration."""

import time
import logging
from typing import Any, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ..core.base import BaseModel, ModelResponse


logger = logging.getLogger(__name__)


class TransformersLLM(BaseModel):
    """Transformers-based Language Model integration."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.device = config.get("device", "auto") if config else "auto"
        self.max_length = config.get("parameters", {}).get("max_length", 512) if config else 512
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    async def initialize(self) -> None:
        """Initialize the Transformers model."""
        if self._initialized:
            return
            
        try:
            logger.info(f"Loading Transformers model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.config.get("cache_dir")
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load as causal LM first, fallback to other types
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    cache_dir=self.config.get("cache_dir"),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device if device != "cpu" else None
                )
                model_type = "text-generation"
            except Exception:
                # Fallback to a general pipeline approach
                logger.info("Failed to load as causal LM, trying pipeline approach")
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.model_name if not self.tokenizer else self.tokenizer,
                    device=0 if device == "cuda" else -1,
                    trust_remote_code=True
                )
                model_type = "pipeline"
            
            self._initialized = True
            logger.info(f"Transformers LLM {self.model_name} initialized successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers LLM: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from text prompt."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            if self.pipeline:
                # Use pipeline approach
                result = self.pipeline(
                    prompt,
                    max_length=kwargs.get("max_tokens", self.max_length),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                
                if result and len(result) > 0:
                    generated_text = result[0]["generated_text"]
                    # Remove the original prompt from the response
                    response_text = generated_text[len(prompt):].strip()
                else:
                    response_text = ""
                    
                tokens_used = None
                
            else:
                # Use model + tokenizer approach
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                
                if torch.cuda.is_available() and self.model.device.type == "cuda":
                    inputs = inputs.to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=kwargs.get("max_tokens", self.max_length),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode the response
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_text = full_response[len(prompt):].strip()
                tokens_used = len(outputs[0])
            
            processing_time = time.time() - start_time
            
            return ModelResponse(
                content=response_text,
                model_name=self.model_name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                metadata={
                    "device": str(self.model.device) if self.model else "pipeline",
                    "max_length": self.max_length,
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._initialized = False