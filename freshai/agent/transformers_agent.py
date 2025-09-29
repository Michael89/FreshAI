"""Transformers-based agent with tool calling capabilities."""
import json
import logging
from typing import Any, Dict, List, Optional
import torch
from freshai.agent.base_agent import BaseAgent, Message, Tool

logger = logging.getLogger(__name__)


class TransformersAgent(BaseAgent):
    """Agent that uses Hugging Face Transformers for LLM inference with tool calling."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        system_prompt: str = "You are a helpful AI assistant.",
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """Initialize the Transformers agent.

        Args:
            model_name: Hugging Face model name or path
            system_prompt: System prompt for the agent
            tools: List of available tools
            max_iterations: Maximum number of tool calling iterations
            verbose: Whether to enable verbose logging
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
        """
        super().__init__(system_prompt, tools, max_iterations, verbose)

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing TransformersAgent with model: {model_name}")
        logger.info(f"Device: {self.device}")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._lazy_load_model()

    def _lazy_load_model(self):
        """Lazy load the model and tokenizer."""
        if self.model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                logger.info(f"Loading model: {self.model_name}")

                # Configure quantization
                quantization_config = None
                if self.load_in_8bit:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("Loading model in 8-bit precision")
                elif self.load_in_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    logger.info("Loading model in 4-bit precision")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
                }

                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config

                if self.device == "cpu":
                    model_kwargs["low_cpu_mem_usage"] = True

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )

                if not (self.load_in_8bit or self.load_in_4bit):
                    self.model = self.model.to(self.device)

                logger.info("Model loaded successfully")

            except ImportError as e:
                logger.error(f"Failed to import transformers: {e}")
                logger.error("Please install transformers: pip install transformers torch")
                raise
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def _format_messages_for_model(self, messages: List[Message]) -> str:
        """Format messages for the model.

        Args:
            messages: Conversation history

        Returns:
            Formatted prompt string
        """
        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted_messages = []
                for msg in messages:
                    if msg.role in ["system", "user", "assistant"]:
                        formatted_messages.append({
                            "role": msg.role,
                            "content": msg.content
                        })
                    elif msg.role == "tool":
                        # Format tool results as user messages
                        formatted_messages.append({
                            "role": "user",
                            "content": f"Tool '{msg.name}' returned: {msg.content}"
                        })

                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.debug(f"Failed to apply chat template: {e}")

        # Fallback: Simple formatting
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
            elif msg.role == "tool":
                prompt += f"Tool Result ({msg.name}): {msg.content}\n\n"

        prompt += "Assistant: "
        return prompt

    def _format_tools_for_prompt(self, tools: List[Tool]) -> str:
        """Format tools as part of the prompt.

        Args:
            tools: List of available tools

        Returns:
            Formatted tool descriptions
        """
        if not tools:
            return ""

        tool_desc = "\n\nYou have access to the following tools:\n\n"

        for tool in tools:
            tool_desc += f"Tool: {tool.name}\n"
            tool_desc += f"Description: {tool.description}\n"
            tool_desc += f"Parameters: {json.dumps(tool.parameters, indent=2)}\n\n"

        tool_desc += """
To use a tool, respond with a JSON object in this exact format:
{
    "tool": true,
    "name": "<tool_name>",
    "arguments": {
        "<param_name>": "<param_value>"
    }
}

After receiving the tool result, continue with your response or call another tool if needed.
Only output the JSON when calling a tool, nothing else.
"""
        return tool_desc

    async def _call_llm(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None
    ) -> Message:
        """Call the Transformers model.

        Args:
            messages: Conversation history
            tools: Available tools for this call

        Returns:
            Response message from model
        """
        # Ensure model is loaded
        self._lazy_load_model()

        logger.debug(f"Calling model with {len(messages)} messages")

        # Add tools to system message if available
        modified_messages = messages.copy()
        if tools and modified_messages:
            if modified_messages[0].role == "system":
                modified_messages[0] = Message(
                    role="system",
                    content=modified_messages[0].content + self._format_tools_for_prompt(tools),
                    timestamp=modified_messages[0].timestamp
                )
            else:
                # Insert system message with tools
                modified_messages.insert(0, Message(
                    role="system",
                    content=self.system_prompt + self._format_tools_for_prompt(tools)
                ))

        # Format messages for model
        prompt = self._format_messages_for_model(modified_messages)
        logger.debug(f"Prompt (first 500 chars): {prompt[:500]}...")

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            logger.debug(f"Model response: {response[:500]}...")

            # Check for tool calls
            tool_calls = None
            if tools:
                # Look for JSON tool calls in response
                try:
                    import re
                    json_pattern = r'\{[^{}]*"tool"\s*:\s*true[^{}]*\}'
                    json_matches = re.findall(json_pattern, response, re.DOTALL)

                    if json_matches:
                        tool_calls = []
                        for match in json_matches:
                            call_data = json.loads(match)
                            if call_data.get("tool") and "name" in call_data and "arguments" in call_data:
                                tool_calls.append({
                                    "id": f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": call_data["name"],
                                        "arguments": json.dumps(call_data["arguments"])
                                    }
                                })
                                logger.info(f"Detected tool call: {call_data['name']}")
                except Exception as e:
                    logger.debug(f"No JSON tool calls found: {e}")

                # Fallback to base parser
                if not tool_calls:
                    tool_calls = self._parse_tool_calls(response)

            return Message(
                role="assistant",
                content=response,
                tool_calls=tool_calls
            )

        except Exception as e:
            error = f"Error during model inference: {str(e)}"
            logger.error(error)
            return Message(role="assistant", content=f"Error: {error}")

    def clear_cache(self):
        """Clear model from memory."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")