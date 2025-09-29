"""Ollama-based agent with tool calling capabilities."""
import json
import logging
import httpx
from typing import Any, Dict, List, Optional
from freshai.agent.base_agent import BaseAgent, Message, Tool

logger = logging.getLogger(__name__)


class OllamaAgent(BaseAgent):
    """Agent that uses Ollama for LLM inference with tool calling."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        system_prompt: str = "You are a helpful AI assistant.",
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        temperature: float = 0.7
    ):
        """Initialize the Ollama agent.

        Args:
            model: Ollama model to use
            base_url: Ollama API base URL
            system_prompt: System prompt for the agent
            tools: List of available tools
            max_iterations: Maximum number of tool calling iterations
            verbose: Whether to enable verbose logging
            temperature: Sampling temperature
        """
        super().__init__(system_prompt, tools, max_iterations, verbose)

        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature

        logger.info(f"Initialized OllamaAgent with model: {model}")
        logger.info(f"Ollama base URL: {base_url}")

    def _format_messages_for_ollama(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Ollama API.

        Args:
            messages: Conversation history

        Returns:
            Formatted messages for Ollama
        """
        formatted = []

        for msg in messages:
            if msg.role in ["system", "user", "assistant"]:
                formatted.append({
                    "role": msg.role,
                    "content": msg.content
                })
            elif msg.role == "tool":
                # Format tool results as assistant messages
                formatted.append({
                    "role": "assistant",
                    "content": f"Tool '{msg.name}' result:\n{msg.content}"
                })

        return formatted

    def _format_tools_for_prompt(self, tools: List[Tool]) -> str:
        """Format tools as part of the system prompt for models without native tool support.

        Args:
            tools: List of available tools

        Returns:
            Formatted tool descriptions
        """
        if not tools:
            return ""

        tool_desc = "\n\nYou have access to the following tools:\n\n"

        for tool in tools:
            tool_desc += f"**{tool.name}**: {tool.description}\n"
            tool_desc += f"Parameters: {json.dumps(tool.parameters, indent=2)}\n\n"

        tool_desc += """
To use a tool, respond ONLY with a JSON object in this exact format and nothing else:
{
    "tool": true,
    "name": "tool_name",
    "arguments": {
        "param1": "value1",
        "param2": "value2"
    }
}

IMPORTANT:
1. When calling a tool, output ONLY the JSON object, no other text.
2. After receiving the tool result, ALWAYS provide a helpful response summarizing what you found.
3. Do not leave responses empty - always explain what the tool showed you.
4. REMEMBER CONTEXT: Use information from previous tool calls and conversation history to answer questions.
5. GIVE DIRECT ANSWERS: When asked specific questions, provide clear yes/no answers based on available evidence.
"""
        return tool_desc

    async def _call_llm(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None
    ) -> Message:
        """Call Ollama API.

        Args:
            messages: Conversation history
            tools: Available tools for this call

        Returns:
            Response message from Ollama
        """
        logger.debug(f"Calling Ollama with {len(messages)} messages")

        # Format messages for Ollama
        formatted_messages = self._format_messages_for_ollama(messages)

        # If tools are available, add them to the system prompt
        # (since most Ollama models don't have native tool calling)
        if tools and formatted_messages:
            if formatted_messages[0]["role"] == "system":
                formatted_messages[0]["content"] += self._format_tools_for_prompt(tools)
            else:
                # Insert system message with tools
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": self.system_prompt + self._format_tools_for_prompt(tools)
                })

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "stream": False
        }

        logger.debug(f"Request payload: {json.dumps(payload, indent=2)[:500]}...")

        try:
            # Make API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )

                if response.status_code != 200:
                    error = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error)
                    return Message(role="assistant", content=f"Error: {error}")

                result = response.json()
                content = result.get("message", {}).get("content", "")

                logger.debug(f"Ollama response: {content[:500]}...")
                logger.debug(f"Full response for tool detection: {repr(content[:1000])}")

                # Try to detect tool calls in the response
                tool_calls = None
                if tools:
                    # Check for JSON tool call in the response
                    try:
                        # Look for JSON blocks (including those in code blocks)
                        import re

                        # Remove markdown code blocks if present
                        clean_content = content
                        code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
                        code_blocks = re.findall(code_block_pattern, content, re.DOTALL)
                        if code_blocks:
                            clean_content = code_blocks[0]

                        # Try multiple JSON patterns
                        json_patterns = [
                            r'\{[^{}]*"tool"\s*:\s*true[^{}]*\}',
                            r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"name"\s*:\s*"[^"]+"\s*[^{}]*\}',
                            r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
                        ]

                        for pattern in json_patterns:
                            json_matches = re.findall(pattern, clean_content, re.DOTALL)
                            if json_matches:
                                tool_calls = []
                                for match in json_matches:
                                    try:
                                        call_data = json.loads(match)
                                        # Check different formats
                                        if "name" in call_data and "arguments" in call_data:
                                            tool_calls.append({
                                                "id": f"call_{len(tool_calls)}",
                                                "type": "function",
                                                "function": {
                                                    "name": call_data["name"],
                                                    "arguments": json.dumps(call_data["arguments"])
                                                }
                                            })
                                            logger.info(f"Detected tool call: {call_data['name']}")
                                    except json.JSONDecodeError:
                                        pass

                                if tool_calls:
                                    break
                    except Exception as e:
                        logger.debug(f"Error detecting tool calls: {e}")

                    # Fallback: try the base parser
                    if not tool_calls:
                        tool_calls = self._parse_tool_calls(content)

                return Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls
                )

        except httpx.TimeoutException:
            error = "Ollama API request timed out"
            logger.error(error)
            return Message(role="assistant", content=f"Error: {error}")
        except Exception as e:
            error = f"Error calling Ollama API: {str(e)}"
            logger.error(error)
            return Message(role="assistant", content=f"Error: {error}")

    async def check_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama.

        Returns:
            True if model is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                    is_available = self.model in available_models

                    if is_available:
                        logger.info(f"Model '{self.model}' is available")
                    else:
                        logger.warning(f"Model '{self.model}' not found. Available models: {available_models}")

                    return is_available

        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")

        return False

    async def pull_model(self) -> bool:
        """Pull the specified model if not available.

        Returns:
            True if model is ready, False otherwise
        """
        logger.info(f"Pulling model '{self.model}'...")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model, "stream": False}
                )

                if response.status_code == 200:
                    logger.info(f"Successfully pulled model '{self.model}'")
                    return True
                else:
                    logger.error(f"Failed to pull model: {response.text}")

        except Exception as e:
            logger.error(f"Error pulling model: {e}")

        return False