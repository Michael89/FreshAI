"""Base agent class with tool calling capabilities."""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool messages

    def to_dict(self):
        """Convert message to dictionary."""
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Tool:
    """Represents a tool that the agent can use."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]  # JSON schema for parameters

    def to_dict(self):
        """Convert tool to dictionary for LLM consumption."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class BaseAgent(ABC):
    """Base agent class with tool calling capabilities."""

    def __init__(
        self,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """Initialize the agent.

        Args:
            system_prompt: System prompt for the agent
            tools: List of available tools
            max_iterations: Maximum number of tool calling iterations
            verbose: Whether to enable verbose logging
        """
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.conversation_history: List[Message] = []
        self.iteration_count = 0

        # Configure logging
        if self.verbose:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        # Add system message
        self.conversation_history.append(
            Message(role="system", content=system_prompt)
        )

        logger.info(f"Initialized {self.__class__.__name__} with {len(self.tools)} tools")
        if self.tools:
            logger.info(f"Available tools: {[t.name for t in self.tools]}")

    @abstractmethod
    async def _call_llm(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None
    ) -> Message:
        """Call the underlying LLM.

        Args:
            messages: Conversation history
            tools: Available tools for this call

        Returns:
            Response message from LLM
        """
        pass

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Tool execution result as string
        """
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {json.dumps(arguments, indent=2)}")

        if tool_name not in self.tool_map:
            error = f"Tool '{tool_name}' not found"
            logger.error(error)
            return f"Error: {error}"

        tool = self.tool_map[tool_name]

        try:
            result = tool.function(**arguments)
            logger.info(f"Tool '{tool_name}' executed successfully")
            logger.debug(f"Tool result: {result}")
            return str(result)
        except Exception as e:
            error = f"Error executing tool '{tool_name}': {str(e)}\n{traceback.format_exc()}"
            logger.error(error)
            return f"Error: {error}"

    def _parse_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from LLM response.

        This is a fallback parser for models that don't have native tool calling.
        Looks for JSON blocks in the response that match tool call format.

        Args:
            content: LLM response content

        Returns:
            List of tool calls if found, None otherwise
        """
        try:
            # Look for JSON-like tool call patterns
            import re

            # Pattern 1: Explicit tool call JSON
            pattern1 = r'\{[^{}]*"tool"[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
            matches = re.findall(pattern1, content, re.DOTALL)

            if matches:
                tool_calls = []
                for match in matches:
                    try:
                        call = json.loads(match)
                        if "name" in call and "arguments" in call:
                            tool_calls.append({
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call["arguments"])
                                }
                            })
                    except:
                        pass

                if tool_calls:
                    logger.debug(f"Parsed {len(tool_calls)} tool calls from content")
                    return tool_calls

            # Pattern 2: Look for function call syntax
            pattern2 = r'(?:call|use|execute)_(\w+)\((.*?)\)'
            matches2 = re.findall(pattern2, content, re.IGNORECASE)

            if matches2:
                tool_calls = []
                for name, args_str in matches2:
                    if name in self.tool_map:
                        try:
                            # Try to parse arguments
                            arguments = {}
                            if args_str.strip():
                                # Simple key=value parsing
                                for arg in args_str.split(','):
                                    if '=' in arg:
                                        key, value = arg.split('=', 1)
                                        arguments[key.strip()] = value.strip().strip('"\'')

                            tool_calls.append({
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments)
                                }
                            })
                        except:
                            pass

                if tool_calls:
                    logger.debug(f"Parsed {len(tool_calls)} tool calls from function syntax")
                    return tool_calls

        except Exception as e:
            logger.debug(f"Failed to parse tool calls: {e}")

        return None

    async def run(self, user_prompt: str) -> str:
        """Run the agent with a user prompt.

        Args:
            user_prompt: Initial user prompt

        Returns:
            Final response from the agent
        """
        logger.info("=" * 80)
        logger.info("Starting agent execution")
        logger.info(f"User prompt: {user_prompt}")
        logger.info("=" * 80)

        # Add user message
        self.conversation_history.append(
            Message(role="user", content=user_prompt)
        )

        self.iteration_count = 0

        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            logger.info(f"\n--- Iteration {self.iteration_count}/{self.max_iterations} ---")

            # Call LLM with current conversation and tools
            response = await self._call_llm(
                self.conversation_history,
                self.tools if self.tools else None
            )

            # Add assistant response to history
            self.conversation_history.append(response)

            # Check if response contains tool calls
            if response.tool_calls:
                logger.info(f"Agent requested {len(response.tool_calls)} tool call(s)")

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    tool_id = tool_call.get("id", f"call_{self.iteration_count}")

                    # Execute tool
                    result = self._execute_tool(tool_name, tool_args)

                    # Add tool result to conversation
                    tool_message = Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    self.conversation_history.append(tool_message)

                # Continue conversation after tool execution
                continue

            else:
                # No tool calls, agent has finished
                logger.info("Agent finished without requesting tools")
                logger.debug(f"Final response content: {repr(response.content[:500])}")

                if not response.content.strip():
                    logger.warning("Final response is empty! Creating summary from tool results.")
                    # Fallback: create a summary from recent tool results
                    tool_results = []
                    for msg in reversed(self.conversation_history):
                        if msg.role == "tool":
                            tool_results.append(f"Tool '{msg.name}' result: {msg.content}")
                        elif msg.role == "user":
                            break  # Stop at the last user message

                    if tool_results:
                        fallback_response = f"I executed the requested tools and here are the results:\n\n" + "\n\n".join(reversed(tool_results))
                        logger.info("Created fallback response from tool results")
                        return fallback_response
                    else:
                        return "Task completed, but no specific results to report."

                logger.info("=" * 80)
                logger.info("Agent execution completed")
                logger.info(f"Total iterations: {self.iteration_count}")
                logger.info("=" * 80)
                return response.content

        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        final_message = "I've reached the maximum number of iterations. Here's what I found so far:\n\n"

        # Get the last assistant message
        for msg in reversed(self.conversation_history):
            if msg.role == "assistant" and not msg.tool_calls:
                final_message += msg.content
                break

        logger.info("=" * 80)
        logger.info("Agent execution completed (max iterations)")
        logger.info("=" * 80)

        return final_message

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return [msg.to_dict() for msg in self.conversation_history]

    def reset(self):
        """Reset the agent state."""
        logger.info("Resetting agent state")
        self.conversation_history = [
            Message(role="system", content=self.system_prompt)
        ]
        self.iteration_count = 0