#!/usr/bin/env python
"""Simple test script for Gemma3 agent with tool calling."""
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from freshai.agent.base_agent import Tool
from freshai.agent.ollama_agent import OllamaAgent
from freshai.agent.tools import create_bash_tool, create_web_search_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)


async def main():
    """Test Gemma3 with tool calling."""
    print("\n" + "=" * 60)
    print("Testing Gemma3 Agent with Tools")
    print("=" * 60)

    # Create tools
    bash_tool_spec = create_bash_tool(safe_mode=True)
    bash_tool = Tool(
        name=bash_tool_spec["name"],
        description=bash_tool_spec["description"],
        function=bash_tool_spec["function"],
        parameters=bash_tool_spec["parameters"]
    )

    web_tool_spec = create_web_search_tool(async_mode=False)
    web_tool = Tool(
        name=web_tool_spec["name"],
        description=web_tool_spec["description"],
        function=web_tool_spec["function"],
        parameters=web_tool_spec["parameters"]
    )

    # System prompt optimized for Gemma3
    system_prompt = """You are a helpful AI assistant with access to tools.

IMPORTANT INSTRUCTIONS:
1. When you need to use a tool, output ONLY a JSON object with the tool call.
2. Do not add any text before or after the JSON when calling a tool.
3. After receiving tool results, provide a natural response to the user.

Available tools will be described below."""

    # Create agent
    agent = OllamaAgent(
        model="gemma3:latest",
        system_prompt=system_prompt,
        tools=[bash_tool, web_tool],
        verbose=True,
        temperature=0.3,  # Lower temperature for more consistent tool calling
        max_iterations=5
    )

    # Check if model is available
    if not await agent.check_model_availability():
        print(f"\nModel '{agent.model}' not available. Please pull it first:")
        print(f"  ollama pull {agent.model.split(':')[0]}")
        return

    # Test with simple prompts
    test_prompts = [
        "List all Python files in the current directory",
        "What is Python and when was it created?",
        "Show me the current working directory"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}: {prompt}")
        print('=' * 60)

        try:
            response = await agent.run(prompt)
            print(f"\n{'=' * 40}")
            print("FINAL RESPONSE:")
            print('=' * 40)
            print(response)
        except Exception as e:
            print(f"Error: {e}")

        # Reset for next prompt
        agent.reset()

        # Wait a bit between prompts
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())