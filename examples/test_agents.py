"""Example script to test the agent implementations."""
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from freshai.agent.base_agent import Tool
from freshai.agent.ollama_agent import OllamaAgent
from freshai.agent.transformers_agent import TransformersAgent
from freshai.agent.tools import create_bash_tool, create_web_search_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


async def test_ollama_agent():
    """Test the Ollama agent with tools."""
    print("\n" + "=" * 80)
    print("Testing Ollama Agent")
    print("=" * 80)

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

    # System prompt
    system_prompt = """You are a helpful AI assistant with access to tools.
You can execute bash commands and search the web.
Always be concise and clear in your responses.
When asked about system information, use the bash tool.
When asked about current events or general knowledge, use the web search tool."""

    # Create agent
    agent = OllamaAgent(
        model="gemma3:latest",  # Using the available gemma3 model
        system_prompt=system_prompt,
        tools=[bash_tool, web_tool],
        verbose=True,
        temperature=0.7
    )

    # Check if model is available
    if not await agent.check_model_availability():
        print(f"\nModel '{agent.model}' not available. Please pull it first:")
        print(f"  ollama pull {agent.model}")
        return

    # Test prompts
    test_prompts = [
        "What files are in the current directory? List them with details.",
        "What is the capital of France and what's the current weather there?",
        "Create a simple Python hello world script and then run it."
    ]

    for prompt in test_prompts:
        print(f"\n{'=' * 60}")
        print(f"PROMPT: {prompt}")
        print('=' * 60)

        try:
            response = await agent.run(prompt)
            print(f"\n{'=' * 60}")
            print("FINAL RESPONSE:")
            print('=' * 60)
            print(response)
        except Exception as e:
            print(f"Error: {e}")

        # Reset for next prompt
        agent.reset()


def test_transformers_agent():
    """Test the Transformers agent with tools (synchronous wrapper)."""
    print("\n" + "=" * 80)
    print("Testing Transformers Agent")
    print("=" * 80)

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

    # System prompt
    system_prompt = """You are a helpful AI assistant with access to tools.
You can execute bash commands and search the web.
Be concise and use tools when needed."""

    try:
        # Create agent
        agent = TransformersAgent(
            model_name="microsoft/Phi-3-mini-4k-instruct",  # Small model for testing
            system_prompt=system_prompt,
            tools=[bash_tool, web_tool],
            verbose=True,
            temperature=0.7,
            max_new_tokens=256,
            load_in_8bit=False  # Set to True if you have bitsandbytes installed
        )

        # Test a simple prompt
        prompt = "What is the current directory and how many Python files are in it?"

        print(f"\n{'=' * 60}")
        print(f"PROMPT: {prompt}")
        print('=' * 60)

        # Run synchronously
        async def run():
            return await agent.run(prompt)

        response = asyncio.run(run())

        print(f"\n{'=' * 60}")
        print("FINAL RESPONSE:")
        print('=' * 60)
        print(response)

        # Clear cache
        agent.clear_cache()

    except ImportError as e:
        print(f"Transformers not available: {e}")
        print("Install with: pip install transformers torch")
    except Exception as e:
        print(f"Error: {e}")


async def test_simple_agent():
    """Test agent without ML dependencies."""
    print("\n" + "=" * 80)
    print("Testing Simple Agent (Mock)")
    print("=" * 80)

    from freshai.agent.base_agent import BaseAgent, Message, Tool

    class MockAgent(BaseAgent):
        """Simple mock agent for testing."""

        async def _call_llm(self, messages, tools=None):
            """Mock LLM that demonstrates tool calling."""
            # Get last user message
            last_user_msg = None
            for msg in reversed(messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break

            if not last_user_msg:
                return Message(role="assistant", content="No user message found")

            # Simple logic to demonstrate tool calling
            if "directory" in last_user_msg.lower() or "files" in last_user_msg.lower():
                # First call bash tool
                if not any(msg.role == "tool" for msg in messages):
                    return Message(
                        role="assistant",
                        content="Let me check the directory contents.",
                        tool_calls=[{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "ls -la"}'
                            }
                        }]
                    )

            if "search" in last_user_msg.lower() or "what is" in last_user_msg.lower():
                # Call web search tool
                if not any(msg.role == "tool" and msg.name == "web_search" for msg in messages):
                    query = "Python programming" # Extract from message in real implementation
                    return Message(
                        role="assistant",
                        content="Let me search for that information.",
                        tool_calls=[{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": f'{{"query": "{query}"}}'
                            }
                        }]
                    )

            # Final response after tool calls
            tool_results = []
            for msg in messages:
                if msg.role == "tool":
                    tool_results.append(f"Tool {msg.name}: {msg.content[:100]}")

            if tool_results:
                return Message(
                    role="assistant",
                    content=f"Based on the tools I used:\n" + "\n".join(tool_results)
                )

            return Message(
                role="assistant",
                content=f"I understood your request: {last_user_msg}"
            )

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

    # Create mock agent
    agent = MockAgent(
        system_prompt="You are a helpful assistant.",
        tools=[bash_tool, web_tool],
        verbose=True
    )

    # Test
    response = await agent.run("Show me the files in the current directory")
    print(f"\nResponse: {response}")


async def main():
    """Run all tests."""
    print("Agent Testing Suite")
    print("=" * 80)

    # Test simple agent (always works)
    await test_simple_agent()

    # Test Ollama agent (requires Ollama running)
    try:
        await test_ollama_agent()
    except Exception as e:
        print(f"\nOllama test skipped: {e}")
        print("Make sure Ollama is running: ollama serve")

    # Test Transformers agent (requires transformers installed)
    try:
        test_transformers_agent()
    except Exception as e:
        print(f"\nTransformers test skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())