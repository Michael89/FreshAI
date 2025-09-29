#!/usr/bin/env python3
"""
FreshAI Transformers Demo Script

This script demonstrates the Transformers agent with bash tool
"""
import sys
import asyncio
from pathlib import Path
from freshai.agent.transformers_agent import TransformersAgent
from freshai.agent.base_agent import Tool
from freshai.agent.tools import create_bash_tool

def main():
    print("Initializing Transformers Agent with Bash Tool...")

    # Step 1: Create the bash tool specification (same as before)
    bash_tool_spec = create_bash_tool(
        safe_mode=True,      # Prevents dangerous commands
        timeout=30,          # Commands timeout after 30 seconds
        max_output_length=10000  # Limit output length
    )

    # Step 2: Convert the tool spec into a Tool object
    bash_tool = Tool(
        name=bash_tool_spec["name"],
        description=bash_tool_spec["description"],
        function=bash_tool_spec["function"],
        parameters=bash_tool_spec["parameters"]
    )

    # Step 3: Create the Transformers agent with the bash tool
    agent = TransformersAgent(
        model_name="google/gemma-2b-it",  # Using Gemma 2B for faster loading (7B is too big to download quickly)
        system_prompt="You are a helpful AI assistant with access to bash commands. Use the bash tool when needed to answer questions about the system or files.",
        tools=[bash_tool],     # Pass the bash tool here
        verbose=True,
        temperature=0.7,
        max_new_tokens=512,    # Response length
        load_in_8bit=False,    # Disable 8-bit for faster loading with 2B model
        device=None            # Auto-detect GPU/CPU
    )

    print("Agent initialized! Running query...")

    # Step 4: Run the agent with a prompt that will use the bash tool
    result = asyncio.run(agent.run("What is the current directory and how many Python files are in it? Look only on top level"))

    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(result)

    # Step 5: Clean up model from memory (optional but good practice)
    agent.clear_cache()
    print("\nModel cache cleared.")

if __name__ == "__main__":
    main()