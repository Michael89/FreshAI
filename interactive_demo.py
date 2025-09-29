#!/usr/bin/env python3
"""
Interactive FreshAI Demo Script

This script allows continuous conversation with the agent until you type /exit
"""
import sys
import asyncio
import argparse
from pathlib import Path
from freshai.agent.ollama_agent import OllamaAgent
from freshai.agent.base_agent import Tool
from freshai.agent.tools import create_bash_tool, create_image_analysis_tool, create_evidence_analyzer_tool

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive FreshAI Agent - Continuous conversation mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Evidence store path
    parser.add_argument(
        "--evidence-store", "-e",
        type=str,
        default="./evidence",
        help="Path to evidence store directory (parent directory containing cases)"
    )

    # Case ID
    parser.add_argument(
        "--case-id", "-c",
        type=str,
        default="CASE_001",
        help="Default case ID to focus on"
    )

    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemma3:latest",
        help="Ollama model to use for the agent"
    )

    # Verbose flag
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    # Temperature
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Temperature for model inference (0.0 to 1.0)"
    )

    return parser.parse_args()

def setup_evidence_store(evidence_path: str) -> Path:
    """Setup the evidence store directory."""
    evidence_dir = Path(evidence_path)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organization
    subdirs = ["cases", "reports", "temp", "logs"]
    for subdir in subdirs:
        (evidence_dir / subdir).mkdir(exist_ok=True)

    return evidence_dir

async def interactive_session():
    """Run interactive session with the agent."""
    args = parse_arguments()

    print("🚀 Interactive FreshAI Agent")
    print("=" * 60)

    # Setup evidence store
    evidence_dir = setup_evidence_store(args.evidence_store)

    print(f"📁 Evidence store: {evidence_dir.absolute()}")
    print(f"📋 Default case: {args.case_id}")
    print(f"🤖 Model: {args.model}")
    print(f"🔧 Temperature: {args.temperature}")
    print("=" * 60)

    # Create tools
    bash_tool_spec = create_bash_tool(
        safe_mode=True,
        timeout=30,
        max_output_length=10000,
        working_dir=str(evidence_dir)
    )

    image_tool_spec = create_image_analysis_tool(
        model="gemma3:12b",
        timeout=60
    )

    evidence_tool_spec = create_evidence_analyzer_tool(
        evidence_store_path=str(evidence_dir),
        vision_model="gemma3:12b"
    )

    # Convert to Tool objects
    bash_tool = Tool(
        name=bash_tool_spec["name"],
        description=bash_tool_spec["description"],
        function=bash_tool_spec["function"],
        parameters=bash_tool_spec["parameters"]
    )

    image_tool = Tool(
        name=image_tool_spec["name"],
        description=image_tool_spec["description"],
        function=image_tool_spec["function"],
        parameters=image_tool_spec["parameters"]
    )

    evidence_tool = Tool(
        name=evidence_tool_spec["name"],
        description=evidence_tool_spec["description"],
        function=evidence_tool_spec["function"],
        parameters=evidence_tool_spec["parameters"]
    )

    # Enhanced system prompt for better context retention
    system_prompt = f"""You are FreshAI, an AI investigation assistant with comprehensive evidence analysis capabilities.

Evidence Store: {evidence_dir.absolute()}
Current Case: {args.case_id}

IMPORTANT INSTRUCTIONS:
1. MAINTAIN CONTEXT: Remember all previous analysis results and conversations
2. GIVE DIRECT ANSWERS: When asked specific questions, provide clear yes/no answers based on your evidence analysis
3. REFERENCE PREVIOUS FINDINGS: If you've already analyzed evidence, refer to those findings rather than re-analyzing

Available Tools:
• evidence_analyzer: Comprehensive analysis of entire cases (use with case_id="{args.case_id}" by default)
• image_analysis: Analyze individual images with specific questions
• bash: Execute system commands, examine files, create reports

When asked follow-up questions:
- Use your memory of previous analysis results
- Give direct, specific answers
- Reference specific evidence (e.g., "Based on photo_3.jpg showing the Colosseum...")

CONVERSATION CONTEXT: This is a continuous conversation. Remember everything from previous exchanges.
When asked to analyze evidence without specifying a case, use case "{args.case_id}" by default."""

    # Create agent
    agent = OllamaAgent(
        model=args.model,
        system_prompt=system_prompt,
        tools=[bash_tool, image_tool, evidence_tool],
        verbose=args.verbose,
        temperature=args.temperature
    )

    print("\n💬 Interactive Mode Active")
    print("Type your questions. Use '/exit' to quit, '/reset' to clear conversation history")
    print("=" * 60)

    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Question: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() == "/exit":
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() == "/reset":
                agent.reset()
                print("\n🔄 Conversation history cleared")
                continue
            elif user_input.lower() == "/help":
                print("""
Commands:
  /exit   - Exit the program
  /reset  - Clear conversation history
  /help   - Show this help message

Ask any investigative questions about your evidence!
Examples:
  - "Analyze all evidence in the current case"
  - "Analyze all evidence in CASE_001" (specific case)
  - "Did the suspect travel to Rome?"
  - "What locations can you identify in the images?"

Note: Current default case is {args.case_id}
                """)
                continue

            # Process user question
            print(f"\n⏳ Processing...")
            if not args.verbose:
                print("(Use -v flag to see detailed logs)")

            result = await agent.run(user_input)

            print("\n" + "=" * 60)
            print("📋 ANSWER:")
            print("=" * 60)
            print(result)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Type /help for available commands")

async def main():
    """Main function."""
    await interactive_session()

if __name__ == "__main__":
    asyncio.run(main())