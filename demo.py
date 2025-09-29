#!/usr/bin/env python3
"""
FreshAI Demo Script

This script demonstrates the core functionality of FreshAI
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
        description="FreshAI Demo Script - AI agent with tool calling capabilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Evidence store path
    parser.add_argument(
        "--evidence-store", "-e",
        type=str,
        default="./evidence",
        help="Path to evidence store directory where all investigation data will be kept"
    )

    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemma3:latest",
        help="Ollama model to use for the agent"
    )

    # Prompt
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="What is the current directory and how many Python files are in it? Look only on top level",
        help="Prompt to send to the agent"
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

    print(f"📁 Evidence store: {evidence_dir.absolute()}")

    # Create subdirectories for organization
    subdirs = ["cases", "reports", "temp", "logs"]
    for subdir in subdirs:
        (evidence_dir / subdir).mkdir(exist_ok=True)

    return evidence_dir

def main():
    # Parse command line arguments
    args = parse_arguments()

    print("🚀 FreshAI Agent Demo")
    print("=" * 60)

    # Setup evidence store
    evidence_dir = setup_evidence_store(args.evidence_store)

    print(f"🤖 Model: {args.model}")
    print(f"🔧 Temperature: {args.temperature}")
    print(f"📝 Prompt: {args.prompt}")
    print("=" * 60)

    # Step 1: Create the bash tool specification
    bash_tool_spec = create_bash_tool(
        safe_mode=True,      # This prevents dangerous commands like rm -rf
        timeout=30,          # Commands timeout after 30 seconds
        max_output_length=10000,  # Limit output to prevent overwhelming responses
        working_dir=str(evidence_dir)  # Set working directory to evidence store
    )

    # Step 2: Create the image analysis tool specification
    image_tool_spec = create_image_analysis_tool(
        model="gemma3:12b",  # Using gemma3:12b as VLM
        timeout=60           # Longer timeout for image analysis
    )

    # Step 3: Create the evidence analyzer tool specification
    evidence_tool_spec = create_evidence_analyzer_tool(
        evidence_store_path=str(evidence_dir),
        vision_model="gemma3:12b"
    )

    # Step 4: Convert the tool specs into Tool objects that the agent can use
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

    # Step 5: Create the agent with all tools
    system_prompt = f"""You are FreshAI, an AI investigation assistant with comprehensive evidence analysis capabilities.

Evidence Store: {evidence_dir.absolute()}

Available Tools:
• bash: Execute system commands, examine files, create reports
• image_analysis: Analyze individual images with specific questions
• evidence_analyzer: COMPREHENSIVE ANALYSIS of entire cases - scans all evidence, analyzes all images, reads documents

IMPORTANT: When asked to investigate a case or analyze evidence:
1. Use 'evidence_analyzer' first to get a complete overview of all available evidence
2. This will automatically analyze ALL images in the case and read ALL documents
3. Use specific tools like 'image_analysis' or 'bash' for detailed follow-up questions

The evidence_analyzer is your primary investigation tool - it will discover and analyze everything in the case automatically."""

    agent = OllamaAgent(
        model=args.model,
        system_prompt=system_prompt,
        tools=[bash_tool, image_tool, evidence_tool],  # All three tools available
        verbose=args.verbose,
        temperature=args.temperature
    )

    # Step 4: Run the agent with the provided prompt
    print("\n🔍 Starting investigation...")
    result = asyncio.run(agent.run(args.prompt))

    print("\n" + "="*60)
    print("📋 FINAL ANSWER:")
    print("="*60)
    print(result)

    print(f"\n💾 Evidence stored in: {evidence_dir.absolute()}")
    
if __name__ == "__main__":
    main()
