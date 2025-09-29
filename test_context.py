#!/usr/bin/env python3
"""
Test script to demonstrate context retention in agent conversations
"""
import asyncio
from freshai.agent.ollama_agent import OllamaAgent
from freshai.agent.base_agent import Tool
from freshai.agent.tools import create_evidence_analyzer_tool

async def test_context():
    """Test that agent maintains context between questions."""

    print("🧪 Testing Agent Context Retention")
    print("=" * 50)

    # Create evidence analyzer tool
    evidence_tool_spec = create_evidence_analyzer_tool(
        evidence_store_path="./evidence",
        vision_model="gemma3:12b"
    )

    evidence_tool = Tool(
        name=evidence_tool_spec["name"],
        description=evidence_tool_spec["description"],
        function=evidence_tool_spec["function"],
        parameters=evidence_tool_spec["parameters"]
    )

    # Create agent with improved context instructions
    system_prompt = """You are FreshAI, an investigation assistant.

CRITICAL: Maintain context between questions and provide DIRECT answers.

When asked follow-up questions:
1. Reference your previous analysis findings
2. Give clear YES/NO answers when asked specific questions
3. Don't re-analyze unless asked to

You have access to evidence_analyzer tool for comprehensive case analysis."""

    agent = OllamaAgent(
        model="gemma3:latest",
        system_prompt=system_prompt,
        tools=[evidence_tool],
        verbose=True,
        temperature=0.3,  # Lower temperature for more consistent responses
    )

    # Test sequence
    print("\n📋 Test 1: Initial Evidence Analysis")
    print("-" * 40)
    response1 = await agent.run("Analyze all evidence in CASE_001")
    print("✅ Response 1:")
    print(response1[:200] + "..." if len(response1) > 200 else response1)

    print(f"\n📋 Test 2: Direct Follow-up Question (Context Test)")
    print("-" * 40)
    print("🔍 Question: 'Did the suspect travel to Rome?'")
    print("🎯 Expected: Direct YES answer referencing photo_3.jpg with Colosseum")

    response2 = await agent.run("Did the suspect travel to Rome?")
    print("✅ Response 2:")
    print(response2)

    # Analyze the responses
    print("\n📊 Context Analysis:")
    print("-" * 40)

    # Check conversation history
    history = agent.get_conversation_history()
    print(f"Conversation history length: {len(history)} messages")

    # Check if agent references previous evidence
    rome_mentioned = "rome" in response2.lower()
    colosseum_mentioned = "colosseum" in response2.lower()
    photo3_mentioned = "photo_3" in response2.lower()

    print(f"✓ Rome mentioned: {rome_mentioned}")
    print(f"✓ Colosseum mentioned: {colosseum_mentioned}")
    print(f"✓ Photo_3 referenced: {photo3_mentioned}")

    # Check if it's a direct answer
    yes_answer = "yes" in response2.lower()[:50]  # Check first 50 chars for direct answer

    print(f"✓ Direct YES answer: {yes_answer}")

    if yes_answer and (colosseum_mentioned or photo3_mentioned):
        print("\n🎉 SUCCESS: Agent maintained context and gave direct answer!")
    else:
        print("\n❌ ISSUE: Agent did not properly use context or give direct answer")
        print(f"Response was: {response2[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_context())