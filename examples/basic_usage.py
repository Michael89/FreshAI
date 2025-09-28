"""Basic usage example for FreshAI."""

import asyncio
from pathlib import Path
from freshai import InvestigatorAgent, Config


async def main():
    """Basic example of using FreshAI for crime investigation."""
    
    # Initialize configuration
    config = Config.load_from_env()
    
    # Create investigator agent
    agent = InvestigatorAgent(config)
    
    try:
        # Initialize the agent
        print("Initializing FreshAI Agent...")
        await agent.initialize()
        
        # Start a new case
        case_id = "EXAMPLE_001"
        case_description = "Digital evidence analysis example"
        
        print(f"\nStarting case: {case_id}")
        case_info = await agent.start_case(case_id, case_description)
        print(f"Case started at: {case_info['start_time']}")
        
        # Example: Analyze text evidence (if file exists)
        text_evidence = Path("./evidence/sample_text.txt")
        if text_evidence.exists():
            print(f"\nAnalyzing text evidence: {text_evidence}")
            text_analysis = await agent.analyze_evidence(
                case_id, 
                str(text_evidence), 
                "text"
            )
            print("Text analysis completed")
            
            # Print key findings
            if "pattern_analysis" in text_analysis.get("results", {}):
                patterns = text_analysis["results"]["pattern_analysis"]
                if "suspicious_keywords" in patterns:
                    keywords = patterns["suspicious_keywords"]
                    if keywords:
                        print(f"Suspicious keywords found: {list(keywords.keys())}")
        
        # Example: Analyze image evidence (if file exists)
        image_evidence = Path("./evidence/sample_image.jpg")
        if image_evidence.exists():
            print(f"\nAnalyzing image evidence: {image_evidence}")
            image_analysis = await agent.analyze_evidence(
                case_id, 
                str(image_evidence), 
                "image"
            )
            print("Image analysis completed")
            
            # Print VLM analysis if available
            if "vlm_analysis" in image_analysis.get("results", {}):
                vlm_result = image_analysis["results"]["vlm_analysis"]
                if "description" in vlm_result:
                    print(f"VLM Description: {vlm_result['description'][:200]}...")
        
        # Ask questions about the case
        questions = [
            "What evidence has been analyzed so far?",
            "Are there any suspicious patterns in the analyzed content?",
            "What are the key findings from this investigation?"
        ]
        
        for question in questions:
            print(f"\nAsking: {question}")
            response = await agent.ask_question(case_id, question)
            
            if "error" not in response:
                print(f"Answer: {response['answer'][:300]}...")
                print(f"Model used: {response['model']}")
            else:
                print(f"Error: {response['error']}")
        
        # Generate case report
        print("\nGenerating case report...")
        report = await agent.generate_case_report(case_id)
        
        if "error" not in report:
            print("Case report generated successfully")
            print(f"Report length: {len(report['report_content'])} characters")
            print(f"Evidence analyzed: {report['case_statistics']['evidence_count']}")
            print(f"Questions asked: {report['case_statistics']['questions_asked']}")
        else:
            print(f"Error generating report: {report['error']}")
        
        # Check case status
        print(f"\nCase status:")
        status = agent.get_case_status(case_id)
        print(f"- Evidence count: {status['evidence_count']}")
        print(f"- Questions asked: {len(status.get('questions', []))}")
        print(f"- Status: {status['status']}")
        
        # Close the case
        print(f"\nClosing case: {case_id}")
        close_result = await agent.close_case(case_id)
        print(f"Case closed with status: {close_result['status']}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
    
    finally:
        # Clean up resources
        await agent.cleanup()
        print("\nAgent cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())