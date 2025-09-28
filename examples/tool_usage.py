"""Example of using FreshAI tools directly."""

import asyncio
from pathlib import Path
from freshai.core import FreshAICore
from freshai.config import Config


async def main():
    """Example of using investigation tools directly."""
    
    # Initialize core system
    config = Config.load_from_env()
    core = FreshAICore(config)
    
    try:
        await core.initialize()
        
        print("FreshAI Core initialized successfully")
        print(f"Available models: {core.get_available_models()}")
        print(f"Available tools: {core.get_available_tools()}")
        
        # Example 1: Text Analysis Tool
        print("\n" + "="*50)
        print("TEXT ANALYSIS EXAMPLE")
        print("="*50)
        
        sample_text = """
        Meet me at the warehouse on 5th Street at 3 PM tomorrow.
        Bring the package. Password is "eagle123".
        Call me at (555) 123-4567 if there are any problems.
        Don't trust anyone else with this information.
        """
        
        text_result = await core.use_tool("text_analyzer", {
            "text": sample_text,
            "analysis_type": "comprehensive"
        })
        
        if text_result["status"] == "success":
            result = text_result["result"]
            
            print(f"Text length: {result['text_length']} characters")
            print(f"Word count: {result['word_count']}")
            
            if "pattern_analysis" in result:
                patterns = result["pattern_analysis"]["extracted_patterns"]
                print(f"Phone numbers found: {patterns.get('phone_numbers', [])}")
                print(f"Addresses found: {patterns.get('addresses', [])}")
                print(f"Times found: {patterns.get('times', [])}")
            
            if "keyword_analysis" in result:
                keywords = result["keyword_analysis"]
                print(f"Risk level: {keywords['risk_level']}")
                print(f"Suspicious categories: {keywords['categories_found']}")
        else:
            print(f"Text analysis failed: {text_result['error']}")
        
        # Example 2: Evidence Analysis Tool
        print("\n" + "="*50)
        print("EVIDENCE ANALYSIS EXAMPLE")
        print("="*50)
        
        # Create a sample evidence file
        evidence_dir = Path("./evidence")
        evidence_dir.mkdir(exist_ok=True)
        
        sample_file = evidence_dir / "sample_evidence.txt"
        sample_file.write_text(sample_text)
        
        evidence_result = await core.use_tool("evidence_analyzer", {
            "file_path": str(sample_file)
        })
        
        if evidence_result["status"] == "success":
            result = evidence_result["result"]
            
            file_info = result["file_info"]
            print(f"File name: {file_info['name']}")
            print(f"File size: {file_info['size_human']}")
            print(f"MIME type: {file_info['mime_type']}")
            
            hash_info = result["hash_analysis"]
            print(f"MD5: {hash_info.get('md5', 'N/A')}")
            print(f"SHA256: {hash_info.get('sha256', 'N/A')}")
            
            if "content_analysis" in result:
                content = result["content_analysis"]
                if "contains_suspicious_keywords" in content:
                    suspicious = content["contains_suspicious_keywords"]
                    if suspicious:
                        print(f"Suspicious keywords: {suspicious}")
        else:
            print(f"Evidence analysis failed: {evidence_result['error']}")
        
        # Example 3: Using LLM for investigation queries
        print("\n" + "="*50)
        print("LLM INVESTIGATION QUERY")
        print("="*50)
        
        investigation_prompt = """
        You are assisting a criminal investigation. Based on the following evidence:
        
        Text message: "Meet me at the warehouse on 5th Street at 3 PM tomorrow. Bring the package."
        
        Please analyze this text and provide:
        1. Potential criminal activity indicators
        2. Key evidence elements (locations, times, etc.)
        3. Recommended follow-up actions
        4. Questions investigators should ask
        
        Provide a professional analysis suitable for law enforcement.
        """
        
        try:
            llm_response = await core.generate_response(investigation_prompt)
            print(f"Model: {llm_response.model_name}")
            print(f"Processing time: {llm_response.processing_time:.2f}s")
            print(f"Analysis:\n{llm_response.content}")
        except Exception as e:
            print(f"LLM analysis failed: {e}")
        
        # Example 4: Image Analysis (if image exists)
        print("\n" + "="*50)
        print("IMAGE ANALYSIS EXAMPLE")
        print("="*50)
        
        # Look for any image files in evidence directory
        image_files = list(evidence_dir.glob("*.jpg")) + list(evidence_dir.glob("*.png"))
        
        if image_files:
            image_file = image_files[0]
            print(f"Analyzing image: {image_file}")
            
            # Try VLM analysis
            try:
                vlm_response = await core.analyze_image(
                    str(image_file),
                    "Analyze this image for potential evidence in a criminal investigation."
                )
                print(f"VLM Analysis: {vlm_response.content[:300]}...")
            except Exception as e:
                print(f"VLM analysis not available: {e}")
            
            # Try image tool analysis
            try:
                image_result = await core.use_tool("image_analyzer", {
                    "image_path": str(image_file),
                    "analysis_type": "quality"
                })
                
                if image_result["status"] == "success":
                    result = image_result["result"]
                    if "quality_analysis" in result:
                        quality = result["quality_analysis"]
                        print(f"Image quality: {quality.get('quality_assessment', 'Unknown')}")
                        print(f"Blur score: {quality.get('blur_score', 'N/A')}")
                        print(f"Is blurry: {quality.get('is_blurry', 'Unknown')}")
            except Exception as e:
                print(f"Image tool analysis failed: {e}")
        else:
            print("No image files found in evidence directory")
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
    except Exception as e:
        print(f"Error during tool usage: {e}")
    
    finally:
        await core.cleanup()
        print("Core cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())