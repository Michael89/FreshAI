"""Main investigator agent for crime investigations."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..core import FreshAICore
from ..config import Config
from ..core.base import ModelResponse


logger = logging.getLogger(__name__)


class InvestigatorAgent:
    """AI Agent specialized for crime investigation assistance."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load_from_env()
        self.core = FreshAICore(self.config)
        self.investigation_context = {}
        self.case_history = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the investigator agent."""
        if self._initialized:
            return
            
        await self.core.initialize()
        self._initialized = True
        logger.info("Investigator Agent initialized successfully")
    
    async def start_case(self, case_id: str, case_description: str = "") -> Dict[str, Any]:
        """Start a new investigation case."""
        if not self._initialized:
            await self.initialize()
        
        case_info = {
            "case_id": case_id,
            "description": case_description,
            "start_time": datetime.now().isoformat(),
            "evidence_count": 0,
            "analysis_results": [],
            "status": "active"
        }
        
        self.investigation_context[case_id] = case_info
        self.case_history.append(case_info)
        
        logger.info(f"Started new case: {case_id}")
        return case_info
    
    async def analyze_evidence(
        self, 
        case_id: str, 
        evidence_path: str, 
        evidence_type: str = "auto"
    ) -> Dict[str, Any]:
        """Analyze evidence file for the investigation."""
        if not self._initialized:
            await self.initialize()
        
        if case_id not in self.investigation_context:
            raise ValueError(f"Case {case_id} not found. Please start the case first.")
        
        evidence_file = Path(evidence_path)
        if not evidence_file.exists():
            raise FileNotFoundError(f"Evidence file not found: {evidence_path}")
        
        # Auto-detect evidence type if not specified
        if evidence_type == "auto":
            evidence_type = self._detect_evidence_type(evidence_file)
        
        analysis_result = {
            "evidence_path": str(evidence_file.absolute()),
            "evidence_type": evidence_type,
            "analysis_timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        try:
            if evidence_type == "image":
                analysis_result["results"] = await self._analyze_image_evidence(evidence_path)
            elif evidence_type == "text":
                analysis_result["results"] = await self._analyze_text_evidence(evidence_path)
            elif evidence_type == "document":
                analysis_result["results"] = await self._analyze_document_evidence(evidence_path)
            else:
                # Use general evidence analyzer
                analysis_result["results"] = await self.core.use_tool(
                    "evidence_analyzer", 
                    {"file_path": evidence_path}
                )
            
            # Add to case context
            self.investigation_context[case_id]["evidence_count"] += 1
            self.investigation_context[case_id]["analysis_results"].append(analysis_result)
            
            logger.info(f"Analyzed evidence for case {case_id}: {evidence_path}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze evidence: {e}")
            analysis_result["results"] = {"error": str(e)}
            return analysis_result
    
    async def _analyze_image_evidence(self, image_path: str) -> Dict[str, Any]:
        """Analyze image evidence using VLM and image tools."""
        results = {}
        
        # Use VLM for image analysis
        try:
            vlm_response = await self.core.analyze_image(
                image_path, 
                "Analyze this image for criminal investigation purposes. "
                "Describe any evidence, objects, people, or suspicious elements you observe."
            )
            results["vlm_analysis"] = {
                "description": vlm_response.content,
                "model": vlm_response.model_name,
                "confidence": vlm_response.confidence,
                "processing_time": vlm_response.processing_time
            }
        except Exception as e:
            results["vlm_analysis"] = {"error": str(e)}
        
        # Use image analysis tool
        try:
            tool_result = await self.core.use_tool(
                "image_analyzer", 
                {"image_path": image_path, "analysis_type": "comprehensive"}
            )
            results["technical_analysis"] = tool_result["result"]
        except Exception as e:
            results["technical_analysis"] = {"error": str(e)}
        
        return results
    
    async def _analyze_text_evidence(self, text_path: str) -> Dict[str, Any]:
        """Analyze text evidence using LLM and text tools."""
        results = {}
        
        # Read the text file
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        except Exception as e:
            return {"error": f"Failed to read text file: {e}"}
        
        # Use LLM for content analysis
        try:
            llm_prompt = f"""
            Analyze the following text for criminal investigation purposes:
            
            {text_content[:2000]}...
            
            Please identify:
            1. Key individuals mentioned
            2. Locations and addresses
            3. Dates and times
            4. Potential evidence or suspicious content
            5. Communication patterns
            6. Any threats or concerning language
            
            Provide a detailed analysis for investigators.
            """
            
            llm_response = await self.core.generate_response(llm_prompt)
            results["llm_analysis"] = {
                "analysis": llm_response.content,
                "model": llm_response.model_name,
                "processing_time": llm_response.processing_time
            }
        except Exception as e:
            results["llm_analysis"] = {"error": str(e)}
        
        # Use text analysis tool
        try:
            tool_result = await self.core.use_tool(
                "text_analyzer", 
                {"file_path": text_path, "analysis_type": "comprehensive"}
            )
            results["pattern_analysis"] = tool_result["result"]
        except Exception as e:
            results["pattern_analysis"] = {"error": str(e)}
        
        return results
    
    async def _analyze_document_evidence(self, doc_path: str) -> Dict[str, Any]:
        """Analyze document evidence."""
        # Use general evidence analyzer for documents
        try:
            tool_result = await self.core.use_tool(
                "evidence_analyzer", 
                {"file_path": doc_path}
            )
            return tool_result["result"]
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_evidence_type(self, file_path: Path) -> str:
        """Auto-detect evidence type based on file extension and content."""
        extension = file_path.suffix.lower()
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        text_extensions = {'.txt', '.log', '.md', '.csv'}
        document_extensions = {'.pdf', '.doc', '.docx', '.rtf'}
        
        if extension in image_extensions:
            return "image"
        elif extension in text_extensions:
            return "text"
        elif extension in document_extensions:
            return "document"
        else:
            return "unknown"
    
    async def ask_question(self, case_id: str, question: str) -> Dict[str, Any]:
        """Ask a question about the case using available evidence and context."""
        if not self._initialized:
            await self.initialize()
        
        if case_id not in self.investigation_context:
            raise ValueError(f"Case {case_id} not found")
        
        case_info = self.investigation_context[case_id]
        
        # Build context from previous analyses
        context_summary = self._build_case_context(case_info)
        
        prompt = f"""
        You are an AI assistant helping with criminal investigation case {case_id}.
        
        Case Description: {case_info.get('description', 'N/A')}
        Evidence Analyzed: {case_info['evidence_count']} items
        
        Context from previous analyses:
        {context_summary}
        
        Investigator Question: {question}
        
        Please provide a detailed and helpful response based on the available evidence and context.
        If you need additional information or evidence to answer properly, please specify what would be helpful.
        """
        
        try:
            response = await self.core.generate_response(prompt)
            
            result = {
                "question": question,
                "answer": response.content,
                "model": response.model_name,
                "timestamp": datetime.now().isoformat(),
                "case_id": case_id,
                "processing_time": response.processing_time
            }
            
            # Add to case history
            if "questions" not in case_info:
                case_info["questions"] = []
            case_info["questions"].append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "case_id": case_id
            }
    
    def _build_case_context(self, case_info: Dict[str, Any]) -> str:
        """Build a summary of case context for LLM prompts."""
        if not case_info.get("analysis_results"):
            return "No evidence has been analyzed yet."
        
        context_parts = []
        
        for i, analysis in enumerate(case_info["analysis_results"]):
            evidence_type = analysis.get("evidence_type", "unknown")
            results = analysis.get("results", {})
            
            context_parts.append(f"Evidence {i+1} ({evidence_type}):")
            
            if "vlm_analysis" in results:
                context_parts.append(f"  Visual Analysis: {results['vlm_analysis'].get('description', 'N/A')[:200]}...")
            
            if "llm_analysis" in results:
                context_parts.append(f"  Content Analysis: {results['llm_analysis'].get('analysis', 'N/A')[:200]}...")
            
            if "pattern_analysis" in results:
                pattern_info = results["pattern_analysis"]
                if "suspicious_keywords" in pattern_info:
                    keywords = pattern_info["suspicious_keywords"]
                    if keywords:
                        context_parts.append(f"  Suspicious Keywords Found: {list(keywords.keys())}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def generate_case_report(self, case_id: str) -> Dict[str, Any]:
        """Generate a comprehensive case report."""
        if case_id not in self.investigation_context:
            raise ValueError(f"Case {case_id} not found")
        
        case_info = self.investigation_context[case_id]
        
        # Use LLM to generate a comprehensive report
        context_summary = self._build_case_context(case_info)
        
        prompt = f"""
        Generate a comprehensive criminal investigation report for case {case_id}.
        
        Case Information:
        - Case ID: {case_id}
        - Description: {case_info.get('description', 'N/A')}
        - Start Time: {case_info['start_time']}
        - Evidence Items Analyzed: {case_info['evidence_count']}
        
        Analysis Summary:
        {context_summary}
        
        Please generate a professional investigation report that includes:
        1. Executive Summary
        2. Evidence Overview
        3. Key Findings
        4. Potential Leads
        5. Recommendations for Further Investigation
        6. Technical Analysis Summary
        
        Format the report professionally for law enforcement use.
        """
        
        try:
            response = await self.core.generate_response(prompt)
            
            report = {
                "case_id": case_id,
                "report_generated": datetime.now().isoformat(),
                "report_content": response.content,
                "case_statistics": {
                    "evidence_count": case_info["evidence_count"],
                    "questions_asked": len(case_info.get("questions", [])),
                    "case_duration": self._calculate_case_duration(case_info["start_time"]),
                },
                "model_used": response.model_name
            }
            
            # Save report to case storage
            await self._save_case_report(case_id, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate case report: {e}")
            return {"error": str(e), "case_id": case_id}
    
    def _calculate_case_duration(self, start_time: str) -> str:
        """Calculate case duration."""
        start_dt = datetime.fromisoformat(start_time)
        now = datetime.now()
        duration = now - start_dt
        
        hours = duration.total_seconds() / 3600
        if hours < 1:
            return f"{int(duration.total_seconds() / 60)} minutes"
        elif hours < 24:
            return f"{int(hours)} hours"
        else:
            return f"{int(hours / 24)} days"
    
    async def _save_case_report(self, case_id: str, report: Dict[str, Any]) -> None:
        """Save case report to storage."""
        try:
            report_path = self.config.case_storage_path / f"{case_id}_report.json"
            
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Case report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save case report: {e}")
    
    def get_case_status(self, case_id: str) -> Dict[str, Any]:
        """Get current status of a case."""
        if case_id not in self.investigation_context:
            raise ValueError(f"Case {case_id} not found")
        
        return self.investigation_context[case_id].copy()
    
    def list_active_cases(self) -> List[str]:
        """List all active cases."""
        return [
            case_id for case_id, case_info in self.investigation_context.items()
            if case_info.get("status") == "active"
        ]
    
    async def close_case(self, case_id: str) -> Dict[str, Any]:
        """Close an investigation case."""
        if case_id not in self.investigation_context:
            raise ValueError(f"Case {case_id} not found")
        
        case_info = self.investigation_context[case_id]
        case_info["status"] = "closed"
        case_info["close_time"] = datetime.now().isoformat()
        
        # Generate final report
        final_report = await self.generate_case_report(case_id)
        
        logger.info(f"Case {case_id} closed")
        return {
            "case_id": case_id,
            "status": "closed",
            "final_report": final_report
        }
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        await self.core.cleanup()
        self._initialized = False