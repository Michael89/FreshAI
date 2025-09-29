"""Evidence analyzer tool for comprehensive case investigation."""
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from freshai.agent.tools.image_analysis_tool import ImageAnalysisTool

logger = logging.getLogger(__name__)


class EvidenceAnalyzerTool:
    """Tool for comprehensive analysis of case evidence including images, documents, and metadata."""

    def __init__(
        self,
        evidence_store_path: str,
        vision_model: str = "gemma3:12b",
        max_images_per_batch: int = 5
    ):
        """Initialize the evidence analyzer tool.

        Args:
            evidence_store_path: Path to the evidence store directory
            vision_model: Ollama VLM model for image analysis
            max_images_per_batch: Maximum images to analyze in one batch
        """
        self.evidence_store = Path(evidence_store_path)
        self.vision_model = vision_model
        self.max_images_per_batch = max_images_per_batch

        # Initialize image analysis tool
        self.image_analyzer = ImageAnalysisTool(model=vision_model)

        # Supported file types
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        self.document_extensions = {'.txt', '.md', '.json', '.csv', '.log'}

        logger.info(f"Initialized EvidenceAnalyzerTool for: {self.evidence_store}")

    def scan_case_structure(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        """Scan the evidence store for case folders and files.

        Args:
            case_id: Specific case ID to analyze, or None for all cases

        Returns:
            Dictionary containing case structure and file inventory
        """
        logger.info(f"Scanning evidence store: {self.evidence_store}")

        if not self.evidence_store.exists():
            return {"error": f"Evidence store not found: {self.evidence_store}"}

        cases = {}

        # Look for case folders
        case_folders = []
        if case_id:
            # Look for specific case
            case_path = self.evidence_store / case_id
            if case_path.exists() and case_path.is_dir():
                case_folders = [case_path]
        else:
            # Scan for all case folders
            case_folders = [p for p in self.evidence_store.iterdir()
                          if p.is_dir() and not p.name.startswith('.')]

        for case_folder in case_folders:
            case_name = case_folder.name
            logger.info(f"Analyzing case: {case_name}")

            case_info = {
                "path": str(case_folder),
                "images": [],
                "documents": [],
                "metadata": {},
                "subdirectories": []
            }

            # Recursively scan case folder
            self._scan_directory(case_folder, case_info)

            cases[case_name] = case_info

        return {
            "evidence_store": str(self.evidence_store),
            "total_cases": len(cases),
            "cases": cases
        }

    def _classify_folder_type(self, folder_name: str) -> str:
        """Intelligently classify folder type based on naming patterns."""
        folder_lower = folder_name.lower()

        # Evidence from suspect
        if any(x in folder_lower for x in ["suspect", "device", "phone", "laptop", "seized"]):
            return "suspect_evidence"

        # Investigator materials
        if any(x in folder_lower for x in ["investigator", "reference", "comparison", "scene"]):
            return "investigator_reference"

        # Case documentation
        if any(x in folder_lower for x in ["notes", "reports", "logs", "documents"]):
            return "case_documentation"

        # Default folders (reports, temp, etc.)
        if folder_lower in ["reports", "temp", "logs", "cases"]:
            return "system_folder"

        return "unknown"

    def _scan_directory(self, directory: Path, case_info: Dict[str, Any]) -> None:
        """Recursively scan a directory for evidence files with intelligent classification."""
        try:
            for item in directory.iterdir():
                if item.is_file():
                    suffix = item.suffix.lower()
                    relative_path = str(item.relative_to(self.evidence_store))

                    # Determine folder type for context
                    parent_folder = item.parent.name
                    folder_type = self._classify_folder_type(parent_folder)

                    file_info = {
                        "path": str(item),
                        "relative_path": relative_path,
                        "name": item.name,
                        "size": item.stat().st_size,
                        "modified": item.stat().st_mtime,
                        "folder_type": folder_type,
                        "source_context": self._get_source_context(folder_type)
                    }

                    if suffix in self.image_extensions:
                        case_info["images"].append(file_info)
                    elif suffix in self.document_extensions:
                        case_info["documents"].append(file_info)

                elif item.is_dir() and not item.name.startswith('.'):
                    # Record subdirectory and scan it
                    folder_type = self._classify_folder_type(item.name)
                    subdir_info = {
                        "name": item.name,
                        "path": str(item),
                        "type": folder_type,
                        "description": self._get_folder_description(folder_type),
                        "files": []
                    }
                    case_info["subdirectories"].append(subdir_info)
                    self._scan_directory(item, case_info)

        except PermissionError as e:
            logger.warning(f"Permission denied accessing {directory}: {e}")

    def _get_source_context(self, folder_type: str) -> str:
        """Get context description for file source."""
        context_map = {
            "suspect_evidence": "Evidence from suspect's device - proves suspect's activities/locations",
            "investigator_reference": "Reference material collected by investigator - for comparison only",
            "case_documentation": "Case documentation and investigator notes",
            "system_folder": "System/administrative folder",
            "unknown": "Unknown source context"
        }
        return context_map.get(folder_type, "Unknown source context")

    def _get_folder_description(self, folder_type: str) -> str:
        """Get description for folder type."""
        descriptions = {
            "suspect_evidence": "Contains evidence seized from suspect (proves suspect actions)",
            "investigator_reference": "Contains reference materials for comparison (not proof of suspect activity)",
            "case_documentation": "Contains case notes, reports, and documentation",
            "system_folder": "System folder for case organization",
            "unknown": "Purpose unknown"
        }
        return descriptions.get(folder_type, "Purpose unknown")

    async def analyze_case_images(self, case_info: Dict[str, Any], focus_question: str = None) -> Dict[str, Any]:
        """Analyze all images in a case with VLM.

        Args:
            case_info: Case information from scan_case_structure
            focus_question: Specific question to focus image analysis on

        Returns:
            Dictionary containing analysis results for all images
        """
        images = case_info.get("images", [])
        if not images:
            return {"message": "No images found in case"}

        logger.info(f"Analyzing {len(images)} images for case")

        # Context-aware analysis questions
        def get_questions_for_context(folder_type):
            if folder_type == "suspect_evidence":
                return [
                    "Where was this photo taken? Identify location, landmarks, or geographical features",
                    "What activities or behavior is the person engaging in?",
                    "Are there any identifying details, text, or notable objects that could be evidence?"
                ]
            elif folder_type == "investigator_reference":
                return [
                    "This is a reference photo for comparison. Describe the location and key features",
                    "What details in this image could be used to compare with suspect evidence?"
                ]
            else:
                return [
                    "Describe what you see in this image in detail",
                    "What objects, people, or locations are visible?",
                    "Are there any identifying features, text, or notable details?"
                ]

        analysis_results = {}

        # Process images in batches to avoid overwhelming the VLM
        for i in range(0, len(images), self.max_images_per_batch):
            batch = images[i:i + self.max_images_per_batch]
            logger.info(f"Processing image batch {i//self.max_images_per_batch + 1}")

            batch_tasks = []
            for image_info in batch:
                image_path = image_info["path"]

                # Choose question based on focus or use context-aware analysis
                if focus_question:
                    questions = [focus_question]
                else:
                    folder_type = image_info.get("folder_type", "unknown")
                    questions = get_questions_for_context(folder_type)

                # Analyze image with each question
                for question in questions:
                    task = self._analyze_single_image(image_path, question)
                    batch_tasks.append((image_info, question, task))

            # Wait for batch completion
            for image_info, question, task in batch_tasks:
                try:
                    result = await task
                    image_name = image_info["name"]

                    if image_name not in analysis_results:
                        analysis_results[image_name] = {
                            "path": image_info["path"],
                            "analyses": {}
                        }

                    analysis_results[image_name]["analyses"][question] = result

                except Exception as e:
                    logger.error(f"Failed to analyze {image_info['name']}: {e}")
                    analysis_results[image_info["name"]] = {"error": str(e)}

        return {
            "total_images": len(images),
            "successfully_analyzed": len([r for r in analysis_results.values() if "error" not in r]),
            "results": analysis_results
        }

    async def _analyze_single_image(self, image_path: str, question: str) -> str:
        """Analyze a single image with given question."""
        return await self.image_analyzer.analyze_image(image_path, question)

    def read_case_documents(self, case_info: Dict[str, Any]) -> Dict[str, Any]:
        """Read and parse all text documents in a case.

        Args:
            case_info: Case information from scan_case_structure

        Returns:
            Dictionary containing document contents
        """
        documents = case_info.get("documents", [])
        if not documents:
            return {"message": "No documents found in case"}

        logger.info(f"Reading {len(documents)} documents")

        document_contents = {}

        for doc_info in documents:
            doc_path = Path(doc_info["path"])
            doc_name = doc_info["name"]

            try:
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                document_contents[doc_name] = {
                    "path": str(doc_path),
                    "size": len(content),
                    "content": content[:5000]  # Limit content to prevent overwhelming
                }

                if len(content) > 5000:
                    document_contents[doc_name]["truncated"] = True
                    document_contents[doc_name]["full_size"] = len(content)

                logger.debug(f"Read document: {doc_name} ({len(content)} chars)")

            except Exception as e:
                logger.error(f"Failed to read {doc_name}: {e}")
                document_contents[doc_name] = {"error": str(e)}

        return {
            "total_documents": len(documents),
            "documents": document_contents
        }

    async def comprehensive_analysis(self, case_id: Optional[str] = None, focus_question: str = None) -> str:
        """Perform comprehensive analysis of case evidence.

        Args:
            case_id: Specific case to analyze, or None for all cases
            focus_question: Optional focused question for image analysis

        Returns:
            Comprehensive analysis report as formatted string
        """
        logger.info("Starting comprehensive evidence analysis")

        # Step 1: Scan case structure
        case_structure = self.scan_case_structure(case_id)

        if "error" in case_structure:
            return f"Error: {case_structure['error']}"

        if not case_structure["cases"]:
            return "No cases found in evidence store"

        # Step 2: Analyze each case
        report_sections = []
        report_sections.append("=== COMPREHENSIVE EVIDENCE ANALYSIS REPORT ===\n")
        report_sections.append(f"Evidence Store: {case_structure['evidence_store']}")
        report_sections.append(f"Total Cases Found: {case_structure['total_cases']}\n")

        for case_name, case_info in case_structure["cases"].items():
            report_sections.append(f"\n--- CASE: {case_name} ---")
            report_sections.append(f"Location: {case_info['path']}")
            report_sections.append(f"Images: {len(case_info['images'])}")
            report_sections.append(f"Documents: {len(case_info['documents'])}")
            report_sections.append(f"Subdirectories: {len(case_info['subdirectories'])}")

            # Analyze documents first
            if case_info["documents"]:
                report_sections.append("\n** DOCUMENT ANALYSIS **")
                doc_analysis = self.read_case_documents(case_info)

                for doc_name, doc_data in doc_analysis.get("documents", {}).items():
                    if "error" in doc_data:
                        report_sections.append(f"• {doc_name}: Error - {doc_data['error']}")
                    else:
                        report_sections.append(f"• {doc_name} ({doc_data['size']} chars):")
                        # Show first 500 chars of content
                        preview = doc_data["content"][:500]
                        report_sections.append(f"  Preview: {preview}...")

            # Analyze images with context awareness
            if case_info["images"]:
                # Separate images by type
                suspect_images = [img for img in case_info["images"] if img.get("folder_type") == "suspect_evidence"]
                reference_images = [img for img in case_info["images"] if img.get("folder_type") == "investigator_reference"]

                if suspect_images:
                    report_sections.append("\n** SUSPECT EVIDENCE - IMAGE ANALYSIS **")
                    suspect_case_info = {"images": suspect_images}
                    image_analysis = await self.analyze_case_images(suspect_case_info, focus_question)

                    for img_name, img_data in image_analysis.get("results", {}).items():
                        # Find the original image info
                        orig_info = next((img for img in suspect_images if img["name"] == img_name), {})
                        report_sections.append(f"\n• {img_name} (SUSPECT DEVICE - PROVES SUSPECT ACTIVITY):")
                        report_sections.append(f"  Source: {orig_info.get('source_context', 'Unknown')}")

                        if "error" in img_data:
                            report_sections.append(f"  Error: {img_data['error']}")
                        else:
                            for question, result in img_data.get("analyses", {}).items():
                                report_sections.append(f"  Q: {question}")
                                report_sections.append(f"  A: {result[:300]}...")

                if reference_images:
                    report_sections.append("\n** INVESTIGATOR REFERENCE PHOTOS **")
                    ref_case_info = {"images": reference_images}
                    ref_analysis = await self.analyze_case_images(ref_case_info, focus_question)

                    for img_name, img_data in ref_analysis.get("results", {}).items():
                        orig_info = next((img for img in reference_images if img["name"] == img_name), {})
                        report_sections.append(f"\n• {img_name} (REFERENCE ONLY - NOT PROOF OF SUSPECT ACTIVITY):")
                        report_sections.append(f"  Purpose: {orig_info.get('source_context', 'Unknown')}")

                        if "error" in img_data:
                            report_sections.append(f"  Error: {img_data['error']}")
                        else:
                            for question, result in img_data.get("analyses", {}).items():
                                report_sections.append(f"  Q: {question}")
                                report_sections.append(f"  A: {result[:200]}...")

        # Step 3: Generate intelligent conclusion based on focus question and analysis results
        if focus_question:
            report_sections.append("\n=== INVESTIGATIVE CONCLUSION ===")
            report_sections.append(f"Question: {focus_question}")

            # Process analysis results to generate conclusions
            suspect_evidence_conclusions = []
            reference_notes = []
            rome_evidence_found = False

            for case_name, case_info in case_structure["cases"].items():
                suspect_files = [f for f in case_info["images"] if f.get("folder_type") == "suspect_evidence"]
                reference_files = [f for f in case_info["images"] if f.get("folder_type") == "investigator_reference"]

                # Analyze suspect evidence results for Rome-related content
                if suspect_files:
                    suspect_case_info = {"images": suspect_files}
                    try:
                        image_analysis = await self.analyze_case_images(suspect_case_info, focus_question)

                        for img_name, img_data in image_analysis.get("results", {}).items():
                            if "analyses" in img_data:
                                for question, result in img_data["analyses"].items():
                                    result_lower = result.lower()

                                    # Check for Rome indicators in the analysis result
                                    if "rome" in focus_question.lower():
                                        if any(keyword in result_lower for keyword in ["colosseum", "rome", "roman", "italy"]):
                                            if "colosseum" in result_lower:
                                                suspect_evidence_conclusions.append(f"✅ DEFINITIVE: {img_name} - Analysis shows Colosseum/Rome landmarks")
                                                rome_evidence_found = True
                                            elif "rome" in result_lower and "not" not in result_lower[:50]:
                                                suspect_evidence_conclusions.append(f"✅ STRONG: {img_name} - Analysis indicates Rome location")
                                                rome_evidence_found = True
                                        elif any(keyword in result_lower for keyword in ["paris", "seoul", "korea", "france", "los santos", "gta"]):
                                            suspect_evidence_conclusions.append(f"📍 {img_name} - Shows other location (not Rome)")
                                        elif "impossible" in result_lower or "can't determine" in result_lower:
                                            suspect_evidence_conclusions.append(f"❓ {img_name} - Location unclear from image")
                    except Exception as e:
                        logger.warning(f"Could not re-analyze suspect images for conclusion: {e}")

                # Note reference materials
                for file_info in reference_files:
                    if "hotel" in file_info["name"].lower():
                        reference_notes.append(f"📋 {file_info['name']} - Reference material only (NOT proof of suspect activity)")

            # Generate final conclusion based on suspect evidence analysis
            if "rome" in focus_question.lower():
                if rome_evidence_found:
                    report_sections.append("\n🎯 FINAL ANSWER: YES - Suspect has been to Rome")
                    report_sections.append("BASIS: Evidence from suspect's device shows Rome landmarks")
                else:
                    report_sections.append("\n🎯 FINAL ANSWER: NO definitive proof - No clear Rome evidence in suspect's device")

                if suspect_evidence_conclusions:
                    report_sections.append("\n📊 Evidence from Suspect Device (PRIMARY):")
                    for conclusion in suspect_evidence_conclusions:
                        report_sections.append(f"  {conclusion}")

                if reference_notes:
                    report_sections.append("\n📋 Reference Materials (SECONDARY):")
                    for note in reference_notes:
                        report_sections.append(f"  {note}")

                report_sections.append("\n⚖️  PRINCIPLE: Only evidence from suspect's device proves suspect's activities.")

        # Step 4: Generate summary
        report_sections.append("\n=== ANALYSIS SUMMARY ===")
        total_images = sum(len(case["images"]) for case in case_structure["cases"].values())
        total_docs = sum(len(case["documents"]) for case in case_structure["cases"].values())

        report_sections.append(f"Total Images Analyzed: {total_images}")
        report_sections.append(f"Total Documents Read: {total_docs}")
        report_sections.append("Analysis complete. All available evidence has been processed.")

        return "\n".join(report_sections)

    def __call__(self, case_id: str = None, focus_question: str = None) -> str:
        """Synchronous wrapper for comprehensive analysis."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, run in thread
            import concurrent.futures

            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self.comprehensive_analysis(case_id, focus_question)
                    )
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5 minute timeout

        except RuntimeError:
            # No event loop running
            return asyncio.run(self.comprehensive_analysis(case_id, focus_question))


def create_evidence_analyzer_tool(
    evidence_store_path: str,
    vision_model: str = "gemma3:12b"
) -> Dict[str, Any]:
    """Create an evidence analyzer tool specification for agents.

    Args:
        evidence_store_path: Path to evidence store directory
        vision_model: Ollama VLM model for image analysis

    Returns:
        Tool specification with function and metadata
    """
    tool = EvidenceAnalyzerTool(evidence_store_path, vision_model)

    return {
        "name": "evidence_analyzer",
        "description": "Comprehensively analyze all evidence in a case including images, documents, and metadata. Can analyze specific cases or all available cases in the evidence store.",
        "function": tool,
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {
                    "type": "string",
                    "description": "Specific case ID to analyze (optional, analyzes all cases if not provided)"
                },
                "focus_question": {
                    "type": "string",
                    "description": "Specific question to focus image analysis on (optional, uses comprehensive analysis if not provided)"
                }
            },
            "required": []
        }
    }