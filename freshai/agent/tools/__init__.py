"""Agent tools package."""
from freshai.agent.tools.bash_tool import BashTool, create_bash_tool
from freshai.agent.tools.web_search_tool import (
    WebSearchTool,
    SimpleWebSearchTool,
    create_web_search_tool
)
from freshai.agent.tools.image_analysis_tool import (
    ImageAnalysisTool,
    create_image_analysis_tool
)
from freshai.agent.tools.evidence_analyzer_tool import (
    EvidenceAnalyzerTool,
    create_evidence_analyzer_tool
)

__all__ = [
    "BashTool",
    "create_bash_tool",
    "WebSearchTool",
    "SimpleWebSearchTool",
    "create_web_search_tool",
    "ImageAnalysisTool",
    "create_image_analysis_tool",
    "EvidenceAnalyzerTool",
    "create_evidence_analyzer_tool"
]