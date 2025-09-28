"""Evidence analysis tools for crime investigation."""

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .registry import BaseTool


class EvidenceAnalyzer(BaseTool):
    """Tool for analyzing digital evidence files."""
    
    def __init__(self):
        super().__init__(
            name="evidence_analyzer",
            description="Analyzes digital evidence files and extracts metadata"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evidence file and return metadata."""
        file_path = parameters.get("file_path")
        if not file_path:
            raise ValueError("file_path parameter is required")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Evidence file not found: {file_path}")
        
        return await self._analyze_evidence(path)
    
    async def _analyze_evidence(self, file_path: Path) -> Dict[str, Any]:
        """Perform comprehensive evidence analysis."""
        result = {
            "file_info": self._get_file_info(file_path),
            "hash_analysis": self._calculate_hashes(file_path),
            "content_analysis": await self._analyze_content(file_path),
            "forensic_metadata": self._extract_forensic_metadata(file_path),
            "analysis_timestamp": datetime.now().isoformat(),
        }
        
        return result
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic file information."""
        stat = file_path.stat()
        mime_type, encoding = mimetypes.guess_type(str(file_path))
        
        return {
            "name": file_path.name,
            "path": str(file_path.absolute()),
            "size_bytes": stat.st_size,
            "size_human": self._format_bytes(stat.st_size),
            "mime_type": mime_type,
            "encoding": encoding,
            "extension": file_path.suffix.lower(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
        }
    
    def _calculate_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate file hashes for integrity verification."""
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
                hashes["md5"] = hashlib.md5(content).hexdigest()
                hashes["sha1"] = hashlib.sha1(content).hexdigest()
                hashes["sha256"] = hashlib.sha256(content).hexdigest()
                
        except Exception as e:
            hashes["error"] = str(e)
        
        return hashes
    
    async def _analyze_content(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file content based on type."""
        content_info = {"type": "unknown"}
        
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            if mime_type:
                if mime_type.startswith("text"):
                    content_info = await self._analyze_text_file(file_path)
                elif mime_type.startswith("image"):
                    content_info = await self._analyze_image_file(file_path)
                elif mime_type == "application/json":
                    content_info = await self._analyze_json_file(file_path)
                else:
                    content_info = {"type": mime_type, "analysis": "binary_file"}
            
        except Exception as e:
            content_info["error"] = str(e)
        
        return content_info
    
    async def _analyze_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze text-based files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            words = content.split()
            
            return {
                "type": "text",
                "line_count": len(lines),
                "word_count": len(words),
                "character_count": len(content),
                "encoding": "utf-8",
                "preview": content[:500] + "..." if len(content) > 500 else content,
                "contains_suspicious_keywords": self._check_suspicious_keywords(content),
            }
            
        except Exception as e:
            return {"type": "text", "error": str(e)}
    
    async def _analyze_image_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze image files."""
        try:
            from PIL import Image
            
            with Image.open(file_path) as img:
                return {
                    "type": "image",
                    "format": img.format,
                    "mode": img.mode,
                    "dimensions": img.size,
                    "has_exif": bool(getattr(img, '_getexif', None)),
                }
                
        except ImportError:
            return {"type": "image", "analysis": "pillow_not_available"}
        except Exception as e:
            return {"type": "image", "error": str(e)}
    
    async def _analyze_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "type": "json",
                "valid_json": True,
                "structure": type(data).__name__,
                "key_count": len(data) if isinstance(data, dict) else None,
                "item_count": len(data) if isinstance(data, list) else None,
                "top_level_keys": list(data.keys())[:10] if isinstance(data, dict) else None,
            }
            
        except json.JSONDecodeError as e:
            return {"type": "json", "valid_json": False, "error": str(e)}
        except Exception as e:
            return {"type": "json", "error": str(e)}
    
    def _extract_forensic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract forensic-relevant metadata."""
        metadata = {
            "chain_of_custody": {
                "analyzed_by": "FreshAI Evidence Analyzer",
                "analysis_date": datetime.now().isoformat(),
                "original_location": str(file_path.absolute()),
            },
            "integrity": {
                "file_exists": file_path.exists(),
                "readable": file_path.is_file(),
                "permissions": oct(file_path.stat().st_mode)[-3:] if file_path.exists() else None,
            }
        }
        
        return metadata
    
    def _check_suspicious_keywords(self, content: str) -> List[str]:
        """Check for suspicious keywords in text content."""
        suspicious_keywords = [
            "password", "key", "secret", "token", "api", "credential",
            "hack", "exploit", "vulnerability", "backdoor", "malware",
            "drug", "weapon", "illegal", "fraud", "money laundering"
        ]
        
        content_lower = content.lower()
        found_keywords = [
            keyword for keyword in suspicious_keywords 
            if keyword in content_lower
        ]
        
        return found_keywords
    
    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes into human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.2f} PB"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the evidence file to analyze"
                }
            },
            "required": ["file_path"]
        }