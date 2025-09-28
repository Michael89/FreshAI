"""Validation utilities for FreshAI."""

from pathlib import Path
from typing import Any, Dict, List
import mimetypes


def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """Validate and return a Path object."""
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    return path


def validate_image_file(file_path: str) -> Path:
    """Validate that file is an image."""
    path = validate_file_path(file_path)
    
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"File is not an image: {file_path}")
    
    return path


def validate_text_file(file_path: str) -> Path:
    """Validate that file is a text file."""
    path = validate_file_path(file_path)
    
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type and not mime_type.startswith('text/'):
        # Try to read as text anyway
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read(100)  # Try to read first 100 chars
        except UnicodeDecodeError:
            raise ValueError(f"File is not readable as text: {file_path}")
    
    return path


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check required sections
    required_sections = ['ollama', 'transformers']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required config section: {section}")
    
    # Validate Ollama config
    if 'ollama' in config:
        ollama_config = config['ollama']
        if 'host' not in ollama_config:
            issues.append("Ollama config missing 'host'")
        if 'port' not in ollama_config:
            issues.append("Ollama config missing 'port'")
        elif not isinstance(ollama_config['port'], int):
            issues.append("Ollama port must be an integer")
    
    # Validate paths
    path_configs = ['evidence_storage_path', 'case_storage_path']
    for path_config in path_configs:
        if path_config in config:
            try:
                path = Path(config[path_config])
                if not path.exists():
                    issues.append(f"Path does not exist: {path_config} = {path}")
            except Exception as e:
                issues.append(f"Invalid path for {path_config}: {e}")
    
    return issues


def validate_case_id(case_id: str) -> str:
    """Validate and normalize case ID."""
    if not case_id or not isinstance(case_id, str):
        raise ValueError("Case ID must be a non-empty string")
    
    # Remove invalid characters
    normalized = "".join(c for c in case_id if c.isalnum() or c in "-_")
    
    if not normalized:
        raise ValueError("Case ID contains no valid characters")
    
    if len(normalized) > 100:
        raise ValueError("Case ID too long (max 100 characters)")
    
    return normalized