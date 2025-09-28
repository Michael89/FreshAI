"""Text analysis tools for crime investigation."""

import re
from pathlib import Path
from typing import Dict, Any, List, Set
from datetime import datetime
from collections import Counter

from .registry import BaseTool


class TextAnalyzer(BaseTool):
    """Tool for analyzing text content in criminal investigations."""
    
    def __init__(self):
        super().__init__(
            name="text_analyzer",
            description="Analyzes text content for investigative purposes"
        )
        
        # Predefined patterns for investigation
        self.patterns = {
            "phone_numbers": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "email_addresses": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "urls": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "ip_addresses": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "credit_cards": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "times": r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?\b',
            "addresses": r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl)\b',
        }
        
        # Suspicious keywords for criminal investigations
        self.suspicious_keywords = {
            "weapons": ["gun", "firearm", "rifle", "pistol", "weapon", "ammunition", "ammo", "bullet", "knife", "blade"],
            "drugs": ["cocaine", "heroin", "marijuana", "meth", "drug", "dealer", "trafficking", "substance", "narcotic"],
            "violence": ["kill", "murder", "assault", "attack", "violence", "threat", "harm", "hurt", "beat"],
            "financial": ["money laundering", "fraud", "embezzlement", "theft", "stolen", "robbery", "burglary"],
            "cyber": ["hack", "malware", "virus", "breach", "exploit", "phishing", "ransomware", "botnet"],
            "organized_crime": ["gang", "cartel", "mafia", "organized", "syndicate", "crew", "family"],
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text for investigation purposes."""
        text_input = parameters.get("text")
        file_path = parameters.get("file_path")
        analysis_type = parameters.get("analysis_type", "comprehensive")
        
        if text_input:
            text = text_input
            source = "direct_input"
        elif file_path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Text file not found: {file_path}")
            
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                source = str(path.absolute())
            except Exception as e:
                raise Exception(f"Failed to read file: {e}")
        else:
            raise ValueError("Either 'text' or 'file_path' parameter is required")
        
        return await self._analyze_text(text, source, analysis_type)
    
    async def _analyze_text(self, text: str, source: str, analysis_type: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis."""
        result = {
            "source": source,
            "analysis_timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split('\n')),
        }
        
        if analysis_type in ["comprehensive", "patterns"]:
            result["pattern_analysis"] = self._extract_patterns(text)
        
        if analysis_type in ["comprehensive", "keywords"]:
            result["keyword_analysis"] = self._analyze_keywords(text)
        
        if analysis_type in ["comprehensive", "entities"]:
            result["entity_analysis"] = self._extract_entities(text)
        
        if analysis_type in ["comprehensive", "sentiment"]:
            result["sentiment_analysis"] = self._analyze_sentiment(text)
        
        if analysis_type in ["comprehensive", "linguistic"]:
            result["linguistic_analysis"] = self._linguistic_analysis(text)
        
        return result
    
    def _extract_patterns(self, text: str) -> Dict[str, Any]:
        """Extract patterns relevant to investigations."""
        extracted = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if pattern_name == "phone_numbers":
                    # Format phone numbers properly
                    formatted_numbers = []
                    for match in matches:
                        if isinstance(match, tuple):
                            formatted_numbers.append(f"({match[0]}) {match[1]}-{match[2]}")
                        else:
                            formatted_numbers.append(match)
                    extracted[pattern_name] = list(set(formatted_numbers))
                else:
                    extracted[pattern_name] = list(set(matches))
        
        return {
            "extracted_patterns": extracted,
            "total_patterns_found": sum(len(v) for v in extracted.values()),
            "pattern_types": list(extracted.keys()),
        }
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze suspicious keywords."""
        text_lower = text.lower()
        found_keywords = {}
        
        for category, keywords in self.suspicious_keywords.items():
            category_matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)
                    category_matches.append({"keyword": keyword, "count": count})
            
            if category_matches:
                found_keywords[category] = category_matches
        
        # Calculate risk score
        total_matches = sum(
            sum(item["count"] for item in matches)
            for matches in found_keywords.values()
        )
        
        risk_level = "low"
        if total_matches > 10:
            risk_level = "high"
        elif total_matches > 5:
            risk_level = "medium"
        
        return {
            "suspicious_keywords": found_keywords,
            "total_suspicious_matches": total_matches,
            "risk_level": risk_level,
            "categories_found": list(found_keywords.keys()),
        }
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities (simplified approach)."""
        # This is a simplified entity extraction
        # In a real implementation, you'd use NLP libraries like spaCy or NLTK
        
        entities = {
            "potential_names": self._extract_potential_names(text),
            "locations": self._extract_potential_locations(text),
            "organizations": self._extract_potential_organizations(text),
        }
        
        return entities
    
    def _extract_potential_names(self, text: str) -> List[str]:
        """Extract potential person names (basic approach)."""
        # Look for capitalized words that might be names
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Filter out common words that aren't names
        common_words = {
            "The", "This", "That", "And", "But", "For", "With", "From", "To",
            "In", "On", "At", "By", "Of", "As", "Is", "Are", "Was", "Were",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        }
        
        potential_names = [word for word in words if word not in common_words]
        
        # Count occurrences and return most frequent ones
        name_counts = Counter(potential_names)
        return [name for name, count in name_counts.most_common(10)]
    
    def _extract_potential_locations(self, text: str) -> List[str]:
        """Extract potential location names."""
        # Look for common location indicators
        location_patterns = [
            r'\b[A-Z][a-z]+\s+(?:City|County|State|Street|Avenue|Road|Drive|Boulevard)\b',
            r'\b(?:Downtown|Uptown|North|South|East|West)\s+[A-Z][a-z]+\b',
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.extend(matches)
        
        return list(set(locations))
    
    def _extract_potential_organizations(self, text: str) -> List[str]:
        """Extract potential organization names."""
        # Look for common organization indicators
        org_patterns = [
            r'\b[A-Z][a-z]+\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation|Organization|Agency)\b',
            r'\b(?:The\s+)?[A-Z][a-z]+\s+(?:Department|Bureau|Office|Division)\b',
        ]
        
        organizations = []
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            organizations.extend(matches)
        
        return list(set(organizations))
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis."""
        # Simplified sentiment analysis using word lists
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "like", "enjoy", "happy", "pleased", "satisfied"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "hate", "dislike",
            "angry", "mad", "frustrated", "disappointed", "sad", "upset"
        ]
        
        threatening_words = [
            "threat", "threaten", "kill", "hurt", "harm", "destroy",
            "revenge", "payback", "retaliate", "punish"
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        threatening_count = sum(1 for word in words if word in threatening_words)
        
        total_words = len(words)
        
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        threat_score = threatening_count / max(total_words, 1)
        
        return {
            "sentiment_score": round(sentiment_score, 3),
            "threat_score": round(threat_score, 3),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "threatening_indicators": threatening_count,
            "overall_sentiment": self._classify_sentiment(sentiment_score, threat_score),
        }
    
    def _classify_sentiment(self, sentiment_score: float, threat_score: float) -> str:
        """Classify overall sentiment."""
        if threat_score > 0.01:
            return "threatening"
        elif sentiment_score > 0.01:
            return "positive"
        elif sentiment_score < -0.01:
            return "negative"
        else:
            return "neutral"
    
    def _linguistic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform linguistic analysis."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate readability metrics (simplified)
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        avg_chars_per_word = sum(len(word) for word in words) / max(len(words), 1)
        
        # Analyze punctuation usage
        punctuation_counts = {
            "periods": text.count('.'),
            "exclamations": text.count('!'),
            "questions": text.count('?'),
            "commas": text.count(','),
        }
        
        return {
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_chars_per_word": round(avg_chars_per_word, 2),
            "sentence_count": len(sentences),
            "punctuation_usage": punctuation_counts,
            "writing_style": self._assess_writing_style(avg_words_per_sentence, punctuation_counts),
        }
    
    def _assess_writing_style(self, avg_words: float, punctuation: Dict[str, int]) -> str:
        """Assess writing style characteristics."""
        if avg_words > 20:
            complexity = "complex"
        elif avg_words > 10:
            complexity = "moderate"
        else:
            complexity = "simple"
        
        if punctuation["exclamations"] > punctuation["periods"]:
            emotion = "emotional"
        elif punctuation["questions"] > punctuation["periods"] * 0.3:
            emotion = "inquisitive"
        else:
            emotion = "neutral"
        
        return f"{complexity}, {emotion}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to analyze"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to text file to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["comprehensive", "patterns", "keywords", "entities", "sentiment", "linguistic"],
                    "description": "Type of analysis to perform",
                    "default": "comprehensive"
                }
            },
            "oneOf": [
                {"required": ["text"]},
                {"required": ["file_path"]}
            ]
        }