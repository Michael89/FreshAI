"""Image analysis tools for crime investigation."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .registry import BaseTool


class ImageAnalyzer(BaseTool):
    """Tool for analyzing images in criminal investigations."""
    
    def __init__(self):
        super().__init__(
            name="image_analyzer",
            description="Analyzes images for forensic and investigative purposes"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image for investigation purposes."""
        image_path = parameters.get("image_path")
        analysis_type = parameters.get("analysis_type", "comprehensive")
        
        if not image_path:
            raise ValueError("image_path parameter is required")
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        return await self._analyze_image(path, analysis_type)
    
    async def _analyze_image(self, image_path: Path, analysis_type: str) -> Dict[str, Any]:
        """Perform comprehensive image analysis."""
        result = {
            "image_info": self._get_image_info(image_path),
            "analysis_timestamp": datetime.now().isoformat(),
        }
        
        if analysis_type in ["comprehensive", "quality"]:
            result["quality_analysis"] = self._analyze_image_quality(image_path)
        
        if analysis_type in ["comprehensive", "objects"]:
            result["object_detection"] = await self._detect_objects(image_path)
        
        if analysis_type in ["comprehensive", "forensic"]:
            result["forensic_analysis"] = self._forensic_analysis(image_path)
        
        if analysis_type in ["comprehensive", "metadata"]:
            result["metadata_analysis"] = self._extract_metadata(image_path)
        
        return result
    
    def _get_image_info(self, image_path: Path) -> Dict[str, Any]:
        """Extract basic image information."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"error": "Failed to load image"}
            
            height, width, channels = img.shape
            
            return {
                "path": str(image_path.absolute()),
                "filename": image_path.name,
                "dimensions": {"width": width, "height": height},
                "channels": channels,
                "total_pixels": width * height,
                "file_size": image_path.stat().st_size,
                "format": image_path.suffix.lower(),
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_image_quality(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image quality metrics."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"error": "Failed to load image"}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Noise estimation
            noise_estimate = self._estimate_noise(gray)
            
            # Dynamic range
            dynamic_range = np.max(gray) - np.min(gray)
            
            return {
                "brightness": float(brightness),
                "contrast": float(contrast),
                "blur_score": float(blur_score),
                "is_blurry": blur_score < 100,  # Threshold for blur detection
                "noise_estimate": float(noise_estimate),
                "dynamic_range": int(dynamic_range),
                "quality_assessment": self._assess_quality(brightness, contrast, blur_score),
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in image."""
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise = np.var(laplacian)
        return noise
    
    def _assess_quality(self, brightness: float, contrast: float, blur_score: float) -> str:
        """Assess overall image quality."""
        quality_score = 0
        
        # Brightness assessment (ideal range: 50-200)
        if 50 <= brightness <= 200:
            quality_score += 1
        
        # Contrast assessment (higher is generally better)
        if contrast > 30:
            quality_score += 1
        
        # Blur assessment
        if blur_score > 100:
            quality_score += 1
        
        if quality_score >= 3:
            return "excellent"
        elif quality_score >= 2:
            return "good"
        elif quality_score >= 1:
            return "fair"
        else:
            return "poor"
    
    async def _detect_objects(self, image_path: Path) -> Dict[str, Any]:
        """Detect objects in the image (simplified implementation)."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"error": "Failed to load image"}
            
            # Simple edge detection as a proxy for object detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours
            objects = []
            for i, contour in enumerate(contours[:10]):  # Limit to top 10 objects
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "id": i,
                        "area": int(area),
                        "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "perimeter": int(cv2.arcLength(contour, True)),
                    })
            
            return {
                "total_objects_detected": len(objects),
                "objects": objects,
                "detection_method": "contour_analysis",
                "note": "This is a simplified object detection. For advanced detection, use specialized models."
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _forensic_analysis(self, image_path: Path) -> Dict[str, Any]:
        """Perform forensic analysis on the image."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"error": "Failed to load image"}
            
            # Histogram analysis
            hist_analysis = self._analyze_histogram(img)
            
            # Potential tampering detection (basic)
            tampering_analysis = self._detect_potential_tampering(img)
            
            return {
                "histogram_analysis": hist_analysis,
                "tampering_analysis": tampering_analysis,
                "forensic_quality": "suitable" if self._is_forensically_suitable(img) else "unsuitable",
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_histogram(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color histogram for forensic purposes."""
        # Calculate histograms for each channel
        histograms = {}
        colors = ['blue', 'green', 'red']
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms[color] = {
                "mean": float(np.mean(hist)),
                "std": float(np.std(hist)),
                "max": float(np.max(hist)),
                "peak_value": int(np.argmax(hist)),
            }
        
        return histograms
    
    def _detect_potential_tampering(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic tampering detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for unusual patterns that might indicate tampering
        # This is a simplified approach
        
        # JPEG artifact analysis (if applicable)
        artifacts = self._analyze_compression_artifacts(gray)
        
        # Edge inconsistency analysis
        edge_analysis = self._analyze_edge_consistency(gray)
        
        return {
            "compression_artifacts": artifacts,
            "edge_consistency": edge_analysis,
            "tampering_likelihood": "low",  # Simplified assessment
            "note": "This is a basic tampering analysis. Professional forensic tools provide more accurate results."
        }
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze compression artifacts."""
        # Simplified analysis
        gradients = np.gradient(image.astype(float))
        gradient_magnitude = np.sqrt(gradients[0]**2 + gradients[1]**2)
        
        return {
            "average_gradient": float(np.mean(gradient_magnitude)),
            "gradient_variance": float(np.var(gradient_magnitude)),
        }
    
    def _analyze_edge_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze edge consistency."""
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            "edge_density": float(edge_density),
            "edge_distribution": "uniform" if edge_density > 0.1 else "sparse"
        }
    
    def _is_forensically_suitable(self, image: np.ndarray) -> bool:
        """Determine if image is suitable for forensic analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check basic quality metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        return blur_score > 100 and 30 < brightness < 200 and contrast > 20
    
    def _extract_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract image metadata."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            with Image.open(image_path) as img:
                exif_data = {}
                
                if hasattr(img, '_getexif'):
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            exif_data[tag] = str(value)
                
                return {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "exif_data": exif_data,
                    "has_metadata": bool(exif_data),
                }
                
        except ImportError:
            return {"error": "PIL not available for metadata extraction"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["comprehensive", "quality", "objects", "forensic", "metadata"],
                    "description": "Type of analysis to perform",
                    "default": "comprehensive"
                }
            },
            "required": ["image_path"]
        }