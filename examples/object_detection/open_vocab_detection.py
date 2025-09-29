#!/usr/bin/env python3
"""
Open Vocabulary Object Detection using Transformers
===================================================

This script demonstrates open vocabulary object detection using OWL-ViT
(Open-World Localization Vision Transformer) from Hugging Face Transformers.

OWL-ViT can detect objects based on text descriptions without being
specifically trained on those object classes.

Features:
- Detect arbitrary objects using text descriptions
- Draw bounding boxes with labels and confidence scores
- Support for batch processing multiple images
- Customizable confidence thresholds
- Crime investigation focused examples

Requirements:
- transformers
- torch
- PIL (Pillow)
- matplotlib
- requests (for downloading demo images)
"""

import os
import time
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from io import BytesIO

try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenVocabDetector:
    """Open vocabulary object detector using OWL-ViT."""

    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str = "auto"):
        """
        Initialize the open vocabulary detector.

        Args:
            model_name: HuggingFace model name for OWL-ViT
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.processor = None
        self.model = None
        self._initialized = False

    def _determine_device(self, device: str) -> str:
        """Determine the appropriate device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def initialize(self) -> None:
        """Initialize the model and processor."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")

        if self._initialized:
            return

        logger.info(f"Loading OWL-ViT model: {self.model_name}")
        start_time = time.time()

        try:
            self.processor = OwlViTProcessor.from_pretrained(self.model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(self.model_name)

            if self.device != "cpu":
                self.model = self.model.to(self.device)

            self._initialized = True
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def detect_objects(
        self,
        image_path: str,
        text_queries: List[str],
        confidence_threshold: float = 0.1,
        max_detections: int = 100
    ) -> Dict:
        """
        Detect objects in an image based on text queries.

        Args:
            image_path: Path to the input image
            text_queries: List of text descriptions of objects to detect
            confidence_threshold: Minimum confidence score for detections
            max_detections: Maximum number of detections to return

        Returns:
            Dictionary containing detection results
        """
        if not self._initialized:
            self.initialize()

        # Load and validate image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise Exception(f"Failed to load image: {e}")

        logger.info(f"Detecting objects in {image_path}")
        logger.info(f"Text queries: {text_queries}")

        start_time = time.time()

        try:
            # Process inputs
            inputs = self.processor(text=text_queries, images=image, return_tensors="pt")

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Process outputs
            target_sizes = torch.Tensor([image.size[::-1]])  # (height, width)

            # Use the newer API if available
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=confidence_threshold
                )
            except (AttributeError, TypeError):
                # Fallback to older API
                results = self.processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=confidence_threshold
                )

            processing_time = time.time() - start_time

            # Format results
            detections = []
            boxes = results[0]["boxes"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()
            labels = results[0]["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if len(detections) >= max_detections:
                    break

                detection = {
                    "box": box.tolist(),  # [x1, y1, x2, y2]
                    "score": float(score),
                    "label": text_queries[label],
                    "label_id": int(label)
                }
                detections.append(detection)

            return {
                "image_path": image_path,
                "image_size": image.size,
                "text_queries": text_queries,
                "detections": detections,
                "processing_time": processing_time,
                "total_detections": len(detections),
                "confidence_threshold": confidence_threshold
            }

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            raise

    def visualize_detections(
        self,
        results: Dict,
        output_path: Optional[str] = None,
        show_image: bool = True,
        font_size: int = 12
    ) -> None:
        """
        Visualize detection results with bounding boxes and labels.

        Args:
            results: Detection results from detect_objects()
            output_path: Path to save the visualization (optional)
            show_image: Whether to display the image
            font_size: Font size for labels
        """
        image_path = results["image_path"]
        detections = results["detections"]

        # Load image
        image = Image.open(image_path)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        # Colors for different labels
        colors = plt.cm.Set3(torch.linspace(0, 1, len(results["text_queries"])))

        # Draw bounding boxes
        for detection in detections:
            box = detection["box"]
            score = detection["score"]
            label = detection["label"]
            label_id = detection["label_id"]

            # Box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=colors[label_id % len(colors)],
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add label with confidence score
            ax.text(
                x1, y1 - 5,
                f"{label}: {score:.2f}",
                fontsize=font_size,
                color=colors[label_id % len(colors)],
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )

        ax.set_title(
            f"Open Vocabulary Object Detection\n"
            f"Found {len(detections)} objects in {results['processing_time']:.2f}s",
            fontsize=14,
            weight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")

        if show_image:
            plt.show()
        else:
            plt.close()


def download_demo_image(url: str, filename: str) -> str:
    """Download a demo image from URL."""
    if os.path.exists(filename):
        logger.info(f"Demo image {filename} already exists")
        return filename

    logger.info(f"Downloading demo image: {filename}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            f.write(response.content)

        logger.info(f"Demo image saved: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def create_demo_images() -> List[str]:
    """Create or download demo images for testing."""
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)

    # Demo image URLs (using free images)
    demo_images = [
        {
            "url": "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=800",
            "filename": "demo_images/car_street.jpg",
            "description": "Street scene with cars"
        },
        {
            "url": "https://images.unsplash.com/photo-1551845041-63e8e76e7e93?w=800",
            "filename": "demo_images/office_desk.jpg",
            "description": "Office desk with computer and objects"
        },
        {
            "url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800",
            "filename": "demo_images/living_room.jpg",
            "description": "Living room with furniture"
        }
    ]

    downloaded_files = []
    for demo in demo_images:
        try:
            filename = download_demo_image(demo["url"], demo["filename"])
            downloaded_files.append(filename)
        except Exception as e:
            logger.warning(f"Could not download {demo['filename']}: {e}")

    return downloaded_files


def run_crime_investigation_demo():
    """Run a demonstration focused on crime investigation scenarios."""
    print("=" * 60)
    print("Open Vocabulary Object Detection - Crime Investigation Demo")
    print("=" * 60)

    # Initialize detector
    detector = OpenVocabDetector()

    try:
        detector.initialize()
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        print("Please install required dependencies: pip install transformers torch")
        return

    # Use real demo images if available, otherwise use generated ones
    demo_images = []

    # Check for downloaded real images first
    real_images = [
        "demo_images/city_street.jpg",
        "demo_images/restaurant.jpg",
        "demo_images/park_people.jpg"
    ]

    for img_path in real_images:
        if Path(img_path).exists():
            demo_images.append(img_path)

    # Fallback to generated images if no real images available
    if not demo_images:
        try:
            demo_images = create_demo_images()
            if not demo_images:
                print("No demo images available. Please provide image paths manually.")
                return
        except Exception as e:
            print(f"Failed to create demo images: {e}")
            return

    # Crime investigation related queries
    investigation_queries = [
        # General objects
        ["person", "car", "building", "window", "door"],

        # Evidence and items of interest
        ["weapon", "knife", "gun", "bottle", "bag", "phone", "computer", "camera"],

        # Furniture and indoor items
        ["chair", "desk", "table", "laptop", "monitor", "keyboard", "mouse", "papers"],

        # Transportation
        ["vehicle", "license plate", "motorcycle", "bicycle", "tire", "wheel"]
    ]

    # Run detection on each demo image
    for i, image_path in enumerate(demo_images):
        print(f"\n--- Processing Image {i+1}: {image_path} ---")

        # Use different query set for each image
        queries = investigation_queries[i % len(investigation_queries)]

        try:
            # Detect objects
            results = detector.detect_objects(
                image_path=image_path,
                text_queries=queries,
                confidence_threshold=0.15,
                max_detections=20
            )

            # Print results
            print(f"Processing time: {results['processing_time']:.2f}s")
            print(f"Total detections: {results['total_detections']}")

            if results['detections']:
                print("\nDetected objects:")
                for j, detection in enumerate(results['detections'][:10]):  # Show top 10
                    print(f"  {j+1}. {detection['label']}: {detection['score']:.3f}")
            else:
                print("No objects detected above threshold")

            # Create visualization
            output_path = f"detection_result_{i+1}.jpg"
            detector.visualize_detections(
                results=results,
                output_path=output_path,
                show_image=False  # Set to True to display images
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"\n{'-'*60}")
    print("Demo completed! Check the generated visualization files.")
    print("Tip: Set show_image=True in visualize_detections() to display images.")


def run_custom_detection(image_path: str, queries: List[str]):
    """Run object detection on a custom image with custom queries."""
    print(f"\nRunning custom detection on: {image_path}")
    print(f"Queries: {queries}")

    detector = OpenVocabDetector()

    try:
        results = detector.detect_objects(
            image_path=image_path,
            text_queries=queries,
            confidence_threshold=0.1
        )

        print(f"Found {results['total_detections']} objects")
        for detection in results['detections']:
            print(f"- {detection['label']}: {detection['score']:.3f}")

        # Visualize
        detector.visualize_detections(results, show_image=True)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Custom mode: python open_vocab_detection.py image.jpg "person,car,building"
        image_path = sys.argv[1]
        queries = sys.argv[2].split(",") if len(sys.argv) > 2 else ["person", "object"]
        run_custom_detection(image_path, queries)
    else:
        # Demo mode
        run_crime_investigation_demo()