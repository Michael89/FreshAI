#!/usr/bin/env python3
"""
Simple Open Vocabulary Object Detection Demo
==========================================

This script provides a simplified demonstration of open vocabulary object detection
without requiring external image downloads. It creates simple test images and
demonstrates the detection capabilities.

Run with:
    python demo_detection.py

Or with custom image:
    python demo_detection.py path/to/image.jpg "person,car,building"
"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging

# Import our open vocabulary detector
from open_vocab_detection import OpenVocabDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_images():
    """Create simple test images for demonstration."""
    demo_dir = Path("test_images")
    demo_dir.mkdir(exist_ok=True)

    test_images = []

    # Test image 1: Simple shapes (simulating objects)
    img1_path = demo_dir / "test_scene_1.jpg"
    if not img1_path.exists():
        img1 = Image.new('RGB', (800, 600), color='lightblue')
        draw = ImageDraw.Draw(img1)

        # Draw some basic shapes to represent objects
        # Rectangle (building/car)
        draw.rectangle([100, 200, 300, 400], fill='gray', outline='black', width=3)
        draw.text((150, 300), "BUILDING", fill='white')

        # Circle (person/head)
        draw.ellipse([400, 150, 500, 250], fill='peachpuff', outline='black', width=3)
        draw.text((420, 200), "HEAD", fill='black')

        # Rectangle (car)
        draw.rectangle([500, 400, 700, 500], fill='red', outline='black', width=3)
        draw.text((570, 440), "CAR", fill='white')

        # Small rectangle (phone/device)
        draw.rectangle([200, 100, 280, 180], fill='black', outline='gray', width=2)
        draw.text((220, 130), "PHONE", fill='white')

        img1.save(img1_path)
        logger.info(f"Created test image: {img1_path}")

    test_images.append(str(img1_path))

    # Test image 2: Office scene
    img2_path = demo_dir / "test_office.jpg"
    if not img2_path.exists():
        img2 = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img2)

        # Desk
        draw.rectangle([50, 300, 750, 350], fill='brown', outline='black', width=2)

        # Computer monitor
        draw.rectangle([200, 150, 400, 300], fill='black', outline='gray', width=3)
        draw.rectangle([210, 160, 390, 280], fill='blue', outline='white', width=1)
        draw.text((280, 210), "MONITOR", fill='white')

        # Keyboard
        draw.rectangle([180, 360, 350, 400], fill='gray', outline='black', width=2)
        draw.text((240, 375), "KEYBOARD", fill='black')

        # Mouse
        draw.ellipse([370, 360, 420, 400], fill='black', outline='gray', width=2)
        draw.text((385, 375), "M", fill='white')

        # Coffee cup
        draw.ellipse([500, 320, 550, 370], fill='white', outline='brown', width=3)
        draw.text((515, 340), "CUP", fill='brown')

        # Papers
        draw.rectangle([580, 310, 680, 390], fill='white', outline='black', width=2)
        draw.text((610, 345), "PAPER", fill='black')

        img2.save(img2_path)
        logger.info(f"Created test image: {img2_path}")

    test_images.append(str(img2_path))

    return test_images


def run_simple_demo():
    """Run a simple demonstration of object detection."""
    print("🔍 Open Vocabulary Object Detection Demo")
    print("=" * 50)

    # Create test images
    test_images = create_test_images()

    # Initialize detector
    print("\n📥 Initializing OWL-ViT model...")
    try:
        detector = OpenVocabDetector()
        detector.initialize()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        print("\n💡 Make sure you have the required dependencies:")
        print("   uv sync")
        print("   # or")
        print("   pip install transformers torch matplotlib pillow")
        return

    # Test scenarios
    test_scenarios = [
        {
            "name": "Scene Analysis",
            "queries": ["person", "building", "car", "phone", "object"],
            "description": "General scene objects"
        },
        {
            "name": "Office Equipment",
            "queries": ["computer", "monitor", "keyboard", "mouse", "cup", "paper", "desk"],
            "description": "Office and work-related items"
        }
    ]

    # Run detection on each test image
    for i, image_path in enumerate(test_images):
        scenario = test_scenarios[i % len(test_scenarios)]

        print(f"\n📸 Processing: {Path(image_path).name}")
        print(f"🎯 Scenario: {scenario['name']} - {scenario['description']}")
        print(f"🔎 Looking for: {', '.join(scenario['queries'])}")

        try:
            # Run detection
            results = detector.detect_objects(
                image_path=image_path,
                text_queries=scenario['queries'],
                confidence_threshold=0.1,
                max_detections=15
            )

            # Display results
            print(f"⏱️  Processing time: {results['processing_time']:.2f}s")
            print(f"📊 Total detections: {results['total_detections']}")

            if results['detections']:
                print("🎯 Detected objects:")
                for j, detection in enumerate(results['detections'][:8]):
                    confidence_emoji = "🟢" if detection['score'] > 0.3 else "🟡" if detection['score'] > 0.15 else "🔴"
                    print(f"   {j+1}. {confidence_emoji} {detection['label']}: {detection['score']:.3f}")
            else:
                print("❌ No objects detected above threshold")

            # Save visualization
            output_path = f"detection_demo_{i+1}.jpg"
            detector.visualize_detections(
                results=results,
                output_path=output_path,
                show_image=False
            )
            print(f"💾 Visualization saved: {output_path}")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

    print(f"\n🎉 Demo completed!")
    print("📁 Check the generated files:")
    print("   - test_images/: Created test images")
    print("   - detection_demo_*.jpg: Detection visualizations")
    print("\n💡 Tips:")
    print("   - Adjust confidence_threshold for more/fewer detections")
    print("   - Try different text queries to detect various objects")
    print("   - Use your own images: python demo_detection.py image.jpg 'queries'")


def run_custom_detection(image_path: str, queries_str: str):
    """Run detection on custom image with custom queries."""
    queries = [q.strip() for q in queries_str.split(',')]

    print(f"🔍 Custom Detection")
    print(f"📸 Image: {image_path}")
    print(f"🎯 Queries: {queries}")

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    try:
        detector = OpenVocabDetector()
        detector.initialize()

        results = detector.detect_objects(
            image_path=image_path,
            text_queries=queries,
            confidence_threshold=0.05
        )

        print(f"\n📊 Results: {results['total_detections']} detections")
        for detection in results['detections']:
            confidence_emoji = "🟢" if detection['score'] > 0.3 else "🟡" if detection['score'] > 0.15 else "🔴"
            print(f"   {confidence_emoji} {detection['label']}: {detection['score']:.3f}")

        # Show visualization
        detector.visualize_detections(results, show_image=True)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom mode
        image_path = sys.argv[1]
        queries = sys.argv[2] if len(sys.argv) > 2 else "person,object,building,car"
        run_custom_detection(image_path, queries)
    else:
        # Demo mode
        run_simple_demo()