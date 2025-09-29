#!/usr/bin/env python3
"""
Quick test script for object detection with very low threshold
"""

from open_vocab_detection import OpenVocabDetector
from pathlib import Path

def main():
    # Test with very low confidence threshold
    detector = OpenVocabDetector()

    image_path = "demo_images/city_street.jpg"
    queries = ["person", "car", "vehicle", "building", "street", "tree", "window", "sign", "road"]

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return

    print(f"Testing detection on {image_path}")
    print(f"Queries: {queries}")
    print(f"Using very low confidence threshold: 0.01")

    try:
        results = detector.detect_objects(
            image_path=image_path,
            text_queries=queries,
            confidence_threshold=0.01,  # Very low threshold
            max_detections=50
        )

        print(f"\nResults:")
        print(f"Total detections: {results['total_detections']}")
        print(f"Processing time: {results['processing_time']:.2f}s")

        if results['detections']:
            print("\nDetections (sorted by confidence):")
            sorted_detections = sorted(results['detections'], key=lambda x: x['score'], reverse=True)
            for i, detection in enumerate(sorted_detections[:15]):
                print(f"  {i+1:2d}. {detection['label']:12s}: {detection['score']:.4f}")

            # Create visualization
            detector.visualize_detections(
                results,
                output_path="test_detection_result.jpg",
                show_image=False
            )
            print(f"\nVisualization saved as: test_detection_result.jpg")
        else:
            print("No detections found even with very low threshold")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()