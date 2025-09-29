#!/usr/bin/env python3
"""
Download real demo images for object detection
==============================================

This script downloads high-quality real images from free sources
that are perfect for demonstrating open vocabulary object detection.
"""

import requests
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_image(url: str, filename: str, description: str = "") -> bool:
    """Download an image from URL to local file."""
    if os.path.exists(filename):
        logger.info(f"✅ {filename} already exists")
        return True

    try:
        logger.info(f"📥 Downloading: {description}")
        logger.info(f"🔗 URL: {url}")

        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Create directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'wb') as f:
            f.write(response.content)

        file_size = len(response.content) / 1024  # KB
        logger.info(f"✅ Downloaded {filename} ({file_size:.1f} KB)")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False


def download_demo_images():
    """Download a collection of demo images suitable for object detection."""

    # Create demo images directory
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)

    # Curated list of direct image URLs from free sources
    # These are direct links to images that work well for object detection
    demo_images = [
        {
            "url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&auto=format&fit=crop&q=60",
            "filename": "demo_images/city_street.jpg",
            "description": "City street with cars and buildings",
            "objects": "cars, buildings, street, traffic, urban scene"
        },
        {
            "url": "https://images.unsplash.com/photo-1551845041-63e8e76e7e93?w=800&auto=format&fit=crop&q=60",
            "filename": "demo_images/office_desk.jpg",
            "description": "Modern office desk setup",
            "objects": "computer, monitor, keyboard, mouse, desk, chair, coffee cup"
        },
        {
            "url": "https://images.unsplash.com/photo-1556909114-7e4c5b7e1f5b?w=800&auto=format&fit=crop&q=60",
            "filename": "demo_images/living_room.jpg",
            "description": "Modern living room interior",
            "objects": "sofa, table, lamp, books, plants, furniture"
        },
        {
            "url": "https://images.unsplash.com/photo-1493836512294-502baa1986e2?w=800&auto=format&fit=crop&q=60",
            "filename": "demo_images/restaurant.jpg",
            "description": "Restaurant interior with people",
            "objects": "people, tables, chairs, food, restaurant, dining"
        },
        {
            "url": "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=800&auto=format&fit=crop&q=60",
            "filename": "demo_images/park_people.jpg",
            "description": "People in a park setting",
            "objects": "people, trees, bench, park, outdoor, walking"
        },
        {
            "url": "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=800&auto=format&fit=crop&q=60",
            "filename": "demo_images/cars_parking.jpg",
            "description": "Cars in parking area",
            "objects": "cars, vehicles, parking, road, automotive"
        }
    ]

    print("🖼️  Downloading Demo Images for Object Detection")
    print("=" * 60)

    successful_downloads = 0

    for i, img_info in enumerate(demo_images, 1):
        print(f"\n📷 Image {i}/{len(demo_images)}: {img_info['description']}")
        print(f"🎯 Expected objects: {img_info['objects']}")

        success = download_image(
            url=img_info["url"],
            filename=img_info["filename"],
            description=img_info["description"]
        )

        if success:
            successful_downloads += 1

    print(f"\n📊 Download Summary:")
    print(f"✅ Successfully downloaded: {successful_downloads}/{len(demo_images)} images")
    print(f"📁 Images saved to: {demo_dir.absolute()}")

    if successful_downloads > 0:
        print(f"\n🚀 Ready to run object detection!")
        print(f"💡 Try: uv run python demo_detection.py")
        print(f"💡 Or: uv run python open_vocab_detection.py")
    else:
        print(f"\n⚠️  No images downloaded. You may need to:")
        print(f"   - Check your internet connection")
        print(f"   - Use different image URLs")
        print(f"   - Manually download images to demo_images/ folder")

    return successful_downloads > 0


if __name__ == "__main__":
    download_demo_images()