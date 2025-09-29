# Grounding DINO Usage Guide

## The Label Combination Issue and Solution

### What Was Happening
Grounding DINO sometimes generates compound labels like:
- `"weapon gun"` instead of `"weapon"` or `"gun"`
- `"person man"` instead of `"person"`
- `"car vehicle"` instead of `"car"`

### Why This Happens
Unlike OWL-ViT which processes each query independently, Grounding DINO:
1. Takes a single concatenated text string as input
2. Can generate more detailed/compound descriptions
3. Sometimes combines related terms from your query list

### How We Fixed It
Our implementation now includes automatic label cleaning that:
1. **Maps compound labels back** to your original queries
2. **Handles common cases** like "weapon gun" → "weapon"
3. **Uses intelligent matching** to find the best original query
4. **Preserves accuracy** while providing clean output

## Command Examples

### Basic Usage
```bash
# Simple detection
uv run python examples/object_detection/advanced_detection.py image.jpg "person,car,building"
```

### Crime Investigation Scenarios
```bash
# Evidence detection (clean labels guaranteed)
uv run python examples/object_detection/advanced_detection.py evidence.jpg "weapon,knife,gun,evidence,device"

# Scene analysis
uv run python examples/object_detection/advanced_detection.py scene.jpg "person,vehicle,building,container,bag"

# Indoor investigation
uv run python examples/object_detection/advanced_detection.py indoor.jpg "computer,phone,document,bottle,device"
```

### Model Selection
```bash
# Highest accuracy (slower)
uv run python examples/object_detection/advanced_detection.py --model grounding-dino-base image.jpg "queries"

# Best speed/accuracy balance (recommended)
uv run python examples/object_detection/advanced_detection.py --model grounding-dino-tiny image.jpg "queries"

# Compare all models
uv run python examples/object_detection/advanced_detection.py --benchmark image.jpg "person,car"
```

## Python API Examples

### Basic Detection with Clean Labels
```python
from examples.object_detection.advanced_detection import AdvancedDetector

detector = AdvancedDetector(model_key="grounding-dino-tiny")

results = detector.detect_objects(
    image_path="evidence.jpg",
    text_queries=["weapon", "gun", "knife", "evidence"],
    confidence_threshold=0.3
)

# Results will have clean, single-word labels
for detection in results['detections']:
    print(f"{detection['label']}: {detection['score']:.3f}")
    # Output: "weapon: 0.542" (not "weapon gun: 0.542")
```

### Batch Processing
```python
images = ["scene1.jpg", "scene2.jpg", "scene3.jpg"]
queries = ["person", "vehicle", "weapon", "evidence"]

detector = AdvancedDetector("grounding-dino-tiny")

for image_path in images:
    results = detector.detect_objects(image_path, queries, 0.25)
    print(f"{image_path}: {results['total_detections']} objects found")
```

## Expected Output Format

### Before Fix (Problematic)
```
🔍 Detected objects:
    1. 🟢 weapon gun     : 0.542
    2. 🟡 weapon gun     : 0.315
    3. 🔴 person man     : 0.284
```

### After Fix (Clean)
```
🔍 Detected objects:
    1. 🟢 weapon         : 0.542
    2. 🟡 weapon         : 0.315
    3. 🔴 person         : 0.284
```

## Performance Comparison

| Aspect | Before Fix | After Fix |
|--------|------------|-----------|
| Label Quality | ❌ Compound labels | ✅ Clean single labels |
| Accuracy | ✅ High detection | ✅ Same high detection |
| Usability | ❌ Confusing output | ✅ Clear, expected output |
| Integration | ❌ Hard to parse | ✅ Easy to process |

## Tips for Best Results

### 1. Choose Specific Queries
```bash
# Good
"person,car,weapon,knife,building"

# Avoid overlapping concepts
"weapon,gun,knife,blade"  # May create compound labels
```

### 2. Adjust Confidence Threshold
```bash
# High precision (fewer false positives)
--threshold 0.4

# High recall (catch more objects)
--threshold 0.2

# Balanced (recommended)
--threshold 0.3
```

### 3. Model Selection by Use Case
```bash
# Real-time applications
--model grounding-dino-tiny

# Maximum accuracy needed
--model grounding-dino-base

# Resource constrained (not recommended)
--model owlvit-base
```

## Troubleshooting

### Still Getting Compound Labels?
1. Check if the compound makes sense (some may be intentional)
2. Verify your query list doesn't have very similar terms
3. Try using more specific single-word queries

### Low Detection Rate?
1. Lower the confidence threshold: `--threshold 0.2`
2. Try more descriptive queries: `"red car"` instead of `"car"`
3. Use the base model for higher accuracy: `--model grounding-dino-base`

### Performance Issues?
1. Use tiny model for speed: `--model grounding-dino-tiny`
2. Reduce max_detections parameter
3. Process images in smaller batches

The label cleaning fix ensures you get the clean, expected output format while maintaining all the accuracy benefits of Grounding DINO!