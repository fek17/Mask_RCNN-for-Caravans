# ONS Caravan Counting Methodology

This document describes the Office for National Statistics (ONS) methodology for counting caravans from satellite imagery using Mask R-CNN, and how to replicate it using this codebase.

## Overview

The ONS methodology uses instance segmentation (Mask R-CNN) to detect individual caravan footprints from high-resolution satellite imagery. The pipeline consists of five main steps:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1. Georeference │ -> │ 2. Create       │ -> │ 3. Label        │
│    Image Tiles  │    │    Training     │    │    Instance     │
│    (EPSG:27700) │    │    Patches      │    │    Masks        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                       ┌─────────────────┐    ┌───────┴───────┐
                       │ 5. Inference &  │ <- │ 4. Train      │
                       │    Evaluation   │    │    Mask R-CNN │
                       └─────────────────┘    └───────────────┘
```

## Data Requirements

### Input Data

1. **Satellite Imagery**: High-resolution tiles (typically 4000x4000 pixels at 0.25m resolution)
2. **Caravan Polygons**: CSV/shapefile with caravan boundary geometries
3. **Tile Metadata**: CSV with tile extents for georeferencing

### Data Format

**caravans.csv** should contain:
```csv
FID,geometry,tile_name
1,"POLYGON ((259472.92 279785.36, ...))",SH7719
2,"POLYGON ((259526.13 279812.91, ...))",SH7719
```

**tiles.csv** should contain:
```csv
tile_name,min_x,min_y,max_x,max_y,localpath
SH7719,276999.9971,318999.9965,277999.9993,320000.0002,/path/to/SH7719.jpg
```

## Pipeline Steps

### Step 1: Georeference Image Tiles

Convert raw imagery to georeferenced GeoTIFFs using British National Grid (EPSG:27700).

```python
from ons_pipeline import ONSCaravanPipeline

pipeline = ONSCaravanPipeline(
    imagery_dir="/path/to/imagery",
    caravan_csv="/path/to/caravans.csv",
    output_dir="/path/to/output"
)

pipeline.georeference_tiles("/path/to/tiles.csv")
```

**Or via command line:**
```bash
python ons_pipeline.py georeference --tiles=/path/to/tiles.csv --output=/path/to/output
```

### Step 2: Create Training Data Patches

Extract 224x224 pixel patches centered on each caravan centroid, along with corresponding binary masks.

```python
pipeline.create_training_data(sample_size=1000)
```

**Command line:**
```bash
python ons_pipeline.py prepare --caravans=/path/to/caravans.csv --output=/path/to/output --sample=1000
```

### Step 3: Label Instance Masks

Convert binary masks to instance-labeled masks using `skimage.measure.label()`. Each connected component (individual caravan) receives a unique ID.

```python
pipeline.label_masks()
```

**Command line:**
```bash
python ons_pipeline.py label --output=/path/to/output
```

### Step 4: Train Mask R-CNN

Train the model using COCO pre-trained weights. The ONS methodology trains the head layers for 50 epochs.

```python
pipeline.train(epochs=50, weights='coco', layers='heads')
```

**Command line:**
```bash
python ons_pipeline.py train --output=/path/to/data --weights=coco --epochs=50
```

### Step 5: Evaluate Model

Compute pixel-level F1 scores and confusion matrices on the test set.

```python
metrics = pipeline.evaluate(weights_path='last')
print(f"Average F1: {metrics['avg_f1']:.4f}")
```

**Command line:**
```bash
python ons_pipeline.py evaluate --output=/path/to/data --weights=last
```

## Model Configuration

The ONS methodology uses these exact parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BACKBONE` | `resnet101` | ResNet-101 feature extractor |
| `NUM_CLASSES` | `2` | Background + Caravan |
| `DETECTION_MIN_CONFIDENCE` | `0.9` | 90% confidence threshold |
| `RPN_ANCHOR_SCALES` | `(32, 64, 128, 256, 512)` | Anchor box sizes |
| `STEPS_PER_EPOCH` | `315` | Training steps per epoch |
| `VALIDATION_STEPS` | `20` | Validation steps |
| `IMAGES_PER_GPU` | `2` | Batch size |
| `IMAGE_SIZE` | `224x224` | Patch dimensions |

## Directory Structure

After running the pipeline, your output directory will contain:

```
output/
├── geotiffs/              # Georeferenced image tiles
├── train/
│   ├── images/            # 224x224 training patches (.tif)
│   └── labels/
│       └── 1/             # Instance-labeled masks (.png)
├── val/
│   ├── images/            # Validation patches
│   └── labels/1/          # Validation masks
├── test/
│   ├── images/            # Test patches
│   └── labels/1/          # Test masks
├── logs/                  # Training logs and weights
│   └── caravan{timestamp}/
│       ├── events.out.*   # TensorBoard logs
│       └── mask_rcnn_caravan_*.h5  # Model weights
└── results/
    └── masks/             # Detection results
```

## Running Inference

### Single Image Detection

```python
from detect_caravans import load_model, detect_caravans, get_caravan_count

model = load_model('last')
results, image = detect_caravans(model, '/path/to/image.tif')
count = get_caravan_count(results)
print(f"Detected {count} caravans")
```

**Command line:**
```bash
python detect_caravans.py detect --image=/path/to/image.tif --weights=last --visualize
```

### Batch Processing

```bash
python detect_caravans.py batch --folder=/path/to/images --weights=last --output=/path/to/results --visualize
```

## Accuracy Metrics

The ONS methodology evaluates accuracy using:

1. **Pixel-level F1 Score**: Harmonic mean of precision and recall at the pixel level
2. **Confusion Matrix**: True positives, false positives, true negatives, false negatives

Expected performance (from ONS testing):
- Average F1 Score: ~0.84 (83.8%)
- Best cases: F1 > 0.93
- Some challenging cases: F1 < 0.50 (complex scenes, shadows, etc.)

## Troubleshooting

### Memory Issues
- Reduce `IMAGES_PER_GPU` in config
- Use smaller `sample_size` for training data

### Poor Detection Results
- Ensure sufficient training data (>500 samples recommended)
- Check that masks are properly aligned with images
- Verify coordinate reference system matches between tiles and polygons

### Training Divergence
- Lower learning rate (modify `LEARNING_RATE` in config)
- Ensure COCO weights are downloaded correctly

## Files in This Directory

| File | Purpose |
|------|---------|
| `ons_pipeline.py` | Complete ONS methodology pipeline |
| `detect_caravans.py` | Inference and detection script |
| `caravan.py` | Original training configuration |
| `example_ons_pipeline.py` | Usage examples |
| `making_training_data.ipynb` | Original notebook (reference) |
| `label_masks.ipynb` | Original notebook (reference) |
| `accuracy_metrics.ipynb` | Original notebook (reference) |

## References

- Matterport Mask R-CNN: https://github.com/matterport/Mask_RCNN
- Mask R-CNN Paper: https://arxiv.org/abs/1703.06870
- British National Grid: EPSG:27700
