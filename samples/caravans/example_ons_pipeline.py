"""
Example: ONS Caravan Counting Pipeline

This example demonstrates how to use the ONS methodology to:
1. Prepare training data from satellite imagery and caravan polygons
2. Train a Mask R-CNN model
3. Run inference to count caravans
4. Evaluate model accuracy

Prerequisites:
- Satellite imagery (GeoTIFF or JPEG tiles)
- CSV file with caravan polygon geometries
- CSV file with tile metadata (for georeferencing)

Directory Structure Expected:
    project/
    ├── imagery/
    │   └── tiles/           # Original satellite tiles (4000x4000)
    ├── caravans.csv         # Caravan polygons with geometry column
    ├── tiles.csv            # Tile metadata with extents
    └── output/              # Will be created by pipeline
        ├── geotiffs/        # Georeferenced tiles
        ├── train/images/    # Training patches
        ├── train/labels/1/  # Training masks
        ├── val/images/      # Validation patches
        ├── val/labels/1/    # Validation masks
        ├── test/images/     # Test patches
        ├── test/labels/1/   # Test masks
        ├── logs/            # Training logs
        └── results/         # Detection results
"""

import os
import sys

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from samples.caravans.ons_pipeline import ONSCaravanPipeline


def example_full_pipeline():
    """
    Example: Run the complete ONS pipeline from scratch.
    """
    print("=" * 70)
    print("ONS CARAVAN COUNTING PIPELINE - FULL EXAMPLE")
    print("=" * 70)

    # Initialize pipeline
    # Replace these paths with your actual data locations
    pipeline = ONSCaravanPipeline(
        imagery_dir="/path/to/your/imagery",
        caravan_csv="/path/to/your/caravans.csv",
        output_dir="/path/to/output"
    )

    # Step 1: Georeference tiles (if not already georeferenced)
    # The tiles CSV should have columns: tile_name, min_x, min_y, max_x, max_y, localpath
    print("\n[STEP 1] Georeferencing image tiles...")
    pipeline.georeference_tiles("/path/to/tiles.csv")

    # Step 2: Create training data patches
    # This extracts 224x224 patches centered on each caravan
    print("\n[STEP 2] Creating training data patches...")
    pipeline.create_training_data(sample_size=1000)  # Sample 1000 caravans

    # Step 3: Label masks with instance IDs
    # Converts binary masks to instance-labeled masks
    print("\n[STEP 3] Labeling masks with instance IDs...")
    pipeline.label_masks()

    # Step 4: Train the model
    # Uses COCO pre-trained weights, trains heads for 50 epochs
    print("\n[STEP 4] Training Mask R-CNN model...")
    pipeline.train(epochs=50, weights='coco')

    # Step 5: Evaluate the model
    # Computes F1 scores and other ONS metrics
    print("\n[STEP 5] Evaluating model accuracy...")
    results = pipeline.evaluate()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Average F1 Score: {results['avg_f1']:.4f}")
    print("=" * 70)


def example_training_only():
    """
    Example: Train model with existing prepared data.

    Use this if you already have your data in the ONS format:
    - train/images/*.tif
    - train/labels/1/*.png
    - val/images/*.tif
    - val/labels/1/*.png
    """
    print("=" * 70)
    print("ONS CARAVAN COUNTING - TRAINING ONLY")
    print("=" * 70)

    pipeline = ONSCaravanPipeline(
        imagery_dir=".",  # Not used for training
        caravan_csv=".",  # Not used for training
        output_dir="/path/to/prepared/data"  # Contains train/val folders
    )

    # Train with default ONS settings
    pipeline.train(
        epochs=50,
        layers='heads',
        weights='coco'
    )


def example_inference():
    """
    Example: Run inference on new images.
    """
    print("=" * 70)
    print("ONS CARAVAN DETECTION - INFERENCE")
    print("=" * 70)

    from samples.caravans.detect_caravans import load_model, detect_caravans, get_caravan_count

    # Load trained model
    model = load_model(
        weights_path="/path/to/trained/weights.h5",
        logs_dir="/path/to/logs"
    )

    # Detect caravans in an image
    results, image = detect_caravans(model, "/path/to/satellite/image.tif")

    # Get count
    count = get_caravan_count(results)
    print(f"Detected {count} caravans")

    # Print details
    for i, score in enumerate(results['scores']):
        print(f"  Caravan {i+1}: confidence={score:.3f}")


def example_batch_inference():
    """
    Example: Run inference on a folder of images.
    """
    print("=" * 70)
    print("ONS CARAVAN DETECTION - BATCH INFERENCE")
    print("=" * 70)

    from samples.caravans.detect_caravans import load_model, batch_detect

    # Load trained model
    model = load_model(
        weights_path="last",  # Use most recent trained weights
        logs_dir="/path/to/logs"
    )

    # Process all images in folder
    results = batch_detect(
        model,
        folder_path="/path/to/images",
        output_dir="/path/to/output",
        visualize=True  # Create visualization images
    )

    # Print summary
    total = sum(r['count'] for r in results.values() if 'count' in r)
    print(f"\nTotal caravans detected across all images: {total}")


def example_evaluation():
    """
    Example: Evaluate model on test set.
    """
    print("=" * 70)
    print("ONS CARAVAN MODEL EVALUATION")
    print("=" * 70)

    pipeline = ONSCaravanPipeline(
        imagery_dir=".",
        caravan_csv=".",
        output_dir="/path/to/data/with/test/set"
    )

    # Evaluate using last trained weights
    metrics = pipeline.evaluate(weights_path="last")

    # Print detailed results
    print("\nF1 Score Distribution:")
    f1_scores = metrics['f1_scores']
    print(f"  Min:    {min(f1_scores):.4f}")
    print(f"  Max:    {max(f1_scores):.4f}")
    print(f"  Mean:   {metrics['avg_f1']:.4f}")
    print(f"  StdDev: {metrics['std_f1']:.4f}")

    # Count images by F1 score range
    excellent = sum(1 for f in f1_scores if f >= 0.9)
    good = sum(1 for f in f1_scores if 0.7 <= f < 0.9)
    poor = sum(1 for f in f1_scores if f < 0.7)

    print(f"\nScore Distribution:")
    print(f"  Excellent (F1 >= 0.9): {excellent} images")
    print(f"  Good (0.7 <= F1 < 0.9): {good} images")
    print(f"  Poor (F1 < 0.7): {poor} images")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ONS Pipeline Examples')
    parser.add_argument('example',
                        choices=['full', 'train', 'infer', 'batch', 'eval'],
                        help='Which example to run')

    args = parser.parse_args()

    if args.example == 'full':
        example_full_pipeline()
    elif args.example == 'train':
        example_training_only()
    elif args.example == 'infer':
        example_inference()
    elif args.example == 'batch':
        example_batch_inference()
    elif args.example == 'eval':
        example_evaluation()
