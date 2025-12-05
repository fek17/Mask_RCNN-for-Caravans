"""
ONS Caravan Counting Methodology - Complete Pipeline

This module replicates the exact methodology used by the Office for National Statistics (ONS)
for counting caravans from satellite imagery using Mask R-CNN.

Pipeline Steps:
1. Georeference image tiles (EPSG:27700 - British National Grid)
2. Create training data (224x224 patches centered on caravan centroids)
3. Label masks with instance IDs
4. Train Mask R-CNN model
5. Run inference and evaluate accuracy

Usage:
    from ons_pipeline import ONSCaravanPipeline

    pipeline = ONSCaravanPipeline(
        imagery_dir="/path/to/imagery",
        caravan_csv="/path/to/caravans.csv",
        output_dir="/path/to/output"
    )

    # Step 1: Georeference tiles
    pipeline.georeference_tiles(tiles_csv="/path/to/tiles.csv")

    # Step 2: Create training patches
    pipeline.create_training_data(sample_size=1000)

    # Step 3: Label masks
    pipeline.label_masks()

    # Step 4: Train model
    pipeline.train(epochs=50, weights="coco")

    # Step 5: Evaluate
    pipeline.evaluate()

Copyright (c) 2024
Based on ONS methodology and Matterport Mask R-CNN implementation
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rst
import rasterio.mask as mask
from rasterio import Affine
from shapely.geometry import Point, box
from shapely.wkt import loads
from skimage.measure import label
from skimage.io import imread, imsave
from sklearn.metrics import confusion_matrix, f1_score
from glob import glob
from copy import deepcopy
import warnings
import argparse

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

# Mask R-CNN imports
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Suppress warnings
warnings.filterwarnings('ignore')


############################################################
#  Configuration - Matches ONS Methodology Exactly
############################################################

class ONSCaravanConfig(Config):
    """
    Configuration for training on caravan dataset.
    Matches the exact ONS methodology parameters.
    """
    # Give the configuration a recognizable name
    NAME = "caravan"

    # GPU settings - adjust based on your hardware
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Network backbone
    BACKBONE = "resnet101"

    # Number of classes (including background)
    # Background + Caravan = 2 classes
    NUM_CLASSES = 1 + 1

    # Training steps per epoch
    STEPS_PER_EPOCH = 315
    VALIDATION_STEPS = 20

    # Detection confidence threshold
    # ONS uses 90% confidence for detections
    DETECTION_MIN_CONFIDENCE = 0.9

    # Anchor scales for Region Proposal Network
    # These are tuned for caravan sizes in satellite imagery
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Image dimensions - patches are 224x224
    IMAGE_MIN_DIM = 224
    IMAGE_MAX_DIM = 224

    # Training image resizing
    IMAGE_RESIZE_MODE = "square"


class ONSInferenceConfig(ONSCaravanConfig):
    """Configuration for inference mode."""
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset Class - Matches ONS Data Structure
############################################################

class ONSCaravanDataset(utils.Dataset):
    """
    Dataset class for loading caravan training data.
    Expects the ONS directory structure:

    dataset_dir/
    ├── train/
    │   ├── images/     # 224x224 TIF patches
    │   └── labels/1/   # Instance-labeled PNG masks
    └── val/
        ├── images/
        └── labels/1/
    """

    def load_caravans(self, dataset_dir, subset):
        """
        Load a subset of the caravan dataset.

        Args:
            dataset_dir: Root directory of the dataset
            subset: "train" or "val"
        """
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add caravan class
        self.add_class("caravan", 1, "caravan")

        # Set up directories
        self._image_dir = os.path.join(dataset_dir, "images")
        self._mask_dir = os.path.join(dataset_dir, "labels")

        # Load all TIF images
        image_files = glob(os.path.join(self._image_dir, "*.tif"))

        for i, filepath in enumerate(image_files):
            filename = os.path.basename(filepath)
            self.add_image(
                "caravan",
                image_id=i,
                path=filepath,
                width=224,
                height=224,
                filename=filename
            )

    def load_mask(self, image_id):
        """
        Load instance masks for an image.

        Returns:
            masks: A bool array of shape [height, width, instance_count]
            class_ids: A 1D array of class IDs for each instance
        """
        info = self.image_info[image_id]
        filename = info["filename"].replace('.tif', '.png')

        masks = []
        class_ids = []

        # Load the instance-labeled mask
        mask_path = os.path.join(self._mask_dir, "1", filename)

        if os.path.exists(mask_path):
            m_src = imread(mask_path)

            # Each unique value (except 0) is a separate instance
            instance_ids = np.unique(m_src)

            for inst_id in instance_ids:
                if inst_id > 0:  # Skip background
                    m = np.zeros(m_src.shape, dtype=np.bool_)
                    m[m_src == inst_id] = True
                    if np.any(m):
                        masks.append(m)
                        class_ids.append(1)  # Caravan class ID

        if masks:
            masks = np.stack(masks, axis=-1)
        else:
            # Return empty mask if no instances found
            masks = np.zeros((224, 224, 0), dtype=np.bool_)

        return masks.astype(np.bool_), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


############################################################
#  ONS Pipeline Class
############################################################

class ONSCaravanPipeline:
    """
    Complete pipeline for the ONS caravan counting methodology.
    """

    def __init__(self, imagery_dir, caravan_csv, output_dir,
                 crs="EPSG:27700", patch_size=224):
        """
        Initialize the pipeline.

        Args:
            imagery_dir: Directory containing satellite image tiles
            caravan_csv: CSV file with caravan polygons and metadata
            output_dir: Output directory for processed data
            crs: Coordinate reference system (default: British National Grid)
            patch_size: Size of training patches (default: 224x224)
        """
        self.imagery_dir = imagery_dir
        self.caravan_csv = caravan_csv
        self.output_dir = output_dir
        self.crs = crs
        self.patch_size = patch_size
        self.half_patch = patch_size // 2  # 112 pixels

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create the ONS-standard directory structure."""
        dirs = [
            os.path.join(self.output_dir, "geotiffs"),
            os.path.join(self.output_dir, "train", "images"),
            os.path.join(self.output_dir, "train", "labels", "1"),
            os.path.join(self.output_dir, "val", "images"),
            os.path.join(self.output_dir, "val", "labels", "1"),
            os.path.join(self.output_dir, "test", "images"),
            os.path.join(self.output_dir, "test", "labels", "1"),
            os.path.join(self.output_dir, "logs"),
            os.path.join(self.output_dir, "results", "masks"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    ############################################################
    #  Step 1: Georeference Image Tiles
    ############################################################

    def georeference_tiles(self, tiles_csv):
        """
        Georeference image tiles using British National Grid (EPSG:27700).

        The tiles CSV should have columns:
        - tile_name: Name of the tile (e.g., "SH7719")
        - min_x, min_y, max_x, max_y: Extent coordinates
        - localpath: Path to the original image file

        Args:
            tiles_csv: Path to CSV with tile metadata
        """
        print("=" * 60)
        print("STEP 1: Georeferencing Image Tiles")
        print("=" * 60)

        df = pd.read_csv(tiles_csv)
        geotiff_dir = os.path.join(self.output_dir, "geotiffs")

        processed = 0
        for idx, row in df.iterrows():
            try:
                # Create affine transformation
                # Cell size is calculated from extent / image size (typically 0.25m)
                transform = Affine.from_gdal(
                    row['min_x'],   # x origin
                    0.25,           # pixel width (0.25m resolution)
                    0,              # rotation
                    row['max_y'],   # y origin
                    0,              # rotation
                    -0.25           # pixel height (negative for top-down)
                )

                # Read original image
                with rst.open(row['localpath']) as src:
                    image_data = src.read()
                    height, width = src.height, src.width

                # Write georeferenced GeoTIFF
                output_path = os.path.join(
                    geotiff_dir,
                    f"{row['tile_name']}.tif"
                )

                with rst.open(
                    output_path, 'w',
                    driver='GTiff',
                    compress='LZW',
                    height=height,
                    width=width,
                    count=image_data.shape[0],
                    dtype='uint8',
                    crs=self.crs,
                    transform=transform
                ) as dst:
                    dst.write(image_data)

                processed += 1
                if processed % 10 == 0:
                    print(f"  Processed {processed} tiles...")

            except Exception as e:
                print(f"  Error processing {row['tile_name']}: {e}")

        print(f"  Georeferenced {processed} tiles to {geotiff_dir}")
        return geotiff_dir

    ############################################################
    #  Step 2: Create Training Data Patches
    ############################################################

    def _get_image_coords(self, geom, xmin, ymax, cellsize):
        """Convert world coordinates to image coordinates."""
        c_x, c_y = np.array(geom.centroid.coords)[0]
        img_x = (c_x - xmin) / cellsize
        img_y = (ymax - c_y) / cellsize
        return Point(img_x, img_y)

    def _get_image_box(self, geom, xmin, ymax, cellsize):
        """Create a 224x224 box around caravan centroid in image coords."""
        c_x, c_y = np.array(geom.centroid.coords)[0]
        img_x = ((c_x - xmin) // cellsize) - self.half_patch
        img_y = ((ymax - c_y) // cellsize) - self.half_patch
        return box(img_x, img_y, img_x + self.patch_size, img_y + self.patch_size)

    def _get_real_box(self, geom, cellsize):
        """Get 224x224 box around caravan in real-world coordinates."""
        c_x, c_y = np.array(geom.centroid.coords)[0]
        half_real = self.half_patch * cellsize
        return box(c_x - half_real, c_y - half_real,
                   c_x + half_real, c_y + half_real)

    def _georeference_box(self, tile, xmin, ymin):
        """Create affine transform for subset patch."""
        trans = list(tile.get_transform())
        trans[0] = trans[0] + (xmin * trans[1])
        trans[3] = trans[3] + (ymin * trans[5])
        return Affine.from_gdal(*trans)

    def _subset_image(self, image, image_box, tile=None, georef=False):
        """Extract a subset patch from the image."""
        xmin, ymin, xmax, ymax = map(int, image_box.bounds)
        subset = image[ymin:ymax, xmin:xmax, :]

        if georef and tile:
            new_transform = self._georeference_box(tile, xmin, ymin)
            return subset, new_transform
        return subset

    def create_training_data(self, sample_size=None, train_ratio=0.7, val_ratio=0.15):
        """
        Create training data patches from georeferenced tiles and caravan CSV.

        This implements the exact ONS methodology:
        1. Load caravan polygons from CSV
        2. For each caravan, extract a 224x224 patch centered on its centroid
        3. Create corresponding binary masks
        4. Split into train/val/test sets

        Args:
            sample_size: Number of caravans to sample (None = all)
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
        """
        print("=" * 60)
        print("STEP 2: Creating Training Data Patches")
        print("=" * 60)

        # Load caravan data
        print("  Loading caravan polygons...")
        caravans = pd.read_csv(self.caravan_csv)

        # Convert to GeoDataFrame if geometry column exists
        if 'geometry' in caravans.columns:
            caravans = gpd.GeoDataFrame(
                caravans,
                crs=self.crs,
                geometry=caravans['geometry'].apply(lambda g: loads(g) if isinstance(g, str) else g)
            )

        print(f"  Loaded {len(caravans)} caravan polygons")

        # Sample if requested
        if sample_size and sample_size < len(caravans):
            caravans = caravans.sample(sample_size, random_state=42)
            print(f"  Sampled {sample_size} caravans")

        # Process each caravan
        processed = 0
        skipped = 0

        for idx, caravan in caravans.iterrows():
            try:
                # Get the geotiff path for this caravan
                if 'geotiff' in caravan:
                    tile_path = caravan['geotiff']
                elif 'tile_name' in caravan:
                    tile_path = os.path.join(
                        self.output_dir, "geotiffs",
                        f"{caravan['tile_name']}.tif"
                    )
                else:
                    continue

                if not os.path.exists(tile_path):
                    skipped += 1
                    continue

                with rst.open(tile_path, 'r') as raster:
                    xmin = raster.bounds.left
                    xmax = raster.bounds.right
                    ymin = raster.bounds.bottom
                    ymax = raster.bounds.top
                    cellsize = raster.res[0]
                    tile_box = box(xmin, ymin, xmax, ymax)

                    # Check if caravan is far enough from edges
                    real_box = self._get_real_box(caravan.geometry, cellsize)
                    if not real_box.within(tile_box):
                        skipped += 1
                        continue

                    # Read image
                    image = raster.read()
                    image = np.moveaxis(image, 0, -1)

                    # Get image box and extract subset
                    image_box = self._get_image_box(
                        caravan.geometry, xmin, ymax, cellsize
                    )
                    subset, subset_trans = self._subset_image(
                        image, image_box, raster, georef=True
                    )

                    # Determine split (train/val/test)
                    rand = np.random.random()
                    if rand < train_ratio:
                        split = "train"
                    elif rand < train_ratio + val_ratio:
                        split = "val"
                    else:
                        split = "test"

                    # Save image patch
                    img_path = os.path.join(
                        self.output_dir, split, "images", f"{idx}.tif"
                    )
                    with rst.open(
                        img_path, 'w',
                        driver='GTiff',
                        compress='LZW',
                        width=subset.shape[1],
                        height=subset.shape[0],
                        count=subset.shape[2],
                        transform=subset_trans,
                        dtype='uint8',
                        crs=self.crs
                    ) as out:
                        out.write(np.moveaxis(subset, -1, 0))

                    # Create and save mask
                    with rst.open(img_path, 'r') as subset_raster:
                        caravan_mask = mask.raster_geometry_mask(
                            subset_raster,
                            caravans['geometry'],
                            invert=True,
                            pad=True
                        )[0].astype(np.uint8)

                    mask_path = os.path.join(
                        self.output_dir, split, "labels", "1", f"{idx}.tif"
                    )
                    with rst.open(
                        mask_path, 'w',
                        driver='GTiff',
                        compress='LZW',
                        width=caravan_mask.shape[1],
                        height=caravan_mask.shape[0],
                        count=1,
                        dtype='uint8',
                        crs=self.crs
                    ) as out:
                        out.write(caravan_mask, 1)

                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed {processed} patches...")

            except Exception as e:
                skipped += 1
                continue

        print(f"  Created {processed} patches (skipped {skipped})")
        return processed

    ############################################################
    #  Step 3: Label Masks with Instance IDs
    ############################################################

    def label_masks(self):
        """
        Label binary masks with instance IDs using skimage.measure.label().

        This converts binary masks (0/1) to instance-labeled masks where
        each connected component (individual caravan) gets a unique ID.
        """
        print("=" * 60)
        print("STEP 3: Labeling Masks with Instance IDs")
        print("=" * 60)

        for split in ["train", "val", "test"]:
            mask_dir = os.path.join(self.output_dir, split, "labels", "1")
            mask_files = glob(os.path.join(mask_dir, "*.tif"))

            print(f"  Processing {split} set ({len(mask_files)} masks)...")

            for mask_path in mask_files:
                # Read binary mask
                binary_mask = imread(mask_path)

                # Label connected components
                labeled_mask = label(binary_mask)

                # Save as PNG (ONS format)
                output_path = mask_path.replace('.tif', '.png')
                imsave(output_path, labeled_mask.astype(np.uint16))

                # Optionally remove original TIF mask
                # os.remove(mask_path)

            print(f"    Labeled {len(mask_files)} masks in {split}")

    ############################################################
    #  Step 4: Train Mask R-CNN Model
    ############################################################

    def train(self, epochs=50, layers='heads', weights='coco'):
        """
        Train Mask R-CNN model using ONS methodology.

        Args:
            epochs: Number of training epochs (default: 50)
            layers: Which layers to train - 'heads', 'all', '3+', '4+', '5+'
            weights: Starting weights - 'coco', 'imagenet', or path to .h5 file
        """
        print("=" * 60)
        print("STEP 4: Training Mask R-CNN Model")
        print("=" * 60)

        config = ONSCaravanConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(
            mode="training",
            config=config,
            model_dir=os.path.join(self.output_dir, "logs")
        )

        # Load weights
        if weights.lower() == "coco":
            weights_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
            if not os.path.exists(weights_path):
                print("  Downloading COCO weights...")
                utils.download_trained_weights(weights_path)
            # Exclude final layers (different number of classes)
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"
            ])
        elif weights.lower() == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        else:
            model.load_weights(weights, by_name=True)

        # Load datasets
        print("  Loading training dataset...")
        dataset_train = ONSCaravanDataset()
        dataset_train.load_caravans(self.output_dir, "train")
        dataset_train.prepare()

        print("  Loading validation dataset...")
        dataset_val = ONSCaravanDataset()
        dataset_val.load_caravans(self.output_dir, "val")
        dataset_val.prepare()

        print(f"  Training images: {len(dataset_train.image_ids)}")
        print(f"  Validation images: {len(dataset_val.image_ids)}")

        # Train
        print(f"\n  Training {layers} for {epochs} epochs...")
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs,
            layers=layers
        )

        print("  Training complete!")
        return model

    ############################################################
    #  Step 5: Inference and Evaluation
    ############################################################

    def detect(self, model, image_path):
        """
        Run detection on a single image.

        Args:
            model: Trained Mask R-CNN model
            image_path: Path to image file

        Returns:
            Dictionary with 'rois', 'masks', 'class_ids', 'scores'
        """
        image = imread(image_path)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        results = model.detect([image], verbose=0)[0]
        return results

    def evaluate(self, weights_path=None):
        """
        Evaluate model using ONS accuracy metrics.

        Computes:
        - Pixel-level F1 scores
        - Confusion matrices
        - Average F1 across test set

        Args:
            weights_path: Path to trained weights (or 'last' for most recent)
        """
        print("=" * 60)
        print("STEP 5: Evaluating Model (ONS Metrics)")
        print("=" * 60)

        config = ONSInferenceConfig()
        model = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=os.path.join(self.output_dir, "logs")
        )

        # Load weights
        if weights_path is None or weights_path.lower() == "last":
            weights_path = model.find_last()
        model.load_weights(weights_path, by_name=True)

        # Load test dataset
        test_dir = os.path.join(self.output_dir, "test")
        image_dir = os.path.join(test_dir, "images")
        label_dir = os.path.join(test_dir, "labels", "1")
        results_dir = os.path.join(self.output_dir, "results", "masks")

        image_files = glob(os.path.join(image_dir, "*.tif"))

        f1_scores = []
        confusion_matrices = []

        print(f"  Evaluating on {len(image_files)} test images...")

        for img_path in image_files:
            filename = os.path.basename(img_path)
            label_path = os.path.join(
                label_dir,
                filename.replace('.tif', '.png')
            )

            if not os.path.exists(label_path):
                continue

            # Run detection
            results = self.detect(model, img_path)

            # Combine all instance masks into single mask
            if results['masks'].shape[-1] > 0:
                pred_mask = np.any(results['masks'], axis=-1).astype(np.uint8)
            else:
                pred_mask = np.zeros((224, 224), dtype=np.uint8)

            # Load ground truth
            gt_mask = imread(label_path)
            gt_binary = (gt_mask > 0).astype(np.uint8)

            # Calculate metrics
            pred_flat = pred_mask.ravel()
            gt_flat = gt_binary.ravel()

            f1 = f1_score(gt_flat, pred_flat, zero_division=0)
            cm = confusion_matrix(gt_flat, pred_flat)

            f1_scores.append(f1)
            confusion_matrices.append(cm)

            # Save result mask
            result_path = os.path.join(results_dir, filename.replace('.tif', '.png'))
            imsave(result_path, (pred_mask * 255).astype(np.uint8))

        # Summary statistics
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        min_f1 = np.min(f1_scores)
        max_f1 = np.max(f1_scores)

        print("\n" + "=" * 40)
        print("EVALUATION RESULTS (ONS Metrics)")
        print("=" * 40)
        print(f"  Test images:     {len(f1_scores)}")
        print(f"  Average F1:      {avg_f1:.4f}")
        print(f"  Std Dev F1:      {std_f1:.4f}")
        print(f"  Min F1:          {min_f1:.4f}")
        print(f"  Max F1:          {max_f1:.4f}")
        print("=" * 40)

        return {
            'f1_scores': f1_scores,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'confusion_matrices': confusion_matrices
        }

    def count_caravans(self, image_path, weights_path=None):
        """
        Count caravans in a single image.

        Args:
            image_path: Path to image
            weights_path: Path to model weights

        Returns:
            Number of detected caravans
        """
        config = ONSInferenceConfig()
        model = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=os.path.join(self.output_dir, "logs")
        )

        if weights_path is None or weights_path.lower() == "last":
            weights_path = model.find_last()
        model.load_weights(weights_path, by_name=True)

        results = self.detect(model, image_path)
        return len(results['rois'])


############################################################
#  Command Line Interface
############################################################

def main():
    parser = argparse.ArgumentParser(
        description='ONS Caravan Counting Pipeline'
    )
    parser.add_argument('command',
                        choices=['georeference', 'prepare', 'label',
                                'train', 'evaluate', 'count', 'full'],
                        help='Pipeline command to run')
    parser.add_argument('--imagery', required=False,
                        help='Directory containing satellite imagery')
    parser.add_argument('--caravans', required=False,
                        help='CSV file with caravan polygons')
    parser.add_argument('--tiles', required=False,
                        help='CSV file with tile metadata for georeferencing')
    parser.add_argument('--output', required=False,
                        help='Output directory')
    parser.add_argument('--weights', default='coco',
                        help='Model weights: coco, imagenet, last, or path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size for training data')
    parser.add_argument('--image', required=False,
                        help='Image path for counting')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ONSCaravanPipeline(
        imagery_dir=args.imagery or ".",
        caravan_csv=args.caravans or "",
        output_dir=args.output or "./caravan_data"
    )

    if args.command == 'georeference':
        if not args.tiles:
            print("Error: --tiles required for georeferencing")
            return
        pipeline.georeference_tiles(args.tiles)

    elif args.command == 'prepare':
        pipeline.create_training_data(sample_size=args.sample)

    elif args.command == 'label':
        pipeline.label_masks()

    elif args.command == 'train':
        pipeline.train(epochs=args.epochs, weights=args.weights)

    elif args.command == 'evaluate':
        pipeline.evaluate(weights_path=args.weights)

    elif args.command == 'count':
        if not args.image:
            print("Error: --image required for counting")
            return
        count = pipeline.count_caravans(args.image, args.weights)
        print(f"Detected caravans: {count}")

    elif args.command == 'full':
        # Run full pipeline
        if args.tiles:
            pipeline.georeference_tiles(args.tiles)
        pipeline.create_training_data(sample_size=args.sample)
        pipeline.label_masks()
        pipeline.train(epochs=args.epochs, weights=args.weights)
        pipeline.evaluate()


if __name__ == '__main__':
    main()
