"""
ONS Caravan Detection Script

This script runs Mask R-CNN inference to detect and count caravans in satellite imagery.
It replicates the exact ONS methodology for caravan detection.

Usage:
    # Detect caravans in a single image
    python detect_caravans.py detect --image=/path/to/image.tif --weights=/path/to/weights.h5

    # Batch detect caravans in a folder
    python detect_caravans.py batch --folder=/path/to/images --weights=/path/to/weights.h5

    # Export results with visualizations
    python detect_caravans.py detect --image=/path/to/image.tif --weights=last --visualize --output=/path/to/output

Copyright (c) 2024
Based on ONS methodology
"""

import os
import sys
import numpy as np
import argparse
from glob import glob
from skimage.io import imread, imsave
import csv
from datetime import datetime

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Import matplotlib with non-interactive backend for saving
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


############################################################
#  Configuration
############################################################

class CaravanInferenceConfig(Config):
    """Configuration for inference - matches ONS training config."""
    NAME = "caravan"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + Caravan
    DETECTION_MIN_CONFIDENCE = 0.9
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    BACKBONE = "resnet101"


############################################################
#  Detection Functions
############################################################

def load_model(weights_path, logs_dir=None):
    """
    Load Mask R-CNN model with trained weights.

    Args:
        weights_path: Path to weights file, or 'last' for most recent
        logs_dir: Directory containing training logs

    Returns:
        Loaded model ready for inference
    """
    config = CaravanInferenceConfig()

    if logs_dir is None:
        logs_dir = os.path.join(ROOT_DIR, "logs")

    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=logs_dir
    )

    # Find weights
    if weights_path.lower() == "last":
        weights_path = model.find_last()
    elif weights_path.lower() == "coco":
        weights_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path, by_name=True)

    return model


def detect_caravans(model, image_path):
    """
    Detect caravans in a single image.

    Args:
        model: Loaded Mask R-CNN model
        image_path: Path to image file

    Returns:
        results: Dictionary containing:
            - rois: Bounding boxes [N, (y1, x1, y2, x2)]
            - masks: Instance masks [H, W, N]
            - class_ids: Class IDs for each detection
            - scores: Confidence scores [N]
        image: The loaded image array
    """
    # Load image
    image = imread(image_path)

    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # Handle RGBA images
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Run detection
    results = model.detect([image], verbose=0)[0]

    return results, image


def get_caravan_count(results):
    """Get the number of detected caravans."""
    return len(results['rois'])


def get_combined_mask(results, image_shape):
    """Combine all instance masks into a single binary mask."""
    if results['masks'].shape[-1] > 0:
        return np.any(results['masks'], axis=-1).astype(np.uint8) * 255
    else:
        return np.zeros(image_shape[:2], dtype=np.uint8)


def get_caravan_areas(results, pixel_size=0.25):
    """
    Calculate area of each detected caravan.

    Args:
        results: Detection results
        pixel_size: Size of each pixel in meters (default 0.25m for OS imagery)

    Returns:
        List of areas in square meters
    """
    areas = []
    pixel_area = pixel_size * pixel_size

    for i in range(results['masks'].shape[-1]):
        mask = results['masks'][:, :, i]
        pixel_count = np.sum(mask)
        area_m2 = pixel_count * pixel_area
        areas.append(area_m2)

    return areas


def get_caravan_centroids(results):
    """
    Get centroid coordinates for each detected caravan.

    Returns:
        List of (x, y) tuples in image coordinates
    """
    centroids = []

    for i in range(results['masks'].shape[-1]):
        mask = results['masks'][:, :, i]
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            centroids.append((centroid_x, centroid_y))

    return centroids


############################################################
#  Visualization Functions
############################################################

def visualize_detections(image, results, output_path, show_masks=True,
                         show_boxes=True, show_scores=True):
    """
    Create visualization of detected caravans.

    Args:
        image: Input image array
        results: Detection results
        output_path: Path to save visualization
        show_masks: Whether to show instance masks
        show_boxes: Whether to show bounding boxes
        show_scores: Whether to show confidence scores
    """
    fig, ax = plt.subplots(1, figsize=(12, 12))

    # Display image
    ax.imshow(image)

    # Draw each detection
    n_instances = len(results['rois'])
    colors = plt.cm.hsv(np.linspace(0, 1, max(n_instances, 1)))

    for i in range(n_instances):
        color = colors[i]

        # Bounding box
        if show_boxes:
            y1, x1, y2, x2 = results['rois'][i]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color,
                facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)

        # Confidence score
        if show_scores:
            score = results['scores'][i]
            y1, x1, y2, x2 = results['rois'][i]
            ax.text(
                x1, y1 - 5, f'{score:.2f}',
                fontsize=10, color='white',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
            )

        # Mask overlay
        if show_masks and results['masks'].shape[-1] > 0:
            mask = results['masks'][:, :, i]
            masked = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked, alpha=0.4, cmap=plt.cm.colors.ListedColormap([color]))

    ax.set_title(f'Detected Caravans: {n_instances}', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")


def save_mask_output(results, output_path, image_shape):
    """Save combined mask as image file."""
    combined_mask = get_combined_mask(results, image_shape)
    imsave(output_path, combined_mask)
    print(f"Mask saved to: {output_path}")


############################################################
#  Batch Processing
############################################################

def batch_detect(model, folder_path, output_dir=None, visualize=False):
    """
    Run detection on all images in a folder.

    Args:
        model: Loaded Mask R-CNN model
        folder_path: Path to folder containing images
        output_dir: Directory for output files (optional)
        visualize: Whether to create visualizations

    Returns:
        Dictionary mapping filenames to caravan counts
    """
    # Find all image files
    extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))
        image_files.extend(glob(os.path.join(folder_path, ext.upper())))

    if not image_files:
        print(f"No images found in {folder_path}")
        return {}

    print(f"Found {len(image_files)} images to process")

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        if visualize:
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    # Process each image
    results_dict = {}
    total_caravans = 0

    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        print(f"Processing [{i+1}/{len(image_files)}]: {filename}")

        try:
            results, image = detect_caravans(model, image_path)
            count = get_caravan_count(results)
            results_dict[filename] = {
                'count': count,
                'scores': results['scores'].tolist(),
                'areas': get_caravan_areas(results)
            }
            total_caravans += count

            # Save outputs
            if output_dir:
                # Save mask
                mask_path = os.path.join(output_dir, "masks", filename.rsplit('.', 1)[0] + "_mask.png")
                save_mask_output(results, mask_path, image.shape)

                # Save visualization
                if visualize:
                    vis_path = os.path.join(output_dir, "visualizations", filename.rsplit('.', 1)[0] + "_detected.png")
                    visualize_detections(image, results, vis_path)

            print(f"  -> Detected {count} caravans")

        except Exception as e:
            print(f"  -> Error: {e}")
            results_dict[filename] = {'count': 0, 'error': str(e)}

    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Images processed: {len(image_files)}")
    print(f"Total caravans detected: {total_caravans}")
    print(f"Average per image: {total_caravans/len(image_files):.1f}")

    # Save summary to CSV
    if output_dir:
        csv_path = os.path.join(output_dir, "detection_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'caravan_count', 'confidence_scores', 'total_area_m2'])
            for filename, data in results_dict.items():
                if 'error' not in data:
                    writer.writerow([
                        filename,
                        data['count'],
                        ';'.join(f"{s:.3f}" for s in data['scores']),
                        sum(data['areas'])
                    ])
        print(f"Results saved to: {csv_path}")

    return results_dict


############################################################
#  Main Entry Point
############################################################

def main():
    parser = argparse.ArgumentParser(
        description='ONS Caravan Detection using Mask R-CNN'
    )
    parser.add_argument('command',
                        choices=['detect', 'batch', 'count'],
                        help='Command to run')
    parser.add_argument('--image', required=False,
                        help='Path to single image for detection')
    parser.add_argument('--folder', required=False,
                        help='Path to folder for batch processing')
    parser.add_argument('--weights', required=True,
                        help='Path to model weights (.h5) or "last"')
    parser.add_argument('--logs', default=None,
                        help='Directory containing training logs')
    parser.add_argument('--output', required=False,
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization images')
    parser.add_argument('--save-mask', action='store_true',
                        help='Save detection mask')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = load_model(args.weights, args.logs)

    if args.command == 'detect':
        if not args.image:
            print("Error: --image required for detect command")
            return

        results, image = detect_caravans(model, args.image)
        count = get_caravan_count(results)

        print(f"\n{'='*50}")
        print(f"DETECTION RESULTS")
        print(f"{'='*50}")
        print(f"Image: {args.image}")
        print(f"Caravans detected: {count}")

        if count > 0:
            print(f"\nConfidence scores:")
            for i, score in enumerate(results['scores']):
                print(f"  Caravan {i+1}: {score:.3f}")

            areas = get_caravan_areas(results)
            print(f"\nEstimated areas (m²):")
            for i, area in enumerate(areas):
                print(f"  Caravan {i+1}: {area:.1f} m²")

        # Save outputs
        if args.output:
            os.makedirs(args.output, exist_ok=True)

        if args.visualize:
            output_path = args.output or os.path.dirname(args.image)
            vis_path = os.path.join(
                output_path,
                os.path.basename(args.image).rsplit('.', 1)[0] + "_detected.png"
            )
            visualize_detections(image, results, vis_path)

        if args.save_mask:
            output_path = args.output or os.path.dirname(args.image)
            mask_path = os.path.join(
                output_path,
                os.path.basename(args.image).rsplit('.', 1)[0] + "_mask.png"
            )
            save_mask_output(results, mask_path, image.shape)

    elif args.command == 'batch':
        if not args.folder:
            print("Error: --folder required for batch command")
            return

        batch_detect(
            model,
            args.folder,
            output_dir=args.output,
            visualize=args.visualize
        )

    elif args.command == 'count':
        if not args.image:
            print("Error: --image required for count command")
            return

        results, _ = detect_caravans(model, args.image)
        count = get_caravan_count(results)
        print(count)  # Just output the number for scripting


if __name__ == '__main__':
    main()
