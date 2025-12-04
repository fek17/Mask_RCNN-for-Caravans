"""
Count Caravans from Satellite Images using Mask R-CNN

Processes folders of satellite images, detects caravans using the trained Mask R-CNN
model, overlays them on images, and outputs deduplicated counts to Excel.

Can also use OpenStreetMap data as a fallback if model weights are not provided,
but OSM coverage for individual caravans is limited.

Expected folder structure:
    images/
    ├── Park Name 1/
    │   ├── 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png
    │   └── ...
    └── Park Name 2/
        └── ...

Usage:
    # Using Mask R-CNN model (recommended)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx \\
        --weights /path/to/mask_rcnn_caravan.h5 --save_overlays

    # Using OSM data (fallback - limited coverage)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --use_osm

Requirements:
    pip install pandas openpyxl Pillow numpy scikit-image tensorflow keras
    # For OSM mode: pip install osmnx geopandas shapely rasterio affine

Copyright (c) 2024
"""

import os
import sys
import re
import math
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from io import BytesIO
import hashlib
import uuid

# Add Mask R-CNN to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

# Mask R-CNN imports (optional - only needed if using model)
MRCNN_AVAILABLE = False
try:
    from mrcnn.config import Config
    from mrcnn import model as modellib
    import skimage.io
    MRCNN_AVAILABLE = True
except ImportError:
    pass

# OSM/Geo libraries (optional - only needed if using OSM mode)
OSM_AVAILABLE = False
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import box, Point, Polygon, MultiPolygon
    from shapely.ops import unary_union
    from rasterio.features import rasterize
    from affine import Affine
    OSM_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# MASK R-CNN CONFIGURATION
# ============================================================================

class CaravanConfig(Config):
    """Configuration for caravan detection inference."""
    NAME = "caravan"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + caravan
    DETECTION_MIN_CONFIDENCE = 0.7
    BACKBONE = "resnet101"


# Global model instance (loaded once)
_model = None
_model_weights_path = None


# ============================================================================
# COORDINATE CONVERSION UTILITIES
# ============================================================================

def lonlat_to_merc(lon, lat):
    """Convert lon/lat to Web Mercator meters"""
    R = 6378137.0
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
    return x, y


def merc_to_lonlat(x, y):
    """Convert Web Mercator meters to lon/lat"""
    R = 6378137.0
    lon = math.degrees(x / R)
    lat = math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
    return lon, lat


def get_image_bounds_from_corners(tl_lat, tl_lon, br_lat, br_lon, width, height):
    """
    Calculate image bounds in Web Mercator from corner coordinates.

    Returns:
        tuple: (minx, miny, maxx, maxy, transform)
    """
    tl_x, tl_y = lonlat_to_merc(tl_lon, tl_lat)
    br_x, br_y = lonlat_to_merc(br_lon, br_lat)

    minx = min(tl_x, br_x)
    maxx = max(tl_x, br_x)
    miny = min(tl_y, br_y)
    maxy = max(tl_y, br_y)

    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height

    transform = Affine(pixel_width, 0, minx, 0, -pixel_height, maxy)

    return minx, miny, maxx, maxy, transform


# ============================================================================
# FILENAME PARSING
# ============================================================================

def parse_filename(filename):
    """
    Parse filename to extract coordinates.
    Format: 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png
    """
    pattern = r'(\d+)_z(\d+)_TL_([-]?\d+\.\d+)_([-]?\d+\.\d+)_BR_([-]?\d+\.\d+)_([-]?\d+\.\d+)'
    match = re.search(pattern, filename)

    if not match:
        return None

    tl_lat = float(match.group(3))
    tl_lon = float(match.group(4))
    br_lat = float(match.group(5))
    br_lon = float(match.group(6))

    return {
        'file_id': match.group(1),
        'zoom': int(match.group(2)),
        'tl_lat': tl_lat,
        'tl_lon': tl_lon,
        'br_lat': br_lat,
        'br_lon': br_lon,
        'center_lat': (tl_lat + br_lat) / 2,
        'center_lon': (tl_lon + br_lon) / 2,
        'min_lat': min(tl_lat, br_lat),
        'max_lat': max(tl_lat, br_lat),
        'min_lon': min(tl_lon, br_lon),
        'max_lon': max(tl_lon, br_lon),
    }


# ============================================================================
# MASK R-CNN MODEL FUNCTIONS
# ============================================================================

def load_model(weights_path):
    """Load the Mask R-CNN model with trained weights."""
    global _model, _model_weights_path

    if _model is not None and _model_weights_path == weights_path:
        return _model

    if not MRCNN_AVAILABLE:
        raise ImportError(
            "Mask R-CNN not available. Install with:\n"
            "  pip install tensorflow keras\n"
            "And ensure mrcnn package is in the path."
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"Loading Mask R-CNN model from: {weights_path}")
    config = CaravanConfig()
    _model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)
    _model.load_weights(weights_path, by_name=True)
    _model_weights_path = weights_path
    print("Model loaded successfully!")

    return _model


def detect_caravans_mrcnn(image_array, model):
    """
    Detect caravans in an image using Mask R-CNN.

    Args:
        image_array: RGB image as numpy array (H, W, 3)
        model: Loaded Mask R-CNN model

    Returns:
        tuple: (mask, list of caravan dicts)
    """
    # Run detection
    results = model.detect([image_array], verbose=0)[0]

    masks = results['masks']      # (H, W, N) boolean array
    scores = results['scores']    # (N,) confidence scores
    rois = results['rois']        # (N, 4) bounding boxes [y1, x1, y2, x2]

    height, width = image_array.shape[:2]
    num_detections = masks.shape[2] if len(masks.shape) == 3 else 0

    # Create combined mask
    if num_detections > 0:
        combined_mask = np.any(masks, axis=2).astype(np.uint8)
    else:
        combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Extract individual caravan info
    caravan_list = []
    for i in range(num_detections):
        mask_i = masks[:, :, i]
        y1, x1, y2, x2 = rois[i]
        score = scores[i]

        # Calculate centroid from mask
        ys, xs = np.where(mask_i)
        if len(xs) > 0 and len(ys) > 0:
            centroid_x = np.mean(xs)
            centroid_y = np.mean(ys)
        else:
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2

        # Generate unique ID based on position and mask
        # This helps with deduplication across overlapping images
        pos_hash = hashlib.md5(f"{centroid_x:.1f}_{centroid_y:.1f}_{mask_i.sum()}".encode()).hexdigest()[:12]
        detection_id = f"det_{pos_hash}"

        caravan_list.append({
            'osm_id': detection_id,  # Using same key for compatibility
            'osm_type': 'detection',
            'building_type': 'caravan',
            'confidence': float(score),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'centroid_x': float(centroid_x),
            'centroid_y': float(centroid_y),
            'pixel_count': int(mask_i.sum()),
            'mask': mask_i  # Keep mask for overlay
        })

    return combined_mask, caravan_list


def deduplicate_detections_spatial(all_caravans, overlap_threshold=0.5):
    """
    Deduplicate caravan detections across images using spatial proximity.

    For detections (unlike OSM), we can't rely on IDs, so we use centroid proximity
    and size similarity.
    """
    if not all_caravans:
        return {}

    # Sort by pixel count (larger first - keep the best detection)
    sorted_caravans = sorted(all_caravans.values(), key=lambda x: x.get('pixel_count', 0), reverse=True)

    unique = {}
    for caravan in sorted_caravans:
        is_duplicate = False

        for existing_id, existing in unique.items():
            # Check if centroids are close (within ~20 pixels at the overlapping region)
            dx = abs(caravan.get('centroid_x', 0) - existing.get('centroid_x', 0))
            dy = abs(caravan.get('centroid_y', 0) - existing.get('centroid_y', 0))
            dist = math.sqrt(dx**2 + dy**2)

            # Also check if sizes are similar
            size1 = caravan.get('pixel_count', 0)
            size2 = existing.get('pixel_count', 0)
            size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0

            # Consider duplicate if close and similar size
            if dist < 30 and size_ratio > 0.5:
                is_duplicate = True
                break

        if not is_duplicate:
            unique[caravan['osm_id']] = caravan

    return unique


# ============================================================================
# OSM DATA FETCHING (using osmnx like your parking script)
# ============================================================================

def fetch_osm_caravans(minx, miny, maxx, maxy, width, height, center_lat, center_lon):
    """
    Fetch caravan/mobile home polygons from OSM using osmnx.
    Returns both a rasterized mask and the raw geometries with IDs.

    Args:
        minx, miny, maxx, maxy: Bounds in Web Mercator
        width, height: Image dimensions
        center_lat, center_lon: Center point for querying

    Returns:
        tuple: (mask array, list of caravan dicts with osm_id, geometry, building_type)
    """
    # Calculate distance for query (diagonal / 2 with buffer)
    dist = math.sqrt((maxx - minx)**2 + (maxy - miny)**2) / 2 * 1.5

    all_caravans = []

    # Query 1: Individual caravan buildings
    try:
        caravans = ox.features_from_point(
            (center_lat, center_lon),
            tags={
                'building': ['static_caravan', 'caravan', 'mobile_home', 'chalet', 'hut']
            },
            dist=dist
        )
        if len(caravans) > 0:
            caravans = caravans.to_crs(3857)
            caravans = caravans[caravans.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            caravans['feature_type'] = 'building'
            all_caravans.append(caravans)
            print(f"    Found {len(caravans)} caravan buildings")
    except Exception as e:
        if "No matching features" not in str(e):
            print(f"    Warning querying buildings: {e}")

    # Query 1b: Features tagged with caravans=yes (camp pitches that allow caravans)
    try:
        caravan_pitches = ox.features_from_point(
            (center_lat, center_lon),
            tags={
                'caravans': True  # Matches caravans=yes, caravans=no, etc.
            },
            dist=dist
        )
        if len(caravan_pitches) > 0:
            # Filter to only caravans=yes
            if 'caravans' in caravan_pitches.columns:
                caravan_pitches = caravan_pitches[caravan_pitches['caravans'] == 'yes']
            caravan_pitches = caravan_pitches.to_crs(3857)
            caravan_pitches = caravan_pitches[caravan_pitches.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            caravan_pitches['feature_type'] = 'caravan_pitch'
            if len(caravan_pitches) > 0:
                all_caravans.append(caravan_pitches)
                print(f"    Found {len(caravan_pitches)} pitches with caravans=yes")
    except Exception as e:
        if "No matching features" not in str(e):
            print(f"    Warning querying caravans=yes: {e}")

    # Query 2: Caravan site boundaries (tourism=caravan_site)
    try:
        sites = ox.features_from_point(
            (center_lat, center_lon),
            tags={
                'tourism': ['caravan_site', 'camp_site']
            },
            dist=dist
        )
        if len(sites) > 0:
            sites = sites.to_crs(3857)
            sites = sites[sites.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            sites['feature_type'] = 'site_boundary'
            # Don't add site boundaries to caravan count - just note them
            if len(sites) > 0:
                print(f"    Found {len(sites)} caravan SITE boundaries (not individual units)")
    except Exception as e:
        if "No matching features" not in str(e):
            print(f"    Warning querying sites: {e}")

    # Query 3: Leisure pitches (individual camping/caravan pitches)
    try:
        pitches = ox.features_from_point(
            (center_lat, center_lon),
            tags={
                'tourism': 'camp_pitch',
                'leisure': 'pitch'
            },
            dist=dist
        )
        if len(pitches) > 0:
            pitches = pitches.to_crs(3857)
            pitches = pitches[pitches.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            pitches['feature_type'] = 'pitch'
            all_caravans.append(pitches)
            print(f"    Found {len(pitches)} camping/caravan pitches")
    except Exception as e:
        if "No matching features" not in str(e):
            print(f"    Warning querying pitches: {e}")

    # Combine all results
    if not all_caravans:
        print("    No individual caravans mapped in OSM for this area")
        print("    TIP: Use Mask R-CNN model to detect caravans from imagery")
        return np.zeros((height, width), dtype=np.uint8), []

    caravans = gpd.GeoDataFrame(pd.concat(all_caravans, ignore_index=True), crs=3857)
    print(f"    Total: {len(caravans)} caravan features")

    if len(caravans) == 0:
        return np.zeros((height, width), dtype=np.uint8), []

    # Create transform for rasterization
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    transform = Affine(pixel_width, 0, minx, 0, -pixel_height, maxy)

    # Rasterize for mask
    shapes = [(geom, 1) for geom in caravans.geometry]
    caravan_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Extract caravan info for deduplication
    caravan_list = []
    for idx, row in caravans.iterrows():
        # Get OSM ID from index or column
        if 'osmid' in caravans.columns:
            osm_id = row.get('osmid', idx)
        elif isinstance(idx, tuple):
            osm_id = idx[1] if len(idx) > 1 else idx[0]
        else:
            osm_id = idx

        building_type = row.get('building', row.get('tourism', row.get('leisure', 'unknown')))
        feature_type = row.get('feature_type', 'unknown')
        geom = row.geometry

        # Get centroid
        centroid = geom.centroid
        centroid_lon, centroid_lat = merc_to_lonlat(centroid.x, centroid.y)

        # Convert geometry to lon/lat for storage
        if geom.type == 'Polygon':
            coords = [(merc_to_lonlat(x, y)) for x, y in geom.exterior.coords]
        else:
            coords = []

        caravan_list.append({
            'osm_id': osm_id,
            'osm_type': feature_type,
            'building_type': building_type,
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'geometry_merc': geom,
            'coords_lonlat': coords
        })

    return caravan_mask, caravan_list


# ============================================================================
# IMAGE ANNOTATION
# ============================================================================

def create_annotated_image(original_image, caravan_mask,
                           alpha=0.5, color=(255, 0, 0)):
    """
    Create annotated image with caravan polygons highlighted.

    Args:
        original_image: Original RGB image as numpy array
        caravan_mask: Binary mask of caravans
        alpha: Transparency for overlay
        color: RGB color for caravan highlighting
    """
    if len(original_image.shape) == 2:
        original_image = np.stack([original_image] * 3, axis=-1)
    elif original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]

    annotated = original_image.copy().astype(np.float32)

    # Overlay caravans in red
    caravan_pixels = (caravan_mask == 1)
    for c in range(3):
        annotated[:, :, c] = np.where(
            caravan_pixels,
            annotated[:, :, c] * (1 - alpha) + color[c] * alpha,
            annotated[:, :, c]
        )

    return annotated.astype(np.uint8)


def draw_caravan_count(image, count, position=(10, 10)):
    """Draw caravan count on image."""
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()

    text = f"Caravans: {count}"

    # Draw background rectangle
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0, 0, 0, 200))

    # Draw text
    draw.text(position, text, fill=(255, 255, 255), font=font)

    return np.array(img_pil)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_single_image(image_path, output_dir=None, save_overlay=False,
                         model=None, use_osm=False):
    """
    Process a single satellite image.

    Args:
        image_path: Path to the satellite image
        output_dir: Directory to save overlays
        save_overlay: Whether to save annotated images
        model: Loaded Mask R-CNN model (if using model detection)
        use_osm: Whether to use OSM data instead of model

    Returns:
        tuple: (file_info dict, list of caravans found)
    """
    filename = os.path.basename(image_path)

    # Parse filename
    file_info = parse_filename(filename)
    if file_info is None:
        print(f"  Error: Could not parse filename")
        return None, []

    # Load image
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    if use_osm:
        # OSM mode - fetch from OpenStreetMap
        if not OSM_AVAILABLE:
            print("  Error: OSM libraries not available")
            return file_info, []

        # Get image bounds
        minx, miny, maxx, maxy, transform = get_image_bounds_from_corners(
            file_info['tl_lat'], file_info['tl_lon'],
            file_info['br_lat'], file_info['br_lon'],
            width, height
        )

        # Fetch caravans from OSM
        caravan_mask, caravan_list = fetch_osm_caravans(
            minx, miny, maxx, maxy, width, height,
            file_info['center_lat'], file_info['center_lon']
        )
    else:
        # Model mode - use Mask R-CNN
        if model is None:
            print("  Error: No model loaded for detection")
            return file_info, []

        caravan_mask, caravan_list = detect_caravans_mrcnn(img_array, model)

    # Save annotated image if requested
    if save_overlay and output_dir and len(caravan_list) > 0:
        annotated = create_annotated_image(img_array, caravan_mask)
        annotated = draw_caravan_count(annotated, len(caravan_list))

        overlay_filename = f"overlay_{filename}"
        overlay_path = os.path.join(output_dir, overlay_filename)
        Image.fromarray(annotated).save(overlay_path, quality=95)

    return file_info, caravan_list


def scan_image_folders(root_dir):
    """Scan directory for satellite images organized by park name."""
    parks = defaultdict(list)

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue

            file_info = parse_filename(filename)
            if file_info is None:
                continue

            rel_path = os.path.relpath(root, root_dir)
            park_name = rel_path if rel_path != '.' else os.path.basename(root)

            parks[park_name].append({
                'filename': filename,
                'filepath': os.path.join(root, filename),
                'file_info': file_info
            })

    return dict(parks)


def process_park(park_name, images, output_dir=None, save_overlays=False,
                 model=None, use_osm=False):
    """
    Process all images for a single park, deduplicating caravans.

    Args:
        park_name: Name of the caravan park
        images: List of image info dicts
        output_dir: Directory for output files
        save_overlays: Whether to save annotated images
        model: Loaded Mask R-CNN model (if using model detection)
        use_osm: Whether to use OSM data instead of model
    """
    all_caravans = {}  # id -> caravan info (for deduplication)

    print(f"\n  Processing {len(images)} images...")

    # Create overlay directory for this park
    park_overlay_dir = None
    if save_overlays and output_dir:
        park_overlay_dir = os.path.join(output_dir, 'overlays', park_name)
        os.makedirs(park_overlay_dir, exist_ok=True)

    for i, img_info in enumerate(images):
        filename = img_info['filename']
        filepath = img_info['filepath']

        print(f"    [{i+1}/{len(images)}] {filename[:50]}...", end=' ')

        try:
            file_info, caravan_list = process_single_image(
                filepath,
                output_dir=park_overlay_dir,
                save_overlay=save_overlays,
                model=model,
                use_osm=use_osm
            )

            # For OSM mode, deduplicate by OSM ID
            # For model mode, we'll do spatial deduplication after
            new_count = 0
            for caravan in caravan_list:
                caravan_id = caravan['osm_id']
                if caravan_id not in all_caravans:
                    all_caravans[caravan_id] = caravan
                    new_count += 1

            print(f"found {len(caravan_list)} ({new_count} new)")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # For model detections, do additional spatial deduplication
    # (since hash-based IDs might differ for same caravan in different images)
    if not use_osm and len(all_caravans) > 0:
        print(f"  Performing spatial deduplication...")
        all_caravans = deduplicate_detections_spatial(all_caravans)
        print(f"  After deduplication: {len(all_caravans)} unique caravans")

    # Count by type
    type_counts = defaultdict(int)
    for caravan in all_caravans.values():
        type_counts[caravan['building_type']] += 1

    return {
        'park_name': park_name,
        'total_images': len(images),
        'total_caravans_deduplicated': len(all_caravans),
        'static_caravans': type_counts.get('static_caravan', 0),
        'mobile_homes': type_counts.get('mobile_home', 0),
        'other_caravans': type_counts.get('caravan', 0),
        'all_caravans': all_caravans
    }


def process_all_parks(input_dir, output_excel, save_overlays=False,
                      weights_path=None, use_osm=False):
    """
    Process all caravan parks found in the input directory.

    Args:
        input_dir: Root directory containing park folders with satellite images
        output_excel: Path to output Excel file
        save_overlays: Whether to save annotated images
        weights_path: Path to Mask R-CNN model weights (if using model detection)
        use_osm: Whether to use OSM data instead of model
    """
    print(f"\nScanning {input_dir} for satellite images...")
    parks = scan_image_folders(input_dir)

    if not parks:
        print("No valid satellite images found!")
        print("Expected filename format: 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png")
        return pd.DataFrame()

    print(f"Found {len(parks)} parks with satellite images:")
    for park_name, images in parks.items():
        print(f"  - {park_name}: {len(images)} images")

    # Load model if using model detection
    model = None
    if not use_osm:
        if weights_path is None:
            print("\nERROR: Model weights required for detection.")
            print("Please provide --weights /path/to/mask_rcnn_caravan.h5")
            print("Or use --use_osm for OSM data (limited coverage)")
            return pd.DataFrame()

        model = load_model(weights_path)

    output_dir = os.path.dirname(output_excel) if output_excel else '.'

    # Process each park
    results = []
    for park_name, images in parks.items():
        print(f"\n{'='*60}")
        print(f"Park: {park_name}")
        print(f"{'='*60}")

        park_result = process_park(
            park_name, images, output_dir,
            save_overlays=save_overlays,
            model=model,
            use_osm=use_osm
        )

        results.append({
            'park_name': park_result['park_name'],
            'total_images': park_result['total_images'],
            'caravan_count': park_result['total_caravans_deduplicated'],
            'static_caravans': park_result['static_caravans'],
            'mobile_homes': park_result['mobile_homes'],
            'other_caravans': park_result['other_caravans']
        })

        print(f"\n  TOTAL (deduplicated): {park_result['total_caravans_deduplicated']} caravans")

    # Create DataFrame and save
    df = pd.DataFrame(results)

    if output_excel:
        df.to_excel(output_excel, index=False)
        print(f"\n\nResults saved to: {output_excel}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    mode_str = "OSM data" if use_osm else "Mask R-CNN model"
    print(f"Detection mode: {mode_str}")
    print(f"Total parks processed: {len(df)}")
    print(f"Total caravans found: {df['caravan_count'].sum()}")
    print(f"  - Static caravans: {df['static_caravans'].sum()}")
    print(f"  - Mobile homes: {df['mobile_homes'].sum()}")
    print(f"  - Other caravans: {df['other_caravans'].sum()}")
    if len(df) > 0:
        print(f"Average caravans per park: {df['caravan_count'].mean():.1f}")
        print(f"Max caravans at single park: {df['caravan_count'].max()}")
    print(f"{'='*60}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Count caravans in satellite images using Mask R-CNN or OSM data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using Mask R-CNN model (recommended - requires trained weights)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx \\
        --weights ./logs/caravan20xx/mask_rcnn_caravan_xxxx.h5 --save_overlays

    # Using OSM data (fallback - limited coverage for individual caravans)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --use_osm

Detection Modes:
    --weights PATH    Use Mask R-CNN model for detection (recommended)
                      The model should be trained on caravan imagery.
                      See caravan.py for training instructions.

    --use_osm         Use OpenStreetMap data instead of model detection.
                      WARNING: OSM has limited coverage for individual caravans.
                      Most sites only have the site boundary mapped, not
                      individual caravan units.

Requirements:
    For model mode: pip install tensorflow keras scikit-image
    For OSM mode:   pip install osmnx geopandas shapely rasterio affine
    Both modes:     pip install pandas openpyxl Pillow numpy
        """
    )

    parser.add_argument('--input_dir', required=True,
                        help='Root directory containing park folders with satellite images')
    parser.add_argument('--output', required=True,
                        help='Path to output Excel file')
    parser.add_argument('--weights', default=None,
                        help='Path to Mask R-CNN model weights (.h5 file)')
    parser.add_argument('--use_osm', action='store_true',
                        help='Use OSM data instead of model (limited coverage)')
    parser.add_argument('--save_overlays', action='store_true',
                        help='Save images with detection overlays')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Detection confidence threshold (default: 0.7)')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Validate mode
    if not args.use_osm and args.weights is None:
        print("ERROR: You must specify either --weights or --use_osm")
        print("\nOptions:")
        print("  1. Use Mask R-CNN model (recommended):")
        print("     --weights /path/to/mask_rcnn_caravan.h5")
        print("")
        print("  2. Use OSM data (limited coverage):")
        print("     --use_osm")
        print("")
        print("To train a model, see: python caravan.py train --help")
        sys.exit(1)

    if args.weights and not os.path.exists(args.weights):
        print(f"Error: Model weights not found: {args.weights}")
        sys.exit(1)

    # Update config confidence if specified
    if args.confidence != 0.7:
        CaravanConfig.DETECTION_MIN_CONFIDENCE = args.confidence

    try:
        df = process_all_parks(
            input_dir=args.input_dir,
            output_excel=args.output,
            save_overlays=args.save_overlays,
            weights_path=args.weights,
            use_osm=args.use_osm
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
