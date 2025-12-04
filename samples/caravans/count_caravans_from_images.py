"""
Count Caravans from Satellite Images with OSM Overlay

Processes folders of satellite images, fetches caravan polygons from OpenStreetMap
using osmnx, overlays them on images, and outputs deduplicated counts to Excel.

Similar approach to the parking lot detection script - uses osmnx for OSM data.

Expected folder structure:
    images/
    ├── Park Name 1/
    │   ├── 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png
    │   └── ...
    └── Park Name 2/
        └── ...

Usage:
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --save_overlays

Requirements:
    pip install pandas openpyxl osmnx geopandas shapely Pillow rasterio affine

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

# OSM/Geo libraries
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterio.features import rasterize
from affine import Affine


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
    print("  Fetching OSM caravan data...")

    # Calculate distance for query (diagonal / 2 with buffer)
    dist = math.sqrt((maxx - minx)**2 + (maxy - miny)**2) / 2 * 1.5

    try:
        # Query for caravan-related buildings
        caravans = ox.features_from_point(
            (center_lat, center_lon),
            tags={
                'building': ['static_caravan', 'caravan', 'mobile_home']
            },
            dist=dist
        )

        if len(caravans) == 0:
            print("    No caravans found in area")
            return np.zeros((height, width), dtype=np.uint8), []

        # Convert to Web Mercator
        caravans = caravans.to_crs(3857)

        # Filter to polygons only
        caravans = caravans[caravans.geometry.type.isin(['Polygon', 'MultiPolygon'])]

        print(f"    Found {len(caravans)} caravan polygons")

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
            # Get OSM ID from index (osmnx uses osmid as index)
            if isinstance(idx, tuple):
                osm_type, osm_id = idx[0], idx[1]
            else:
                osm_type, osm_id = 'way', idx

            building_type = row.get('building', 'unknown')
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
                'osm_type': osm_type,
                'building_type': building_type,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'geometry_merc': geom,
                'coords_lonlat': coords
            })

        return caravan_mask, caravan_list

    except Exception as e:
        print(f"    Warning: Could not fetch caravans: {e}")
        return np.zeros((height, width), dtype=np.uint8), []


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

def process_single_image(image_path, output_dir=None, save_overlay=False):
    """
    Process a single satellite image.

    Returns:
        dict with image info and list of caravans found
    """
    filename = os.path.basename(image_path)
    print(f"\nProcessing: {filename}")

    # Parse filename
    file_info = parse_filename(filename)
    if file_info is None:
        print(f"  Error: Could not parse filename")
        return None, []

    # Load image
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Get image bounds
    minx, miny, maxx, maxy, transform = get_image_bounds_from_corners(
        file_info['tl_lat'], file_info['tl_lon'],
        file_info['br_lat'], file_info['br_lon'],
        width, height
    )

    # Fetch caravans
    caravan_mask, caravan_list = fetch_osm_caravans(
        minx, miny, maxx, maxy, width, height,
        file_info['center_lat'], file_info['center_lon']
    )

    print(f"  Found {len(caravan_list)} caravans in this image")

    # Save annotated image if requested
    if save_overlay and output_dir and len(caravan_list) > 0:
        annotated = create_annotated_image(img_array, caravan_mask)
        annotated = draw_caravan_count(annotated, len(caravan_list))

        overlay_filename = f"overlay_{filename}"
        overlay_path = os.path.join(output_dir, overlay_filename)
        Image.fromarray(annotated).save(overlay_path, quality=95)
        print(f"  Saved overlay: {overlay_filename}")

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


def process_park(park_name, images, output_dir=None, save_overlays=False):
    """
    Process all images for a single park, deduplicating caravans by OSM ID.
    """
    all_caravans = {}  # osm_id -> caravan info (for deduplication)

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
                save_overlay=save_overlays
            )

            # Deduplicate by OSM ID
            new_count = 0
            for caravan in caravan_list:
                osm_id = caravan['osm_id']
                if osm_id not in all_caravans:
                    all_caravans[osm_id] = caravan
                    new_count += 1

            print(f"found {len(caravan_list)} ({new_count} new)")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

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


def process_all_parks(input_dir, output_excel, save_overlays=False):
    """Process all caravan parks found in the input directory."""
    print(f"\nScanning {input_dir} for satellite images...")
    parks = scan_image_folders(input_dir)

    if not parks:
        print("No valid satellite images found!")
        print("Expected filename format: 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png")
        return pd.DataFrame()

    print(f"Found {len(parks)} parks with satellite images:")
    for park_name, images in parks.items():
        print(f"  - {park_name}: {len(images)} images")

    output_dir = os.path.dirname(output_excel) if output_excel else '.'

    # Process each park
    results = []
    for park_name, images in parks.items():
        print(f"\n{'='*60}")
        print(f"Park: {park_name}")
        print(f"{'='*60}")

        park_result = process_park(
            park_name, images, output_dir,
            save_overlays=save_overlays
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
    print(f"Total parks processed: {len(df)}")
    print(f"Total caravans found: {df['caravan_count'].sum()}")
    print(f"  - Static caravans: {df['static_caravans'].sum()}")
    print(f"  - Mobile homes: {df['mobile_homes'].sum()}")
    print(f"  - Other caravans: {df['other_caravans'].sum()}")
    print(f"Average caravans per park: {df['caravan_count'].mean():.1f}")
    print(f"Max caravans at single park: {df['caravan_count'].max()}")
    print(f"{'='*60}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Count caravan polygons from OSM and overlay on satellite images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx

    # With overlay images
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --save_overlays

Requirements:
    pip install pandas openpyxl osmnx geopandas shapely Pillow rasterio affine
        """
    )

    parser.add_argument('--input_dir', required=True,
                        help='Root directory containing park folders with satellite images')
    parser.add_argument('--output', required=True,
                        help='Path to output Excel file')
    parser.add_argument('--save_overlays', action='store_true',
                        help='Save images with polygon overlays')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        df = process_all_parks(
            input_dir=args.input_dir,
            output_excel=args.output,
            save_overlays=args.save_overlays
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
