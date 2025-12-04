"""
Count Caravans from Satellite Images with OSM Overlay

This script processes satellite image screenshots organized in folders (by caravan park name),
queries OpenStreetMap for caravan polygons within each image's bounding box, overlays
the polygons on the images, and outputs deduplicated counts to Excel.

Expected folder structure:
    images/
    ├── Park Name 1/
    │   ├── 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png
    │   ├── 0002_z18_TL_50.020742_-5.091075_BR_50.013683_-5.080089_recent_2025-03-27.png
    │   └── ...
    ├── Park Name 2/
    │   └── ...
    └── ...

Filename format:
    {id}_z{zoom}_TL_{lat1}_{lon1}_BR_{lat2}_{lon2}_recent_{date}.{ext}
    - TL = Top Left coordinate (lat, lon)
    - BR = Bottom Right coordinate (lat, lon)

Usage:
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx

    # With overlay images saved
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --save_overlays

Requirements:
    pip install pandas openpyxl requests Pillow numpy

Copyright (c) 2024
"""

import os
import sys
import re
import argparse
import time
import hashlib
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@dataclass
class BoundingBox:
    """Represents a geographic bounding box."""
    min_lat: float  # South
    max_lat: float  # North
    min_lon: float  # West
    max_lon: float  # East

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is within this bounding box."""
        return (self.min_lat <= lat <= self.max_lat and
                self.min_lon <= lon <= self.max_lon)

    def overlaps(self, other: 'BoundingBox') -> bool:
        """Check if this bbox overlaps with another."""
        return not (self.max_lat < other.min_lat or
                    self.min_lat > other.max_lat or
                    self.max_lon < other.min_lon or
                    self.min_lon > other.max_lon)


@dataclass
class CaravanPolygon:
    """Represents a single caravan polygon from OSM."""
    osm_id: int
    osm_type: str  # 'node' or 'way'
    center_lat: float
    center_lon: float
    building_type: str
    nodes: List[Tuple[float, float]]  # List of (lat, lon) for ways

    def unique_id(self) -> str:
        """Generate unique ID for deduplication."""
        return f"{self.osm_type}_{self.osm_id}"


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse satellite image filename to extract coordinates.

    Expected format: 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png

    Returns dict with: id, zoom, tl_lat, tl_lon, br_lat, br_lon, date
    """
    # Pattern to match the filename format
    pattern = r'(\d+)_z(\d+)_TL_([-\d.]+)_([-\d.]+)_BR_([-\d.]+)_([-\d.]+)_recent_([\d-]+)'

    match = re.search(pattern, filename)
    if not match:
        return None

    return {
        'id': match.group(1),
        'zoom': int(match.group(2)),
        'tl_lat': float(match.group(3)),
        'tl_lon': float(match.group(4)),
        'br_lat': float(match.group(5)),
        'br_lon': float(match.group(6)),
        'date': match.group(7)
    }


def get_bbox_from_parsed(parsed: Dict) -> BoundingBox:
    """Create BoundingBox from parsed filename data."""
    # TL is top-left (north-west), BR is bottom-right (south-east)
    # So TL has higher lat (north), BR has lower lat (south)
    return BoundingBox(
        min_lat=min(parsed['tl_lat'], parsed['br_lat']),
        max_lat=max(parsed['tl_lat'], parsed['br_lat']),
        min_lon=min(parsed['tl_lon'], parsed['br_lon']),
        max_lon=max(parsed['tl_lon'], parsed['br_lon'])
    )


def query_overpass(query: str, max_retries: int = 3) -> Optional[Dict]:
    """Execute an Overpass API query with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={'data': query},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"    Request failed, retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return None
    return None


def get_caravans_in_bbox(bbox: BoundingBox) -> List[CaravanPolygon]:
    """
    Query OSM for caravan polygons within a bounding box.

    Returns list of CaravanPolygon objects with full geometry.
    """
    bbox_str = f"{bbox.min_lat},{bbox.min_lon},{bbox.max_lat},{bbox.max_lon}"

    query = f"""
    [out:json][timeout:30];
    (
      way["building"="static_caravan"]({bbox_str});
      way["building"="caravan"]({bbox_str});
      way["building"="mobile_home"]({bbox_str});
      node["building"="static_caravan"]({bbox_str});
      node["building"="caravan"]({bbox_str});
      node["building"="mobile_home"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """

    data = query_overpass(query)
    if data is None:
        return []

    elements = data.get('elements', [])

    # Build node lookup for way geometries
    node_coords = {}
    for elem in elements:
        if elem['type'] == 'node':
            node_coords[elem['id']] = (elem['lat'], elem['lon'])

    caravans = []
    for elem in elements:
        tags = elem.get('tags', {})
        building_type = tags.get('building', '')

        if building_type not in ['static_caravan', 'caravan', 'mobile_home']:
            continue

        if elem['type'] == 'way':
            # Get node coordinates for this way
            nodes = []
            for node_id in elem.get('nodes', []):
                if node_id in node_coords:
                    nodes.append(node_coords[node_id])

            if nodes:
                # Calculate center
                center_lat = sum(n[0] for n in nodes) / len(nodes)
                center_lon = sum(n[1] for n in nodes) / len(nodes)

                caravans.append(CaravanPolygon(
                    osm_id=elem['id'],
                    osm_type='way',
                    center_lat=center_lat,
                    center_lon=center_lon,
                    building_type=building_type,
                    nodes=nodes
                ))

        elif elem['type'] == 'node' and 'lat' in elem:
            caravans.append(CaravanPolygon(
                osm_id=elem['id'],
                osm_type='node',
                center_lat=elem['lat'],
                center_lon=elem['lon'],
                building_type=building_type,
                nodes=[(elem['lat'], elem['lon'])]
            ))

    return caravans


def latlon_to_pixel(lat: float, lon: float, bbox: BoundingBox,
                    img_width: int, img_height: int) -> Tuple[int, int]:
    """Convert lat/lon coordinates to pixel coordinates within an image."""
    # Normalize to 0-1 range within bbox
    x_norm = (lon - bbox.min_lon) / (bbox.max_lon - bbox.min_lon)
    y_norm = (bbox.max_lat - lat) / (bbox.max_lat - bbox.min_lat)  # Inverted for image coords

    # Convert to pixel coordinates
    x = int(x_norm * img_width)
    y = int(y_norm * img_height)

    return (x, y)


def draw_polygons_on_image(image_path: str, caravans: List[CaravanPolygon],
                           bbox: BoundingBox, output_path: str,
                           outline_color: str = 'red', fill_color: str = None,
                           outline_width: int = 2) -> bool:
    """
    Draw caravan polygons on a satellite image.

    Args:
        image_path: Path to input satellite image
        caravans: List of CaravanPolygon objects to draw
        bbox: Bounding box of the image
        output_path: Path to save the annotated image
        outline_color: Color for polygon outlines
        fill_color: Optional fill color (with alpha for transparency)
        outline_width: Width of outline in pixels

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Create overlay for semi-transparent fills
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        img_width, img_height = img.size

        for caravan in caravans:
            if len(caravan.nodes) >= 3:
                # It's a polygon (way)
                pixels = [latlon_to_pixel(lat, lon, bbox, img_width, img_height)
                          for lat, lon in caravan.nodes]

                # Draw filled polygon with transparency
                if fill_color:
                    draw.polygon(pixels, fill=fill_color)

                # Draw outline
                draw.polygon(pixels, outline=outline_color, width=outline_width)

            elif len(caravan.nodes) == 1:
                # It's a point (node) - draw a circle
                lat, lon = caravan.nodes[0]
                x, y = latlon_to_pixel(lat, lon, bbox, img_width, img_height)
                radius = 5
                draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                             outline=outline_color, fill=fill_color, width=outline_width)

        # Composite overlay onto original image
        img = Image.alpha_composite(img, overlay)

        # Save result
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        img.save(output_path)
        return True

    except Exception as e:
        print(f"    Error drawing polygons: {e}")
        return False


def scan_image_folders(root_dir: str) -> Dict[str, List[Dict]]:
    """
    Scan directory for satellite images organized by park name.

    Returns dict mapping park_name -> list of image info dicts
    """
    parks = defaultdict(list)

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue

            parsed = parse_filename(filename)
            if parsed is None:
                continue

            # Park name is the folder name (relative to root_dir)
            rel_path = os.path.relpath(root, root_dir)
            park_name = rel_path if rel_path != '.' else os.path.basename(root)

            image_info = {
                'filename': filename,
                'filepath': os.path.join(root, filename),
                'parsed': parsed,
                'bbox': get_bbox_from_parsed(parsed)
            }
            parks[park_name].append(image_info)

    return dict(parks)


def process_park(park_name: str, images: List[Dict], output_dir: str = None,
                 save_overlays: bool = False, rate_limit: float = 1.0) -> Dict:
    """
    Process all images for a single park, deduplicating caravans.

    Returns dict with park results including deduplicated caravan count.
    """
    all_caravans: Dict[str, CaravanPolygon] = {}  # unique_id -> polygon
    image_results = []

    print(f"\n  Processing {len(images)} images...")

    for i, img_info in enumerate(images):
        filename = img_info['filename']
        filepath = img_info['filepath']
        bbox = img_info['bbox']

        print(f"    [{i+1}/{len(images)}] {filename[:50]}...", end=' ')

        # Query OSM for this image's bbox
        caravans = get_caravans_in_bbox(bbox)

        # Track which caravans are in this image
        image_caravan_ids = set()
        new_caravans = 0

        for caravan in caravans:
            uid = caravan.unique_id()
            image_caravan_ids.add(uid)
            if uid not in all_caravans:
                all_caravans[uid] = caravan
                new_caravans += 1

        print(f"found {len(caravans)} ({new_caravans} new)")

        # Draw overlays if requested
        overlay_path = None
        if save_overlays and output_dir and caravans:
            overlay_dir = os.path.join(output_dir, 'overlays', park_name)
            overlay_path = os.path.join(overlay_dir, f"overlay_{filename}")

            # Use semi-transparent red fill
            success = draw_polygons_on_image(
                filepath, caravans, bbox, overlay_path,
                outline_color='red',
                fill_color=(255, 0, 0, 80),  # Semi-transparent red
                outline_width=2
            )
            if success:
                print(f"      Saved overlay: {overlay_path}")

        image_results.append({
            'filename': filename,
            'caravans_in_image': len(caravans),
            'new_caravans': new_caravans,
            'overlay_path': overlay_path
        })

        # Rate limiting
        if i < len(images) - 1:
            time.sleep(rate_limit)

    # Count by type
    type_counts = defaultdict(int)
    for caravan in all_caravans.values():
        type_counts[caravan.building_type] += 1

    return {
        'park_name': park_name,
        'total_images': len(images),
        'total_caravans_deduplicated': len(all_caravans),
        'static_caravans': type_counts.get('static_caravan', 0),
        'mobile_homes': type_counts.get('mobile_home', 0),
        'other_caravans': type_counts.get('caravan', 0),
        'image_results': image_results,
        'all_caravans': all_caravans
    }


def process_all_parks(input_dir: str, output_excel: str,
                      save_overlays: bool = False, rate_limit: float = 1.0) -> pd.DataFrame:
    """
    Process all caravan parks found in the input directory.

    Args:
        input_dir: Root directory containing park folders
        output_excel: Path to output Excel file
        save_overlays: Whether to save overlay images
        rate_limit: Seconds between API calls

    Returns:
        DataFrame with results for all parks
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

    # Determine output directory for overlays
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
            rate_limit=rate_limit
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

    # Create DataFrame and save to Excel
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
    # Basic usage
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx

    # Save overlay images with polygons drawn
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --save_overlays

Expected folder structure:
    images/
    ├── Park Name 1/
    │   ├── 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png
    │   └── ...
    └── Park Name 2/
        └── ...

Filename format:
    {id}_z{zoom}_TL_{lat}_{lon}_BR_{lat}_{lon}_recent_{date}.{ext}
        """
    )

    parser.add_argument('--input_dir', required=True,
                        help='Root directory containing park folders with satellite images')
    parser.add_argument('--output', required=True,
                        help='Path to output Excel file')
    parser.add_argument('--save_overlays', action='store_true',
                        help='Save images with polygon overlays')
    parser.add_argument('--rate_limit', type=float, default=1.0,
                        help='Seconds between API calls (default: 1.0)')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        df = process_all_parks(
            input_dir=args.input_dir,
            output_excel=args.output,
            save_overlays=args.save_overlays,
            rate_limit=args.rate_limit
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
