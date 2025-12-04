"""
Count Caravans from Satellite Images with OSM Overlay

This script processes satellite image screenshots organized in folders (by caravan park name),
loads caravan polygons from a local GeoJSON file (or queries OpenStreetMap), overlays
the polygons on the images, and outputs deduplicated counts to Excel.

Expected folder structure:
    images/
    ├── Park Name 1/
    │   ├── 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png
    │   └── ...
    └── Park Name 2/
        └── ...

Filename format:
    {id}_z{zoom}_TL_{lat1}_{lon1}_BR_{lat2}_{lon2}_recent_{date}.{ext}

Usage:
    # Using local GeoJSON (recommended - no API rate limits!)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --geojson uk_caravans.geojson

    # Using Overpass API (has rate limits)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx

    # With overlay images saved
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --geojson uk_caravans.geojson --save_overlays

Requirements:
    pip install pandas openpyxl requests Pillow

Copyright (c) 2024
"""

import os
import sys
import re
import json
import argparse
import time
import requests
import pandas as pd
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional
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


class GeoJSONCaravanStore:
    """Loads and queries caravan polygons from a local GeoJSON file."""

    def __init__(self, geojson_path: str):
        """Load GeoJSON file and index caravans for fast bbox queries."""
        print(f"Loading GeoJSON from {geojson_path}...")

        with open(geojson_path, 'r') as f:
            data = json.load(f)

        self.caravans: List[CaravanPolygon] = []

        for feature in data.get('features', []):
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})

            if geom is None:
                continue

            osm_id = props.get('osm_id', 0)
            osm_type = props.get('osm_type', 'unknown')
            building_type = props.get('building', 'unknown')

            if geom['type'] == 'Polygon':
                # GeoJSON coordinates are [lon, lat], we need (lat, lon)
                coords = geom['coordinates'][0]  # Outer ring
                nodes = [(lat, lon) for lon, lat in coords]

                if nodes:
                    center_lat = sum(n[0] for n in nodes) / len(nodes)
                    center_lon = sum(n[1] for n in nodes) / len(nodes)

                    self.caravans.append(CaravanPolygon(
                        osm_id=osm_id,
                        osm_type=osm_type,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        building_type=building_type,
                        nodes=nodes
                    ))

            elif geom['type'] == 'Point':
                lon, lat = geom['coordinates']
                self.caravans.append(CaravanPolygon(
                    osm_id=osm_id,
                    osm_type=osm_type,
                    center_lat=lat,
                    center_lon=lon,
                    building_type=building_type,
                    nodes=[(lat, lon)]
                ))

        print(f"Loaded {len(self.caravans)} caravan polygons from GeoJSON")

    def get_caravans_in_bbox(self, bbox: BoundingBox) -> List[CaravanPolygon]:
        """Get all caravans whose center falls within the bounding box."""
        results = []
        for caravan in self.caravans:
            if bbox.contains_point(caravan.center_lat, caravan.center_lon):
                results.append(caravan)
        return results


# Global store for GeoJSON data (loaded once)
_geojson_store: Optional[GeoJSONCaravanStore] = None


def load_geojson_store(geojson_path: str) -> GeoJSONCaravanStore:
    """Load GeoJSON store (cached globally)."""
    global _geojson_store
    if _geojson_store is None:
        _geojson_store = GeoJSONCaravanStore(geojson_path)
    return _geojson_store


def parse_filename(filename: str) -> Optional[Dict]:
    """Parse satellite image filename to extract coordinates."""
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


def get_caravans_in_bbox_api(bbox: BoundingBox) -> List[CaravanPolygon]:
    """Query OSM Overpass API for caravan polygons within a bounding box."""
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
            nodes = []
            for node_id in elem.get('nodes', []):
                if node_id in node_coords:
                    nodes.append(node_coords[node_id])

            if nodes:
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


def get_caravans_in_bbox(bbox: BoundingBox, geojson_store: Optional[GeoJSONCaravanStore] = None) -> List[CaravanPolygon]:
    """Get caravans in bbox from GeoJSON store or API."""
    if geojson_store:
        return geojson_store.get_caravans_in_bbox(bbox)
    else:
        return get_caravans_in_bbox_api(bbox)


def latlon_to_pixel(lat: float, lon: float, bbox: BoundingBox,
                    img_width: int, img_height: int) -> Tuple[int, int]:
    """Convert lat/lon coordinates to pixel coordinates within an image."""
    x_norm = (lon - bbox.min_lon) / (bbox.max_lon - bbox.min_lon)
    y_norm = (bbox.max_lat - lat) / (bbox.max_lat - bbox.min_lat)

    x = int(x_norm * img_width)
    y = int(y_norm * img_height)

    return (x, y)


def draw_polygons_on_image(image_path: str, caravans: List[CaravanPolygon],
                           bbox: BoundingBox, output_path: str,
                           outline_color: str = 'red',
                           fill_color: Tuple[int, int, int, int] = (255, 0, 0, 80),
                           outline_width: int = 2) -> bool:
    """Draw caravan polygons on a satellite image."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        img_width, img_height = img.size

        for caravan in caravans:
            if len(caravan.nodes) >= 3:
                pixels = [latlon_to_pixel(lat, lon, bbox, img_width, img_height)
                          for lat, lon in caravan.nodes]

                if fill_color:
                    draw.polygon(pixels, fill=fill_color)
                draw.polygon(pixels, outline=outline_color, width=outline_width)

            elif len(caravan.nodes) == 1:
                lat, lon = caravan.nodes[0]
                x, y = latlon_to_pixel(lat, lon, bbox, img_width, img_height)
                radius = 5
                draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                             outline=outline_color, fill=fill_color, width=outline_width)

        img = Image.alpha_composite(img, overlay)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        img.save(output_path)
        return True

    except Exception as e:
        print(f"    Error drawing polygons: {e}")
        return False


def scan_image_folders(root_dir: str) -> Dict[str, List[Dict]]:
    """Scan directory for satellite images organized by park name."""
    parks = defaultdict(list)

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue

            parsed = parse_filename(filename)
            if parsed is None:
                continue

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
                 save_overlays: bool = False, geojson_store: Optional[GeoJSONCaravanStore] = None,
                 rate_limit: float = 1.0) -> Dict:
    """Process all images for a single park, deduplicating caravans."""
    all_caravans: Dict[str, CaravanPolygon] = {}
    image_results = []

    print(f"\n  Processing {len(images)} images...")

    for i, img_info in enumerate(images):
        filename = img_info['filename']
        filepath = img_info['filepath']
        bbox = img_info['bbox']

        print(f"    [{i+1}/{len(images)}] {filename[:50]}...", end=' ')

        # Get caravans for this image's bbox
        caravans = get_caravans_in_bbox(bbox, geojson_store)

        # Track and deduplicate
        new_caravans = 0
        for caravan in caravans:
            uid = caravan.unique_id()
            if uid not in all_caravans:
                all_caravans[uid] = caravan
                new_caravans += 1

        print(f"found {len(caravans)} ({new_caravans} new)")

        # Draw overlays if requested
        if save_overlays and output_dir and caravans:
            overlay_dir = os.path.join(output_dir, 'overlays', park_name)
            overlay_path = os.path.join(overlay_dir, f"overlay_{filename}")

            success = draw_polygons_on_image(
                filepath, caravans, bbox, overlay_path,
                outline_color='red',
                fill_color=(255, 0, 0, 80),
                outline_width=2
            )
            if success:
                print(f"      Saved overlay: {overlay_path}")

        image_results.append({
            'filename': filename,
            'caravans_in_image': len(caravans),
            'new_caravans': new_caravans
        })

        # Rate limiting (only for API mode)
        if geojson_store is None and i < len(images) - 1:
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
                      save_overlays: bool = False, geojson_path: str = None,
                      rate_limit: float = 1.0) -> pd.DataFrame:
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

    # Load GeoJSON store if provided
    geojson_store = None
    if geojson_path:
        geojson_store = load_geojson_store(geojson_path)
    else:
        print("\nWARNING: No GeoJSON file provided. Using Overpass API (may hit rate limits).")
        print("To avoid rate limits, first run: python download_uk_caravans.py")

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
            geojson_store=geojson_store,
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
    # Using local GeoJSON (recommended - no rate limits!)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --geojson uk_caravans.geojson

    # With overlay images
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx --geojson uk_caravans.geojson --save_overlays

    # Using Overpass API (has rate limits)
    python count_caravans_from_images.py --input_dir ./images --output results.xlsx

To download UK caravan data first (recommended):
    python download_uk_caravans.py --output uk_caravans.geojson
        """
    )

    parser.add_argument('--input_dir', required=True,
                        help='Root directory containing park folders with satellite images')
    parser.add_argument('--output', required=True,
                        help='Path to output Excel file')
    parser.add_argument('--geojson', default=None,
                        help='Path to GeoJSON file with caravan data (recommended)')
    parser.add_argument('--save_overlays', action='store_true',
                        help='Save images with polygon overlays')
    parser.add_argument('--rate_limit', type=float, default=1.0,
                        help='Seconds between API calls if not using GeoJSON (default: 1.0)')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if args.geojson and not os.path.exists(args.geojson):
        print(f"Error: GeoJSON file not found: {args.geojson}")
        sys.exit(1)

    try:
        df = process_all_parks(
            input_dir=args.input_dir,
            output_excel=args.output,
            save_overlays=args.save_overlays,
            geojson_path=args.geojson,
            rate_limit=args.rate_limit
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
