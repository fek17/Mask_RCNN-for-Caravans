"""
Caravan Counter Module - OpenStreetMap Version

A module for counting caravan polygons from OpenStreetMap and overlaying them
on satellite images. Can be imported and used in Jupyter notebooks or scripts.

Example usage:
    from caravan_counter import CaravanCounter

    # Initialize counter
    counter = CaravanCounter()

    # Count caravans at a location
    result = counter.count_at_location(lat=51.5074, lon=-0.1278, radius=500)
    print(f"Found {result['total_polygons']} caravans")

    # Process satellite images from folders
    df = counter.process_image_folders(
        input_dir='./images',
        output_excel='results.xlsx',
        save_overlays=True
    )

    # Process Excel file with coordinates
    df = counter.process_excel('sites.xlsx', 'results.xlsx')

Requirements:
    pip install pandas openpyxl requests Pillow numpy
"""

import os
import re
import time
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


class CaravanCounter:
    """
    A class for counting caravan polygons from OpenStreetMap.

    Can process:
    - Single locations (lat/lon with radius)
    - Bounding boxes
    - Excel files with coordinates
    - Folders of satellite images with coordinates in filenames

    Example:
        counter = CaravanCounter()

        # Single location
        result = counter.count_at_location(51.5, -0.1, radius=500)

        # Process satellite image folders
        df = counter.process_image_folders('./images', 'results.xlsx')
    """

    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize the CaravanCounter.

        Args:
            rate_limit: Seconds to wait between API calls (default: 1.0)
        """
        self.rate_limit = rate_limit
        self._last_request_time = 0

    def _query_overpass(self, query: str, max_retries: int = 3) -> Optional[Dict]:
        """Execute an Overpass API query with retry logic and rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OVERPASS_URL,
                    data={'data': query},
                    timeout=60
                )
                response.raise_for_status()
                self._last_request_time = time.time()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"  Request failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return None
        return None

    def count_at_location(self, lat: float, lon: float, radius: float = 500) -> Dict:
        """
        Count caravan polygons within a radius of a point.

        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            radius: Search radius in meters (default 500m)

        Returns:
            Dictionary with counts and polygon details
        """
        query = f"""
        [out:json][timeout:30];
        (
          way["building"="static_caravan"](around:{radius},{lat},{lon});
          way["building"="caravan"](around:{radius},{lat},{lon});
          way["building"="mobile_home"](around:{radius},{lat},{lon});
          node["building"="static_caravan"](around:{radius},{lat},{lon});
          node["building"="caravan"](around:{radius},{lat},{lon});
          node["building"="mobile_home"](around:{radius},{lat},{lon});
        );
        out body;
        """

        result = {
            'total_polygons': 0,
            'static_caravans': 0,
            'mobile_homes': 0,
            'caravan_buildings': 0,
            'error': None
        }

        data = self._query_overpass(query)

        if data is None:
            result['error'] = 'API request failed'
            return result

        elements = data.get('elements', [])

        for elem in elements:
            tags = elem.get('tags', {})
            building_type = tags.get('building', '')

            if building_type == 'static_caravan':
                result['static_caravans'] += 1
            elif building_type == 'caravan':
                result['caravan_buildings'] += 1
            elif building_type == 'mobile_home':
                result['mobile_homes'] += 1

        result['total_polygons'] = len(elements)
        return result

    def get_caravans_in_bbox(self, bbox: BoundingBox) -> List[CaravanPolygon]:
        """
        Get caravan polygons within a bounding box with full geometry.

        Args:
            bbox: BoundingBox object

        Returns:
            List of CaravanPolygon objects
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

        data = self._query_overpass(query)
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

    @staticmethod
    def parse_image_filename(filename: str) -> Optional[Dict]:
        """
        Parse satellite image filename to extract coordinates.

        Expected format: 0001_z18_TL_50.020742_-5.102061_BR_50.013683_-5.091075_recent_2025-03-27.png
        """
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

    @staticmethod
    def bbox_from_parsed(parsed: Dict) -> BoundingBox:
        """Create BoundingBox from parsed filename data."""
        return BoundingBox(
            min_lat=min(parsed['tl_lat'], parsed['br_lat']),
            max_lat=max(parsed['tl_lat'], parsed['br_lat']),
            min_lon=min(parsed['tl_lon'], parsed['br_lon']),
            max_lon=max(parsed['tl_lon'], parsed['br_lon'])
        )

    def draw_polygons_on_image(self, image_path: str, caravans: List[CaravanPolygon],
                                bbox: BoundingBox, output_path: str,
                                outline_color: str = 'red',
                                fill_color: Tuple[int, int, int, int] = (255, 0, 0, 80),
                                outline_width: int = 2) -> bool:
        """
        Draw caravan polygons on a satellite image.

        Args:
            image_path: Path to input satellite image
            caravans: List of CaravanPolygon objects
            bbox: Bounding box of the image
            output_path: Path to save annotated image
            outline_color: Color for polygon outlines
            fill_color: RGBA tuple for semi-transparent fill
            outline_width: Width of outline in pixels

        Returns:
            True if successful
        """
        try:
            img = Image.open(image_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            img_width, img_height = img.size

            for caravan in caravans:
                pixels = []
                for lat, lon in caravan.nodes:
                    x_norm = (lon - bbox.min_lon) / (bbox.max_lon - bbox.min_lon)
                    y_norm = (bbox.max_lat - lat) / (bbox.max_lat - bbox.min_lat)
                    x = int(x_norm * img_width)
                    y = int(y_norm * img_height)
                    pixels.append((x, y))

                if len(pixels) >= 3:
                    if fill_color:
                        draw.polygon(pixels, fill=fill_color)
                    draw.polygon(pixels, outline=outline_color, width=outline_width)
                elif len(pixels) == 1:
                    x, y = pixels[0]
                    radius = 5
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                                 outline=outline_color, fill=fill_color, width=outline_width)

            img = Image.alpha_composite(img, overlay)
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            img.save(output_path)
            return True

        except Exception as e:
            print(f"Error drawing polygons: {e}")
            return False

    def process_image_folders(self, input_dir: str, output_excel: str = None,
                               save_overlays: bool = False) -> pd.DataFrame:
        """
        Process satellite images organized in folders by park name.

        Args:
            input_dir: Root directory containing park folders
            output_excel: Path to output Excel file (optional)
            save_overlays: Whether to save images with polygon overlays

        Returns:
            DataFrame with deduplicated caravan counts per park
        """
        # Scan for images
        parks = defaultdict(list)
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    continue

                parsed = self.parse_image_filename(filename)
                if parsed is None:
                    continue

                rel_path = os.path.relpath(root, input_dir)
                park_name = rel_path if rel_path != '.' else os.path.basename(root)

                parks[park_name].append({
                    'filename': filename,
                    'filepath': os.path.join(root, filename),
                    'parsed': parsed,
                    'bbox': self.bbox_from_parsed(parsed)
                })

        if not parks:
            print("No valid satellite images found!")
            return pd.DataFrame()

        print(f"Found {len(parks)} parks")

        # Process each park
        results = []
        output_dir = os.path.dirname(output_excel) if output_excel else '.'

        for park_name, images in parks.items():
            print(f"\n{'='*50}")
            print(f"Park: {park_name} ({len(images)} images)")
            print(f"{'='*50}")

            all_caravans: Dict[str, CaravanPolygon] = {}

            for i, img_info in enumerate(images):
                print(f"  [{i+1}/{len(images)}] {img_info['filename'][:40]}...", end=' ')

                caravans = self.get_caravans_in_bbox(img_info['bbox'])
                new_count = 0

                for caravan in caravans:
                    uid = caravan.unique_id()
                    if uid not in all_caravans:
                        all_caravans[uid] = caravan
                        new_count += 1

                print(f"found {len(caravans)} ({new_count} new)")

                # Save overlay if requested
                if save_overlays and caravans:
                    overlay_dir = os.path.join(output_dir, 'overlays', park_name)
                    overlay_path = os.path.join(overlay_dir, f"overlay_{img_info['filename']}")
                    self.draw_polygons_on_image(
                        img_info['filepath'], caravans, img_info['bbox'], overlay_path
                    )

            # Count by type
            type_counts = defaultdict(int)
            for caravan in all_caravans.values():
                type_counts[caravan.building_type] += 1

            results.append({
                'park_name': park_name,
                'total_images': len(images),
                'caravan_count': len(all_caravans),
                'static_caravans': type_counts.get('static_caravan', 0),
                'mobile_homes': type_counts.get('mobile_home', 0),
                'other_caravans': type_counts.get('caravan', 0)
            })

            print(f"  TOTAL (deduplicated): {len(all_caravans)} caravans")

        df = pd.DataFrame(results)

        if output_excel:
            df.to_excel(output_excel, index=False)
            print(f"\nSaved to: {output_excel}")

        print(f"\nTotal caravans across all parks: {df['caravan_count'].sum()}")
        return df

    def process_excel(self, input_excel: str, output_excel: str = None,
                      lat_column: str = None, lon_column: str = None,
                      radius: float = 500) -> pd.DataFrame:
        """
        Process caravan sites from an Excel file with coordinates.

        Args:
            input_excel: Path to input Excel file
            output_excel: Path to output Excel file (optional)
            lat_column: Name of latitude column (auto-detected if None)
            lon_column: Name of longitude column (auto-detected if None)
            radius: Search radius in meters

        Returns:
            DataFrame with caravan counts
        """
        df = pd.read_excel(input_excel)
        print(f"Read {len(df)} rows from {input_excel}")

        # Auto-detect columns
        lat_col = self._find_column(df, ['latitude', 'lat', 'Latitude', 'Lat'])
        lon_col = self._find_column(df, ['longitude', 'lon', 'lng', 'Longitude', 'Lon'])

        if lat_column:
            lat_col = lat_column
        if lon_column:
            lon_col = lon_column

        if not lat_col or not lon_col:
            raise ValueError(f"Could not find lat/lon columns. Available: {list(df.columns)}")

        name_col = self._find_column(df, ['site_name', 'name', 'site', 'Name', 'Site'])

        df['caravan_count'] = 0
        df['static_caravans'] = 0
        df['mobile_homes'] = 0
        df['other_caravans'] = 0
        df['query_status'] = ''

        for idx, row in df.iterrows():
            site_name = row[name_col] if name_col else f"Site {idx + 1}"
            lat = float(row[lat_col])
            lon = float(row[lon_col])

            print(f"[{idx + 1}/{len(df)}] {site_name}", end=' ')

            result = self.count_at_location(lat, lon, radius)

            if result['error']:
                df.at[idx, 'query_status'] = f"Error: {result['error']}"
                print(f"-> Error")
            else:
                df.at[idx, 'caravan_count'] = result['total_polygons']
                df.at[idx, 'static_caravans'] = result['static_caravans']
                df.at[idx, 'mobile_homes'] = result['mobile_homes']
                df.at[idx, 'other_caravans'] = result['caravan_buildings']
                df.at[idx, 'query_status'] = 'Success'
                print(f"-> {result['total_polygons']} caravans")

        if output_excel:
            df.to_excel(output_excel, index=False)
            print(f"\nSaved to: {output_excel}")

        return df

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column name."""
        for col in candidates:
            if col in df.columns:
                return col
            for df_col in df.columns:
                if df_col.lower() == col.lower():
                    return df_col
        return None


# Convenience function
def count_caravans(lat: float, lon: float, radius: float = 500) -> int:
    """Quick function to count caravans at a location."""
    counter = CaravanCounter(rate_limit=0)
    result = counter.count_at_location(lat, lon, radius)
    return result['total_polygons']
