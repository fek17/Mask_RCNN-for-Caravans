"""
Count Caravans from OpenStreetMap

This script reads a list of caravan sites from an Excel file, queries OpenStreetMap
for caravan/mobile home polygons at each location, counts them, and outputs all
original data plus the polygon count to a new Excel file.

Usage:
    python count_caravans_excel.py --input sites.xlsx --output results.xlsx

    # With custom column names
    python count_caravans_excel.py --input sites.xlsx --output results.xlsx \
        --lat_column latitude --lon_column longitude --radius 500

Required columns in Excel (configurable):
    - latitude (or lat/Latitude/Lat)
    - longitude (or lon/lng/Longitude/Lon)

Or bounding box columns:
    - min_lat, max_lat, min_lon, max_lon

Optional columns:
    - site_name / name
    - radius (in meters, overrides default)

Requirements:
    pip install pandas openpyxl requests

Copyright (c) 2024
"""

import os
import sys
import argparse
import time
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tags that represent individual caravans/mobile homes
CARAVAN_TAGS = [
    'building=static_caravan',
    'building=caravan',
    'building=mobile_home',
    'tourism=caravan_site',  # Sometimes individual pitches are tagged
    'amenity=caravan_parking',
    'leisure=pitch',  # Camping pitches (may include caravans)
]


def query_overpass(query: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Execute an Overpass API query with retry logic.

    Args:
        query: Overpass QL query string
        max_retries: Maximum number of retry attempts

    Returns:
        JSON response as dictionary, or None if failed
    """
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
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                print(f"    Request failed, retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return None
    return None


def count_caravans_at_location(lat: float, lon: float, radius: float = 500) -> Dict:
    """
    Count caravan polygons within a radius of a point using Overpass API.

    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius: Search radius in meters (default 500m)

    Returns:
        Dictionary with count details
    """
    # Build Overpass query for caravan-related features
    # We query for nodes, ways, and relations that represent caravans
    query = f"""
    [out:json][timeout:30];
    (
      // Static caravans and mobile homes (buildings)
      way["building"="static_caravan"](around:{radius},{lat},{lon});
      way["building"="caravan"](around:{radius},{lat},{lon});
      way["building"="mobile_home"](around:{radius},{lat},{lon});

      // Also check for nodes (some are mapped as points)
      node["building"="static_caravan"](around:{radius},{lat},{lon});
      node["building"="caravan"](around:{radius},{lat},{lon});
      node["building"="mobile_home"](around:{radius},{lat},{lon});

      // Caravan pitches
      way["tourism"="caravan_site"]["pitch"](around:{radius},{lat},{lon});
      way["leisure"="pitch"]["tents"="no"](around:{radius},{lat},{lon});
    );
    out count;
    out body;
    """

    result = {
        'total_polygons': 0,
        'static_caravans': 0,
        'mobile_homes': 0,
        'caravan_buildings': 0,
        'pitches': 0,
        'error': None,
        'raw_elements': []
    }

    data = query_overpass(query)

    if data is None:
        result['error'] = 'API request failed'
        return result

    elements = data.get('elements', [])
    result['raw_elements'] = elements

    # Count by type
    for elem in elements:
        tags = elem.get('tags', {})
        building_type = tags.get('building', '')

        if building_type == 'static_caravan':
            result['static_caravans'] += 1
        elif building_type == 'caravan':
            result['caravan_buildings'] += 1
        elif building_type == 'mobile_home':
            result['mobile_homes'] += 1
        elif tags.get('leisure') == 'pitch' or tags.get('tourism') == 'caravan_site':
            result['pitches'] += 1

    result['total_polygons'] = len(elements)

    return result


def count_caravans_in_bbox(min_lat: float, min_lon: float,
                           max_lat: float, max_lon: float) -> Dict:
    """
    Count caravan polygons within a bounding box.

    Args:
        min_lat, min_lon: Southwest corner
        max_lat, max_lon: Northeast corner

    Returns:
        Dictionary with count details
    """
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
    [out:json][timeout:30];
    (
      // Static caravans and mobile homes (buildings)
      way["building"="static_caravan"]({bbox});
      way["building"="caravan"]({bbox});
      way["building"="mobile_home"]({bbox});

      // Also check for nodes
      node["building"="static_caravan"]({bbox});
      node["building"="caravan"]({bbox});
      node["building"="mobile_home"]({bbox});

      // Caravan pitches
      way["leisure"="pitch"]["tents"="no"]({bbox});
    );
    out count;
    out body;
    """

    result = {
        'total_polygons': 0,
        'static_caravans': 0,
        'mobile_homes': 0,
        'caravan_buildings': 0,
        'pitches': 0,
        'error': None,
        'raw_elements': []
    }

    data = query_overpass(query)

    if data is None:
        result['error'] = 'API request failed'
        return result

    elements = data.get('elements', [])
    result['raw_elements'] = elements

    for elem in elements:
        tags = elem.get('tags', {})
        building_type = tags.get('building', '')

        if building_type == 'static_caravan':
            result['static_caravans'] += 1
        elif building_type == 'caravan':
            result['caravan_buildings'] += 1
        elif building_type == 'mobile_home':
            result['mobile_homes'] += 1
        elif tags.get('leisure') == 'pitch':
            result['pitches'] += 1

    result['total_polygons'] = len(elements)

    return result


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
        # Case-insensitive search
        for df_col in df.columns:
            if df_col.lower() == col.lower():
                return df_col
    return None


def process_excel(input_path: str, output_path: str,
                  lat_col: str = None, lon_col: str = None,
                  radius: float = 500, sheet_name=0,
                  rate_limit: float = 1.0) -> pd.DataFrame:
    """
    Process Excel file: read sites, count OSM caravan polygons, save results.

    Args:
        input_path: Path to input Excel file
        output_path: Path to output Excel file
        lat_col: Column name for latitude (auto-detected if None)
        lon_col: Column name for longitude (auto-detected if None)
        radius: Default search radius in meters
        sheet_name: Sheet name or index to read
        rate_limit: Seconds to wait between API calls

    Returns:
        DataFrame with results
    """
    print(f"\nReading Excel file: {input_path}")
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    print(f"Found {len(df)} rows")
    print(f"Columns: {', '.join(df.columns.tolist())}")

    # Auto-detect coordinate columns
    lat_candidates = ['latitude', 'lat', 'Latitude', 'Lat', 'LAT', 'y', 'Y']
    lon_candidates = ['longitude', 'lon', 'lng', 'Longitude', 'Lon', 'LON', 'Long', 'x', 'X']

    if lat_col is None:
        lat_col = find_column(df, lat_candidates)
    if lon_col is None:
        lon_col = find_column(df, lon_candidates)

    # Check for bounding box columns
    bbox_cols = {
        'min_lat': find_column(df, ['min_lat', 'south', 'min_latitude']),
        'max_lat': find_column(df, ['max_lat', 'north', 'max_latitude']),
        'min_lon': find_column(df, ['min_lon', 'west', 'min_longitude']),
        'max_lon': find_column(df, ['max_lon', 'east', 'max_longitude']),
    }

    use_bbox = all(bbox_cols.values())
    use_point = lat_col and lon_col

    if not use_bbox and not use_point:
        raise ValueError(
            "Could not find coordinate columns. Need either:\n"
            "  - latitude/longitude columns, or\n"
            "  - bounding box columns (min_lat, max_lat, min_lon, max_lon)\n"
            f"Available columns: {', '.join(df.columns.tolist())}"
        )

    if use_point:
        print(f"\nUsing point coordinates: {lat_col}, {lon_col}")
        print(f"Search radius: {radius}m")
    else:
        print(f"\nUsing bounding boxes")

    # Check for radius column override
    radius_col = find_column(df, ['radius', 'search_radius', 'buffer'])
    if radius_col:
        print(f"Found radius column: {radius_col}")

    # Check for site name column
    name_col = find_column(df, ['site_name', 'name', 'site', 'title', 'Name', 'Site'])

    # Initialize result columns
    df['caravan_count'] = 0
    df['static_caravans'] = 0
    df['mobile_homes'] = 0
    df['other_caravans'] = 0
    df['pitches'] = 0
    df['query_status'] = ''

    # Process each row
    total = len(df)
    for idx, row in df.iterrows():
        # Get site name for display
        site_name = row[name_col] if name_col else f"Site {idx + 1}"
        print(f"\n[{idx + 1}/{total}] {site_name}")

        # Get coordinates
        if use_bbox:
            min_lat = float(row[bbox_cols['min_lat']])
            max_lat = float(row[bbox_cols['max_lat']])
            min_lon = float(row[bbox_cols['min_lon']])
            max_lon = float(row[bbox_cols['max_lon']])
            print(f"  BBox: ({min_lat:.5f}, {min_lon:.5f}) to ({max_lat:.5f}, {max_lon:.5f})")

            result = count_caravans_in_bbox(min_lat, min_lon, max_lat, max_lon)
        else:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            r = float(row[radius_col]) if radius_col and pd.notna(row[radius_col]) else radius
            print(f"  Location: ({lat:.5f}, {lon:.5f}), radius: {r}m")

            result = count_caravans_at_location(lat, lon, r)

        # Update dataframe
        if result['error']:
            df.at[idx, 'query_status'] = f"Error: {result['error']}"
            print(f"  ERROR: {result['error']}")
        else:
            df.at[idx, 'caravan_count'] = result['total_polygons']
            df.at[idx, 'static_caravans'] = result['static_caravans']
            df.at[idx, 'mobile_homes'] = result['mobile_homes']
            df.at[idx, 'other_caravans'] = result['caravan_buildings']
            df.at[idx, 'pitches'] = result['pitches']
            df.at[idx, 'query_status'] = 'Success'
            print(f"  Found {result['total_polygons']} polygons "
                  f"(static: {result['static_caravans']}, "
                  f"mobile: {result['mobile_homes']}, "
                  f"other: {result['caravan_buildings']}, "
                  f"pitches: {result['pitches']})")

        # Rate limiting to be nice to Overpass API
        if idx < total - 1:
            time.sleep(rate_limit)

    # Save results
    print(f"\n\nSaving results to: {output_path}")
    df.to_excel(output_path, index=False)

    # Print summary
    successful = df[df['query_status'] == 'Success']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total sites processed: {total}")
    print(f"Successful queries: {len(successful)}")
    print(f"Failed queries: {total - len(successful)}")
    print(f"{'='*60}")
    print(f"Total caravan polygons found: {df['caravan_count'].sum()}")
    print(f"  - Static caravans: {df['static_caravans'].sum()}")
    print(f"  - Mobile homes: {df['mobile_homes'].sum()}")
    print(f"  - Other caravans: {df['other_caravans'].sum()}")
    print(f"  - Pitches: {df['pitches'].sum()}")
    print(f"{'='*60}")
    print(f"Average caravans per site: {df['caravan_count'].mean():.1f}")
    print(f"Max caravans at single site: {df['caravan_count'].max()}")
    print(f"Sites with zero caravans: {len(df[df['caravan_count'] == 0])}")
    print(f"{'='*60}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Count caravan polygons from OpenStreetMap for sites listed in Excel.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-detect coordinate columns)
    python count_caravans_excel.py --input sites.xlsx --output results.xlsx

    # Specify coordinate columns
    python count_caravans_excel.py --input sites.xlsx --output results.xlsx \\
        --lat_column Latitude --lon_column Longitude

    # Custom search radius
    python count_caravans_excel.py --input sites.xlsx --output results.xlsx --radius 1000

Expected Excel format:
    Your Excel should contain either:

    1. Point coordinates (latitude, longitude columns):
       | site_name    | latitude  | longitude  |
       |--------------|-----------|------------|
       | Park A       | 51.5074   | -0.1278    |
       | Park B       | 52.4862   | -1.8904    |

    2. Bounding boxes (min/max lat/lon columns):
       | site_name | min_lat | max_lat | min_lon | max_lon |
       |-----------|---------|---------|---------|---------|
       | Park A    | 51.50   | 51.52   | -0.13   | -0.11   |

OSM tags queried:
    - building=static_caravan
    - building=caravan
    - building=mobile_home
    - leisure=pitch (camping pitches)
        """
    )

    parser.add_argument('--input', required=True,
                        help='Path to input Excel file')
    parser.add_argument('--output', required=True,
                        help='Path to output Excel file')
    parser.add_argument('--lat_column', default=None,
                        help='Column name for latitude (auto-detected if not specified)')
    parser.add_argument('--lon_column', default=None,
                        help='Column name for longitude (auto-detected if not specified)')
    parser.add_argument('--radius', type=float, default=500,
                        help='Search radius in meters (default: 500)')
    parser.add_argument('--sheet', default=0,
                        help='Sheet name or index (default: first sheet)')
    parser.add_argument('--rate_limit', type=float, default=1.0,
                        help='Seconds between API calls (default: 1.0)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        df = process_excel(
            input_path=args.input,
            output_path=args.output,
            lat_col=args.lat_column,
            lon_col=args.lon_column,
            radius=args.radius,
            sheet_name=args.sheet,
            rate_limit=args.rate_limit
        )
        print(f"\nResults saved to: {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
