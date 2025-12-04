"""
Download UK Caravan Data from OpenStreetMap (Regional)

Downloads caravan/mobile home polygons for the UK from OpenStreetMap
in regional chunks to avoid timeout errors.

Usage:
    python download_uk_caravans.py --output uk_caravans.geojson

Requirements:
    pip install requests
"""

import argparse
import json
import requests
import time
import sys


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# UK split into smaller regional bounding boxes to avoid timeouts
# Format: (name, south, west, north, east)
UK_REGIONS = [
    ("Scotland North", 57.0, -8.0, 61.0, -1.5),
    ("Scotland South", 54.5, -6.0, 57.0, -1.5),
    ("Northern Ireland", 54.0, -8.5, 55.5, -5.4),
    ("North England", 53.5, -3.5, 55.8, 0.0),
    ("Wales", 51.3, -5.5, 53.5, -2.6),
    ("Midlands", 52.0, -3.0, 53.5, 0.5),
    ("East England", 51.5, -0.5, 53.0, 2.0),
    ("South West", 49.9, -6.5, 51.5, -2.0),
    ("South Central", 50.5, -2.0, 52.0, 0.0),
    ("South East", 50.7, -1.0, 51.8, 1.5),
    ("London Area", 51.2, -0.6, 51.8, 0.4),
]


def download_region(name: str, south: float, west: float, north: float, east: float,
                    timeout: int = 120) -> list:
    """Download caravan data for a single region."""
    bbox = f"{south},{west},{north},{east}"

    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["building"="static_caravan"]({bbox});
      way["building"="mobile_home"]({bbox});
      way["building"="caravan"]({bbox});
      node["building"="static_caravan"]({bbox});
      node["building"="mobile_home"]({bbox});
      node["building"="caravan"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={'data': query},
                timeout=timeout + 30
            )
            response.raise_for_status()
            return response.json().get('elements', [])
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"    Retry {attempt + 1}/{max_retries} in {wait_time}s... ({type(e).__name__})")
                time.sleep(wait_time)
            else:
                print(f"    FAILED: {e}")
                return []

    return []


def elements_to_geojson(all_elements: list) -> dict:
    """Convert OSM elements to GeoJSON format."""
    # Build node lookup
    node_coords = {}
    for elem in all_elements:
        if elem['type'] == 'node' and 'lat' in elem and 'lon' in elem:
            node_coords[elem['id']] = (elem['lon'], elem['lat'])

    # Track seen IDs to avoid duplicates
    seen_ids = set()
    features = []
    stats = {'static_caravan': 0, 'mobile_home': 0, 'caravan': 0, 'ways': 0, 'nodes': 0}

    for elem in all_elements:
        tags = elem.get('tags', {})
        building_type = tags.get('building', '')

        if building_type not in ['static_caravan', 'caravan', 'mobile_home']:
            continue

        # Deduplicate
        unique_id = f"{elem['type']}_{elem['id']}"
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)

        stats[building_type] = stats.get(building_type, 0) + 1

        feature = {
            'type': 'Feature',
            'properties': {
                'osm_id': elem['id'],
                'osm_type': elem['type'],
                'building': building_type,
            },
            'geometry': None
        }

        if elem['type'] == 'way':
            coords = []
            for node_id in elem.get('nodes', []):
                if node_id in node_coords:
                    coords.append(node_coords[node_id])

            if len(coords) >= 3:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

                feature['geometry'] = {
                    'type': 'Polygon',
                    'coordinates': [coords]
                }
                stats['ways'] += 1
                features.append(feature)

        elif elem['type'] == 'node' and 'lat' in elem and 'lon' in elem:
            feature['geometry'] = {
                'type': 'Point',
                'coordinates': [elem['lon'], elem['lat']]
            }
            stats['nodes'] += 1
            features.append(feature)

    return {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'source': 'OpenStreetMap via Overpass API',
            'region': 'United Kingdom',
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_features': len(features),
            'stats': stats
        }
    }, stats


def download_uk_caravans(output_path: str, timeout: int = 120) -> dict:
    """Download all UK caravan data in regional chunks."""
    print("Downloading UK caravan data from OpenStreetMap...")
    print(f"Splitting into {len(UK_REGIONS)} regions to avoid timeouts.\n")

    all_elements = []
    start_time = time.time()

    for i, (name, south, west, north, east) in enumerate(UK_REGIONS):
        print(f"[{i+1}/{len(UK_REGIONS)}] {name}...", end=' ', flush=True)

        elements = download_region(name, south, west, north, east, timeout)
        print(f"got {len(elements)} elements")

        all_elements.extend(elements)

        # Rate limiting between regions
        if i < len(UK_REGIONS) - 1:
            time.sleep(2)

    elapsed = time.time() - start_time
    print(f"\nDownload completed in {elapsed:.1f} seconds")
    print(f"Total raw elements: {len(all_elements)}")

    # Convert to GeoJSON
    print("\nConverting to GeoJSON...")
    geojson, stats = elements_to_geojson(all_elements)

    # Save to file
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(geojson, f)

    file_size = len(json.dumps(geojson)) / (1024 * 1024)

    print(f"\n{'='*50}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"Total caravan polygons: {len(geojson['features'])}")
    print(f"  - Static caravans: {stats['static_caravan']}")
    print(f"  - Mobile homes: {stats['mobile_home']}")
    print(f"  - Other caravans: {stats['caravan']}")
    print(f"  - Polygons (ways): {stats['ways']}")
    print(f"  - Points (nodes): {stats['nodes']}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Saved to: {output_path}")
    print(f"{'='*50}")

    return geojson


def main():
    parser = argparse.ArgumentParser(
        description='Download UK caravan data from OpenStreetMap'
    )
    parser.add_argument('--output', '-o', default='uk_caravans.geojson',
                        help='Output GeoJSON file path (default: uk_caravans.geojson)')
    parser.add_argument('--timeout', '-t', type=int, default=120,
                        help='Timeout per region in seconds (default: 120)')

    args = parser.parse_args()

    download_uk_caravans(args.output, args.timeout)

    print(f"\nYou can now use this with:")
    print(f"  python count_caravans_from_images.py --input_dir ./images --output results.xlsx --geojson {args.output}")


if __name__ == '__main__':
    main()
