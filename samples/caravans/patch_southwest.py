"""
Patch South West region into existing UK caravans GeoJSON.

Splits South West into smaller sub-regions to avoid timeout,
then merges with existing GeoJSON file.

Usage:
    python patch_southwest.py --geojson uk_caravans.geojson
"""

import argparse
import json
import requests
import time
import sys

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# South West split into smaller chunks (original was 49.9, -6.5, 51.5, -2.0)
SOUTHWEST_SUBREGIONS = [
    ("SW Cornwall", 49.9, -6.5, 50.5, -4.5),
    ("SW Devon South", 50.0, -4.5, 50.6, -3.2),
    ("SW Devon North", 50.6, -4.5, 51.2, -3.2),
    ("SW Dorset", 50.4, -3.2, 51.0, -2.0),
    ("SW Somerset", 50.8, -3.5, 51.5, -2.5),
    ("SW Bristol Area", 51.0, -3.0, 51.5, -2.0),
]


def download_region(name: str, south: float, west: float, north: float, east: float,
                    timeout: int = 180) -> list:
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
            print(f"    Attempting download...", end=' ', flush=True)
            response = requests.post(
                OVERPASS_URL,
                data={'data': query},
                timeout=timeout + 30
            )
            response.raise_for_status()
            print("OK")
            return response.json().get('elements', [])
        except requests.exceptions.RequestException as e:
            print(f"RETRY")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"      Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"      FAILED: {e}")
                return []

    return []


def elements_to_features(all_elements: list) -> list:
    """Convert OSM elements to GeoJSON features."""
    # Build node lookup
    node_coords = {}
    for elem in all_elements:
        if elem['type'] == 'node' and 'lat' in elem and 'lon' in elem:
            node_coords[elem['id']] = (elem['lon'], elem['lat'])

    features = []
    seen_ids = set()

    for elem in all_elements:
        tags = elem.get('tags', {})
        building_type = tags.get('building', '')

        if building_type not in ['static_caravan', 'caravan', 'mobile_home']:
            continue

        unique_id = f"{elem['type']}_{elem['id']}"
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)

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
                features.append(feature)

        elif elem['type'] == 'node' and 'lat' in elem and 'lon' in elem:
            feature['geometry'] = {
                'type': 'Point',
                'coordinates': [elem['lon'], elem['lat']]
            }
            features.append(feature)

    return features


def main():
    parser = argparse.ArgumentParser(
        description='Patch South West region into existing UK caravans GeoJSON'
    )
    parser.add_argument('--geojson', '-g', default='uk_caravans.geojson',
                        help='Existing GeoJSON file to patch (default: uk_caravans.geojson)')
    parser.add_argument('--timeout', '-t', type=int, default=180,
                        help='Timeout per sub-region in seconds (default: 180)')

    args = parser.parse_args()

    # Load existing GeoJSON
    print(f"Loading existing GeoJSON: {args.geojson}")
    try:
        with open(args.geojson, 'r') as f:
            geojson = json.load(f)
        original_count = len(geojson['features'])
        print(f"  Loaded {original_count} existing features")
    except FileNotFoundError:
        print(f"Error: File not found: {args.geojson}")
        sys.exit(1)

    # Get existing OSM IDs to avoid duplicates
    existing_ids = set()
    for feature in geojson['features']:
        osm_id = feature.get('properties', {}).get('osm_id')
        osm_type = feature.get('properties', {}).get('osm_type', 'way')
        if osm_id:
            existing_ids.add(f"{osm_type}_{osm_id}")

    print(f"\nDownloading South West region in {len(SOUTHWEST_SUBREGIONS)} sub-regions...")

    all_elements = []
    for i, (name, south, west, north, east) in enumerate(SOUTHWEST_SUBREGIONS):
        print(f"  [{i+1}/{len(SOUTHWEST_SUBREGIONS)}] {name}...")
        elements = download_region(name, south, west, north, east, args.timeout)
        print(f"      Got {len(elements)} elements")
        all_elements.extend(elements)

        # Rate limiting
        if i < len(SOUTHWEST_SUBREGIONS) - 1:
            time.sleep(3)

    print(f"\nTotal raw elements: {len(all_elements)}")

    # Convert to features
    print("Converting to GeoJSON features...")
    new_features = elements_to_features(all_elements)
    print(f"  Converted to {len(new_features)} features")

    # Filter out duplicates
    unique_new = []
    for feature in new_features:
        osm_id = feature.get('properties', {}).get('osm_id')
        osm_type = feature.get('properties', {}).get('osm_type', 'way')
        unique_key = f"{osm_type}_{osm_id}"

        if unique_key not in existing_ids:
            unique_new.append(feature)
            existing_ids.add(unique_key)

    print(f"  {len(unique_new)} new unique features (not in existing file)")

    # Merge
    geojson['features'].extend(unique_new)
    final_count = len(geojson['features'])

    # Update metadata
    if 'metadata' in geojson:
        geojson['metadata']['total_features'] = final_count
        geojson['metadata']['patched'] = f"Added {len(unique_new)} South West features"

    # Save
    print(f"\nSaving patched GeoJSON...")
    with open(args.geojson, 'w') as f:
        json.dump(geojson, f)

    file_size = len(json.dumps(geojson)) / (1024 * 1024)

    print(f"\n{'='*50}")
    print("PATCH COMPLETE")
    print(f"{'='*50}")
    print(f"Original features: {original_count}")
    print(f"New SW features: {len(unique_new)}")
    print(f"Total features: {final_count}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Saved to: {args.geojson}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
