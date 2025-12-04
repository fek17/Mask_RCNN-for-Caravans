"""
Download UK Caravan Data from OpenStreetMap

Downloads all caravan/mobile home polygons for the UK from OpenStreetMap
and saves them as a GeoJSON file for local use (no more API rate limits).

Usage:
    python download_uk_caravans.py --output uk_caravans.geojson

This only needs to be run once. The resulting GeoJSON can then be used
with count_caravans_from_images.py --geojson uk_caravans.geojson

Requirements:
    pip install requests
"""

import argparse
import json
import requests
import time
import sys


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# UK bounding box (approximate)
UK_BBOX = "49.5,-8.0,61.0,2.0"  # south,west,north,east


def download_uk_caravans(output_path: str, timeout: int = 300) -> dict:
    """
    Download all caravan polygons in the UK from OpenStreetMap.

    Args:
        output_path: Path to save GeoJSON file
        timeout: Query timeout in seconds (default 5 minutes)

    Returns:
        GeoJSON dict
    """
    print("Downloading all UK caravan data from OpenStreetMap...")
    print("This may take a few minutes...\n")

    # Query for all caravan-related buildings in UK
    query = f"""
    [out:json][timeout:{timeout}];
    (
      // Static caravans
      way["building"="static_caravan"]({UK_BBOX});
      node["building"="static_caravan"]({UK_BBOX});

      // Mobile homes
      way["building"="mobile_home"]({UK_BBOX});
      node["building"="mobile_home"]({UK_BBOX});

      // Generic caravan buildings
      way["building"="caravan"]({UK_BBOX});
      node["building"="caravan"]({UK_BBOX});
    );
    out body;
    >;
    out skel qt;
    """

    print(f"Querying Overpass API...")
    start_time = time.time()

    try:
        response = requests.post(
            OVERPASS_URL,
            data={'data': query},
            timeout=timeout + 30  # Extra buffer for network
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        print("ERROR: Query timed out. The Overpass server may be busy.")
        print("Try again later or increase --timeout")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download data: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"Download completed in {elapsed:.1f} seconds")

    elements = data.get('elements', [])
    print(f"Retrieved {len(elements)} elements from OSM")

    # Build node lookup for way geometries
    node_coords = {}
    for elem in elements:
        if elem['type'] == 'node' and 'lat' in elem and 'lon' in elem:
            node_coords[elem['id']] = (elem['lon'], elem['lat'])  # GeoJSON is lon,lat

    # Convert to GeoJSON
    features = []
    stats = {'static_caravan': 0, 'mobile_home': 0, 'caravan': 0, 'nodes': 0, 'ways': 0}

    for elem in elements:
        tags = elem.get('tags', {})
        building_type = tags.get('building', '')

        if building_type not in ['static_caravan', 'caravan', 'mobile_home']:
            continue

        stats[building_type] = stats.get(building_type, 0) + 1

        feature = {
            'type': 'Feature',
            'properties': {
                'osm_id': elem['id'],
                'osm_type': elem['type'],
                'building': building_type,
                **{k: v for k, v in tags.items() if k != 'building'}
            },
            'geometry': None
        }

        if elem['type'] == 'way':
            # Get coordinates for the way
            coords = []
            for node_id in elem.get('nodes', []):
                if node_id in node_coords:
                    coords.append(node_coords[node_id])

            if len(coords) >= 3:
                # Close the polygon if not already closed
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

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'source': 'OpenStreetMap via Overpass API',
            'region': 'United Kingdom',
            'bbox': UK_BBOX,
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_features': len(features),
            'stats': stats
        }
    }

    # Save to file
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(geojson, f)

    file_size = len(json.dumps(geojson)) / (1024 * 1024)

    print(f"\n{'='*50}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"Total caravan polygons: {len(features)}")
    print(f"  - Static caravans: {stats['static_caravan']}")
    print(f"  - Mobile homes: {stats['mobile_home']}")
    print(f"  - Other caravans: {stats['caravan']}")
    print(f"  - Ways (polygons): {stats['ways']}")
    print(f"  - Nodes (points): {stats['nodes']}")
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
    parser.add_argument('--timeout', '-t', type=int, default=300,
                        help='Query timeout in seconds (default: 300)')

    args = parser.parse_args()

    download_uk_caravans(args.output, args.timeout)

    print(f"\nYou can now use this with:")
    print(f"  python count_caravans_from_images.py --input_dir ./images --output results.xlsx --geojson {args.output}")


if __name__ == '__main__':
    main()
