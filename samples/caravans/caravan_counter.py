"""
Caravan Counter Module - OpenStreetMap Version

A simple module for counting caravan polygons from OpenStreetMap.
Can be imported and used in Jupyter notebooks or other Python scripts.

Example usage:
    from caravan_counter import CaravanCounter

    # Initialize counter
    counter = CaravanCounter()

    # Count caravans at a location (returns dict with counts)
    result = counter.count_at_location(lat=51.5074, lon=-0.1278, radius=500)
    print(f"Found {result['total_polygons']} caravans")

    # Process Excel file
    results_df = counter.process_excel(
        input_excel='sites.xlsx',
        output_excel='results.xlsx'
    )

Requirements:
    pip install pandas openpyxl requests
"""

import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class CaravanCounter:
    """
    A class for counting caravan polygons from OpenStreetMap.

    Attributes:
        rate_limit: Seconds between API calls (default 1.0)

    Example:
        counter = CaravanCounter()
        result = counter.count_at_location(51.5, -0.1, radius=500)
        print(f"Found {result['total_polygons']} caravans")
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
        # Rate limiting
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
            Dictionary with keys:
                - total_polygons: Total count of all caravan-related polygons
                - static_caravans: Count of building=static_caravan
                - mobile_homes: Count of building=mobile_home
                - caravan_buildings: Count of building=caravan
                - pitches: Count of leisure=pitch
                - error: Error message if query failed, None otherwise
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
          way["leisure"="pitch"]["tents"="no"](around:{radius},{lat},{lon});
        );
        out body;
        """

        result = {
            'total_polygons': 0,
            'static_caravans': 0,
            'mobile_homes': 0,
            'caravan_buildings': 0,
            'pitches': 0,
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
            elif tags.get('leisure') == 'pitch':
                result['pitches'] += 1

        result['total_polygons'] = len(elements)
        return result

    def count_in_bbox(self, min_lat: float, min_lon: float,
                      max_lat: float, max_lon: float) -> Dict:
        """
        Count caravan polygons within a bounding box.

        Args:
            min_lat, min_lon: Southwest corner coordinates
            max_lat, max_lon: Northeast corner coordinates

        Returns:
            Dictionary with count details (same format as count_at_location)
        """
        bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        query = f"""
        [out:json][timeout:30];
        (
          way["building"="static_caravan"]({bbox});
          way["building"="caravan"]({bbox});
          way["building"="mobile_home"]({bbox});
          node["building"="static_caravan"]({bbox});
          node["building"="caravan"]({bbox});
          node["building"="mobile_home"]({bbox});
          way["leisure"="pitch"]["tents"="no"]({bbox});
        );
        out body;
        """

        result = {
            'total_polygons': 0,
            'static_caravans': 0,
            'mobile_homes': 0,
            'caravan_buildings': 0,
            'pitches': 0,
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
            elif tags.get('leisure') == 'pitch':
                result['pitches'] += 1

        result['total_polygons'] = len(elements)
        return result

    def process_excel(self, input_excel: str, output_excel: str = None,
                      lat_column: str = None, lon_column: str = None,
                      radius: float = 500, sheet_name=0) -> pd.DataFrame:
        """
        Process caravan sites from an Excel file.

        Args:
            input_excel: Path to input Excel file
            output_excel: Path to output Excel file (optional)
            lat_column: Name of column containing latitude (auto-detected if None)
            lon_column: Name of column containing longitude (auto-detected if None)
            radius: Default search radius in meters
            sheet_name: Sheet name or index to read (default: first sheet)

        Returns:
            DataFrame with original data plus caravan counts
        """
        df = pd.read_excel(input_excel, sheet_name=sheet_name)
        print(f"Read {len(df)} rows from {input_excel}")

        # Auto-detect columns
        lat_col = self._find_column(df, ['latitude', 'lat', 'Latitude', 'Lat', 'y', 'Y'])
        lon_col = self._find_column(df, ['longitude', 'lon', 'lng', 'Longitude', 'Lon', 'x', 'X'])

        if lat_column:
            lat_col = lat_column
        if lon_column:
            lon_col = lon_column

        if not lat_col or not lon_col:
            raise ValueError(f"Could not find lat/lon columns. Available: {list(df.columns)}")

        print(f"Using columns: {lat_col}, {lon_col}")

        # Find name column
        name_col = self._find_column(df, ['site_name', 'name', 'site', 'Name', 'Site'])

        # Add result columns
        df['caravan_count'] = 0
        df['static_caravans'] = 0
        df['mobile_homes'] = 0
        df['other_caravans'] = 0
        df['pitches'] = 0
        df['query_status'] = ''

        for idx, row in df.iterrows():
            site_name = row[name_col] if name_col else f"Site {idx + 1}"
            lat = float(row[lat_col])
            lon = float(row[lon_col])

            print(f"[{idx + 1}/{len(df)}] {site_name} ({lat:.5f}, {lon:.5f})", end=' ')

            result = self.count_at_location(lat, lon, radius)

            if result['error']:
                df.at[idx, 'query_status'] = f"Error: {result['error']}"
                print(f"-> Error")
            else:
                df.at[idx, 'caravan_count'] = result['total_polygons']
                df.at[idx, 'static_caravans'] = result['static_caravans']
                df.at[idx, 'mobile_homes'] = result['mobile_homes']
                df.at[idx, 'other_caravans'] = result['caravan_buildings']
                df.at[idx, 'pitches'] = result['pitches']
                df.at[idx, 'query_status'] = 'Success'
                print(f"-> {result['total_polygons']} caravans")

        if output_excel:
            df.to_excel(output_excel, index=False)
            print(f"\nSaved to: {output_excel}")

        print(f"\nTotal caravans found: {df['caravan_count'].sum()}")
        return df

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column name from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
            for df_col in df.columns:
                if df_col.lower() == col.lower():
                    return df_col
        return None


# Convenience function
def count_caravans(lat: float, lon: float, radius: float = 500) -> int:
    """
    Quick function to count caravans at a location.

    Args:
        lat: Latitude
        lon: Longitude
        radius: Search radius in meters

    Returns:
        Number of caravan polygons found
    """
    counter = CaravanCounter(rate_limit=0)
    result = counter.count_at_location(lat, lon, radius)
    return result['total_polygons']
