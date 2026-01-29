"""
Grid data fetcher for geographic wind data.

This module handles fetching wind data for multiple grid points,
with caching, progress reporting, and parallel processing support.
"""

import logging
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Callable

from src.data.grid import (
    GridBounds,
    GridData,
    GridPointData,
    create_grid_points,
    estimate_grid_size,
)
from src.api.weather_client import WeatherClient
from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_grid_cache_key(
    bounds: GridBounds,
    resolution: float,
    start_date: date,
    end_date: Optional[date] = None,
) -> str:
    """
    Generate a cache key for grid data.
    
    Args:
        bounds: Grid bounds
        resolution: Grid resolution
        start_date: Start date
        end_date: End date (defaults to start_date)
        
    Returns:
        Cache key string
    """
    if end_date is None:
        end_date = start_date
    key_data = f"{bounds.lat_min}_{bounds.lat_max}_{bounds.lon_min}_{bounds.lon_max}_{resolution}_{start_date.isoformat()}_{end_date.isoformat()}"
    return f"grid_{hashlib.md5(key_data.encode()).hexdigest()[:16]}"


def load_grid_cache(cache_key: str, cache_dir: Optional[Path] = None) -> Optional[GridData]:
    """
    Load cached grid data if available.
    
    Args:
        cache_key: Cache key
        cache_dir: Cache directory
        
    Returns:
        GridData if cache hit, None otherwise
    """
    settings = get_settings()
    cache_dir = cache_dir or settings.cache_dir
    cache_path = cache_dir / f"{cache_key}.json"
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        return GridData.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load grid cache: {e}")
        return None


def save_grid_cache(
    grid_data: GridData,
    cache_key: str,
    cache_dir: Optional[Path] = None,
) -> None:
    """
    Save grid data to cache.
    
    Args:
        grid_data: Grid data to cache
        cache_key: Cache key
        cache_dir: Cache directory
    """
    settings = get_settings()
    cache_dir = cache_dir or settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    
    try:
        with open(cache_path, "w") as f:
            json.dump(grid_data.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved grid cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save grid cache: {e}")


def fetch_point_data(
    client: WeatherClient,
    lat: float,
    lon: float,
    start_date: date,
    end_date: Optional[date] = None,
) -> GridPointData:
    """
    Fetch wind data for a single grid point.
    
    Args:
        client: Weather API client
        lat: Latitude
        lon: Longitude
        start_date: Start date
        end_date: End date (defaults to start_date if not provided)
        
    Returns:
        GridPointData with results or error
    """
    if end_date is None:
        end_date = start_date
        
    try:
        response = client.get_historical_wind(
            latitude=lat,
            longitude=lon,
            start_date=start_date,
            end_date=end_date,
        )
        
        if not response or not response.observations:
            return GridPointData(
                latitude=lat,
                longitude=lon,
                error="No data available",
            )
        
        # Calculate statistics from observations
        wind_speeds = [obs.wind_speed for obs in response.observations if obs.is_valid]
        gusts = [obs.wind_gust for obs in response.observations if obs.has_gust]
        
        if not wind_speeds:
            return GridPointData(
                latitude=lat,
                longitude=lon,
                observation_count=len(response.observations),
                error="No valid wind data",
            )
        
        # Find peak wind observation
        peak_obs = max(response.observations, key=lambda o: o.wind_speed if o.is_valid else 0)
        
        return GridPointData(
            latitude=lat,
            longitude=lon,
            peak_wind_speed=max(wind_speeds),
            avg_wind_speed=sum(wind_speeds) / len(wind_speeds),
            peak_gust=max(gusts) if gusts else None,
            peak_direction=peak_obs.wind_direction,
            observation_count=len(wind_speeds),
        )
        
    except Exception as e:
        logger.warning(f"Error fetching ({lat}, {lon}): {e}")
        return GridPointData(
            latitude=lat,
            longitude=lon,
            error=str(e),
        )


def fetch_grid_data(
    bounds: GridBounds,
    start_date: date,
    end_date: Optional[date] = None,
    resolution: float = 0.25,
    client: Optional[WeatherClient] = None,
    use_cache: bool = True,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> GridData:
    """
    Fetch wind data for a grid of points.
    
    Args:
        bounds: Geographic bounds for the grid
        start_date: Start date to fetch data for
        end_date: End date (defaults to start_date for single day)
        resolution: Grid spacing in degrees (default 0.25°)
        client: WeatherClient instance (creates one if not provided)
        use_cache: Whether to use cached grid data
        max_workers: Maximum parallel workers for fetching
        progress_callback: Optional callback(completed, total) for progress
        
    Returns:
        GridData with wind data for all grid points
    """
    if end_date is None:
        end_date = start_date
        
    # Check cache first
    cache_key = get_grid_cache_key(bounds, resolution, start_date, end_date)
    
    if use_cache:
        cached = load_grid_cache(cache_key)
        if cached:
            logger.info(f"Using cached grid data: {cache_key}")
            return cached
    
    # Create client if not provided
    if client is None:
        client = WeatherClient()
    
    # Generate grid points
    points = create_grid_points(bounds, resolution)
    total_points = len(points)
    
    logger.info(f"Fetching data for {total_points} grid points at {resolution}° resolution")
    
    # Estimate time
    grid_size = estimate_grid_size(bounds, resolution)
    logger.info(f"Grid size: {grid_size['lat_points']} lat × {grid_size['lon_points']} lon")
    
    # Fetch data for all points
    results: List[GridPointData] = []
    completed = 0
    
    # Use thread pool for parallel fetching (with rate limiting in client)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_point = {
            executor.submit(fetch_point_data, client, lat, lon, start_date, end_date): (lat, lon)
            for lat, lon in points
        }
        
        for future in as_completed(future_to_point):
            point_data = future.result()
            results.append(point_data)
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total_points)
    
    # Create GridData object
    # Use date range string for target_date field
    if start_date == end_date:
        date_str = start_date.isoformat()
    else:
        date_str = f"{start_date.isoformat()} to {end_date.isoformat()}"
    
    grid_data = GridData(
        bounds=bounds.to_dict(),
        resolution=resolution,
        target_date=date_str,
        points=results,
        fetch_timestamp=datetime.now(),
        units="mph",
        source="weather_company",
    )
    
    # Cache the results
    if use_cache:
        save_grid_cache(grid_data, cache_key)
    
    # Log statistics
    stats = grid_data.get_statistics()
    logger.info(f"Grid fetch complete: {stats['valid_points']}/{stats['total_points']} points ({stats['coverage_pct']:.1f}%)")
    if stats.get('max_wind_speed'):
        logger.info(f"Max wind speed: {stats['max_wind_speed']:.1f} mph at {stats['max_location']}")
    
    return grid_data


def fetch_grid_data_with_progress(
    bounds: GridBounds,
    start_date: date,
    end_date: Optional[date] = None,
    resolution: float = 0.25,
    client: Optional[WeatherClient] = None,
    use_cache: bool = True,
) -> GridData:
    """
    Fetch grid data with console progress bar.
    
    Simple wrapper around fetch_grid_data with a text progress bar.
    
    Args:
        bounds: Geographic bounds for the grid
        start_date: Start date to fetch data for
        end_date: End date (defaults to start_date for single day)
        resolution: Grid spacing in degrees
        client: WeatherClient instance
        use_cache: Whether to use cached grid data
        
    Returns:
        GridData with wind data for all grid points
    """
    total = estimate_grid_size(bounds, resolution)["total"]
    
    def progress(completed: int, total: int):
        pct = 100 * completed / total
        bar_len = 40
        filled = int(bar_len * completed / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\rFetching: [{bar}] {pct:.1f}% ({completed}/{total})", end="", flush=True)
    
    result = fetch_grid_data(
        bounds=bounds,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        client=client,
        use_cache=use_cache,
        progress_callback=progress,
    )
    
    print()  # New line after progress bar
    return result
