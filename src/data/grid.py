"""
Grid data structures for geographic wind data.

This module provides data structures and utilities for managing
gridded wind data across geographic regions.
"""

import logging
import os
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from urllib.request import urlretrieve

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Shapefile cache directory
SHAPEFILE_DIR = Path(__file__).parent.parent.parent / "data" / "shapefiles"

# Census TIGER state shapefile URL (2023)
TIGER_STATE_URL = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"


@dataclass
class GridBounds:
    """
    Geographic bounding box for a region.
    
    Attributes:
        lat_min: Minimum latitude (southern boundary)
        lat_max: Maximum latitude (northern boundary)
        lon_min: Minimum longitude (western boundary)
        lon_max: Maximum longitude (eastern boundary)
        name: Optional name for the region
    """
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate bounds."""
        if self.lat_min >= self.lat_max:
            raise ValueError("lat_min must be less than lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError("lon_min must be less than lon_max")
        if not -90 <= self.lat_min <= 90:
            raise ValueError(f"lat_min must be between -90 and 90: {self.lat_min}")
        if not -90 <= self.lat_max <= 90:
            raise ValueError(f"lat_max must be between -90 and 90: {self.lat_max}")
        if not -180 <= self.lon_min <= 180:
            raise ValueError(f"lon_min must be between -180 and 180: {self.lon_min}")
        if not -180 <= self.lon_max <= 180:
            raise ValueError(f"lon_max must be between -180 and 180: {self.lon_max}")
    
    @property
    def lat_range(self) -> float:
        """Get latitude range in degrees."""
        return self.lat_max - self.lat_min
    
    @property
    def lon_range(self) -> float:
        """Get longitude range in degrees."""
        return self.lon_max - self.lon_min
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (lat, lon)."""
        return (
            (self.lat_min + self.lat_max) / 2,
            (self.lon_min + self.lon_max) / 2
        )
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within bounds."""
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "name": self.name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridBounds":
        """Create from dictionary."""
        return cls(
            lat_min=data["lat_min"],
            lat_max=data["lat_max"],
            lon_min=data["lon_min"],
            lon_max=data["lon_max"],
            name=data.get("name"),
        )


# Preset regions for common use cases
PRESET_REGIONS: Dict[str, GridBounds] = {
    "north_carolina": GridBounds(
        lat_min=33.8,
        lat_max=36.6,
        lon_min=-84.3,
        lon_max=-75.4,
        name="North Carolina"
    ),
    "outer_banks": GridBounds(
        lat_min=34.5,
        lat_max=36.5,
        lon_min=-76.5,
        lon_max=-75.3,
        name="Outer Banks, NC"
    ),
    "florida": GridBounds(
        lat_min=24.5,
        lat_max=31.0,
        lon_min=-87.6,
        lon_max=-80.0,
        name="Florida"
    ),
    "texas_coast": GridBounds(
        lat_min=26.0,
        lat_max=30.0,
        lon_min=-97.5,
        lon_max=-93.5,
        name="Texas Coast"
    ),
    "northeast_us": GridBounds(
        lat_min=38.5,
        lat_max=45.0,
        lon_min=-80.5,
        lon_max=-66.9,
        name="Northeast US"
    ),
}


def create_grid(
    bounds: GridBounds,
    resolution: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a grid of lat/lon points within bounds.
    
    Args:
        bounds: Geographic bounding box
        resolution: Grid spacing in degrees (smaller = higher resolution)
        
    Returns:
        Tuple of (latitudes, longitudes) 1D arrays defining the grid
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive")
    
    # Generate latitude array (south to north)
    lats = np.arange(bounds.lat_min, bounds.lat_max + resolution/2, resolution)
    
    # Generate longitude array (west to east)
    lons = np.arange(bounds.lon_min, bounds.lon_max + resolution/2, resolution)
    
    return lats, lons


def create_grid_points(
    bounds: GridBounds,
    resolution: float = 0.25,
) -> List[Tuple[float, float]]:
    """
    Create a list of (lat, lon) grid points within bounds.
    
    Args:
        bounds: Geographic bounding box
        resolution: Grid spacing in degrees
        
    Returns:
        List of (latitude, longitude) tuples
    """
    lats, lons = create_grid(bounds, resolution)
    
    points = []
    for lat in lats:
        for lon in lons:
            points.append((float(lat), float(lon)))
    
    return points


def estimate_grid_size(bounds: GridBounds, resolution: float) -> Dict[str, int]:
    """
    Estimate the number of grid points for given bounds and resolution.
    
    Args:
        bounds: Geographic bounding box
        resolution: Grid spacing in degrees
        
    Returns:
        Dictionary with lat_points, lon_points, and total
    """
    lat_points = int(np.ceil(bounds.lat_range / resolution)) + 1
    lon_points = int(np.ceil(bounds.lon_range / resolution)) + 1
    
    return {
        "lat_points": lat_points,
        "lon_points": lon_points,
        "total": lat_points * lon_points,
    }


class GridPointData(BaseModel):
    """Data for a single grid point."""
    latitude: float
    longitude: float
    peak_wind_speed: Optional[float] = None
    avg_wind_speed: Optional[float] = None
    peak_gust: Optional[float] = None
    peak_direction: Optional[float] = None
    observation_count: int = 0
    error: Optional[str] = None
    
    @property
    def has_data(self) -> bool:
        """Check if point has valid data."""
        return self.peak_wind_speed is not None and self.observation_count > 0


class GridData(BaseModel):
    """
    Container for gridded wind data.
    
    Stores wind data for a grid of geographic points.
    """
    bounds: Dict[str, Any] = Field(..., description="Grid bounds dictionary")
    resolution: float = Field(..., description="Grid resolution in degrees")
    target_date: str = Field(..., description="Target date (YYYY-MM-DD)")
    points: List[GridPointData] = Field(default_factory=list)
    fetch_timestamp: Optional[datetime] = None
    units: str = "mph"
    source: str = "api"
    
    @property
    def grid_bounds(self) -> GridBounds:
        """Get bounds as GridBounds object."""
        return GridBounds.from_dict(self.bounds)
    
    @property
    def lat_array(self) -> np.ndarray:
        """Get sorted unique latitudes."""
        lats = sorted(set(p.latitude for p in self.points))
        return np.array(lats)
    
    @property
    def lon_array(self) -> np.ndarray:
        """Get sorted unique longitudes."""
        lons = sorted(set(p.longitude for p in self.points))
        return np.array(lons)
    
    def to_2d_array(self, field: str = "peak_wind_speed") -> np.ndarray:
        """
        Convert point data to 2D array for visualization.
        
        Args:
            field: Which field to extract ('peak_wind_speed', 'avg_wind_speed', etc.)
            
        Returns:
            2D numpy array with shape (n_lats, n_lons)
        """
        lats = self.lat_array
        lons = self.lon_array
        
        # Create empty array
        data = np.full((len(lats), len(lons)), np.nan)
        
        # Create lookup for indices
        lat_idx = {lat: i for i, lat in enumerate(lats)}
        lon_idx = {lon: i for i, lon in enumerate(lons)}
        
        # Fill array with data
        for point in self.points:
            if point.has_data:
                i = lat_idx.get(point.latitude)
                j = lon_idx.get(point.longitude)
                if i is not None and j is not None:
                    value = getattr(point, field, None)
                    if value is not None:
                        data[i, j] = value
        
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the grid."""
        valid_points = [p for p in self.points if p.has_data]
        
        if not valid_points:
            return {
                "total_points": len(self.points),
                "valid_points": 0,
                "coverage_pct": 0,
            }
        
        peak_winds = [p.peak_wind_speed for p in valid_points if p.peak_wind_speed]
        
        # Find max point
        max_point = max(valid_points, key=lambda p: p.peak_wind_speed or 0)
        
        return {
            "total_points": len(self.points),
            "valid_points": len(valid_points),
            "coverage_pct": 100 * len(valid_points) / len(self.points),
            "max_wind_speed": max(peak_winds) if peak_winds else None,
            "min_wind_speed": min(peak_winds) if peak_winds else None,
            "avg_wind_speed": np.mean(peak_winds) if peak_winds else None,
            "max_location": (max_point.latitude, max_point.longitude),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bounds": self.bounds,
            "resolution": self.resolution,
            "target_date": self.target_date,
            "points": [p.model_dump() for p in self.points],
            "fetch_timestamp": self.fetch_timestamp.isoformat() if self.fetch_timestamp else None,
            "units": self.units,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridData":
        """Create from dictionary."""
        points = [GridPointData(**p) for p in data.get("points", [])]
        fetch_ts = data.get("fetch_timestamp")
        if fetch_ts and isinstance(fetch_ts, str):
            fetch_ts = datetime.fromisoformat(fetch_ts)
        
        return cls(
            bounds=data["bounds"],
            resolution=data["resolution"],
            target_date=data["target_date"],
            points=points,
            fetch_timestamp=fetch_ts,
            units=data.get("units", "mph"),
            source=data.get("source", "api"),
        )


# =============================================================================
# State Boundary Loading (Census TIGER Shapefiles)
# =============================================================================

# Cache for loaded state geometries
_state_geometry_cache: Dict[str, Any] = {}


def _download_tiger_shapefile() -> Path:
    """
    Download Census TIGER state shapefile if not already cached.
    
    Returns:
        Path to the extracted shapefile directory
    """
    SHAPEFILE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SHAPEFILE_DIR / "tl_2023_us_state.zip"
    shp_path = SHAPEFILE_DIR / "tl_2023_us_state.shp"
    
    # Check if already extracted
    if shp_path.exists():
        logger.debug(f"Using cached shapefile: {shp_path}")
        return shp_path
    
    # Download if zip doesn't exist
    if not zip_path.exists():
        logger.info(f"Downloading TIGER state shapefile from Census Bureau...")
        try:
            urlretrieve(TIGER_STATE_URL, zip_path)
            logger.info(f"Downloaded to {zip_path}")
        except Exception as e:
            logger.error(f"Failed to download shapefile: {e}")
            raise RuntimeError(f"Could not download TIGER shapefile: {e}")
    
    # Extract
    logger.info("Extracting shapefile...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(SHAPEFILE_DIR)
    
    if not shp_path.exists():
        raise RuntimeError(f"Shapefile not found after extraction: {shp_path}")
    
    return shp_path


def get_state_geometry(state_name: str = "North Carolina"):
    """
    Get the geometry for a US state from Census TIGER data.
    
    Args:
        state_name: Full name of the state (e.g., "North Carolina")
        
    Returns:
        Shapely geometry object for the state, or None if not found
    """
    # Check cache first
    if state_name in _state_geometry_cache:
        return _state_geometry_cache[state_name]
    
    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not installed - cannot load state boundaries")
        return None
    
    try:
        shp_path = _download_tiger_shapefile()
        gdf = gpd.read_file(shp_path)
        
        # Find the state by name
        state_row = gdf[gdf['NAME'] == state_name]
        
        if state_row.empty:
            logger.warning(f"State '{state_name}' not found in shapefile")
            return None
        
        geometry = state_row.geometry.iloc[0]
        
        # Cache for future use
        _state_geometry_cache[state_name] = geometry
        logger.info(f"Loaded geometry for {state_name}")
        
        return geometry
        
    except Exception as e:
        logger.error(f"Error loading state geometry: {e}")
        return None


def get_state_boundary_coords(state_name: str = "North Carolina") -> Optional[List[Tuple[float, float]]]:
    """
    Get state boundary as list of (lon, lat) coordinate tuples.
    
    This is a convenience function for matplotlib plotting.
    
    Args:
        state_name: Full name of the state
        
    Returns:
        List of (longitude, latitude) tuples forming the boundary polygon,
        or None if geometry cannot be loaded
    """
    geometry = get_state_geometry(state_name)
    
    if geometry is None:
        return None
    
    try:
        from shapely.geometry import Polygon, MultiPolygon
        
        # Handle MultiPolygon (e.g., states with islands)
        if isinstance(geometry, MultiPolygon):
            # Get the largest polygon (mainland)
            largest = max(geometry.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
        elif isinstance(geometry, Polygon):
            coords = list(geometry.exterior.coords)
        else:
            logger.warning(f"Unexpected geometry type: {type(geometry)}")
            return None
        
        # Coords are already (lon, lat) from shapefile
        return [(lon, lat) for lon, lat in coords]
        
    except Exception as e:
        logger.error(f"Error extracting boundary coordinates: {e}")
        return None


# Legacy fallback - simplified NC boundary for when shapefiles aren't available
NC_BOUNDARY_FALLBACK = [
    (-84.32, 35.22), (-84.29, 35.23), (-84.09, 35.00), (-83.96, 34.99),
    (-83.62, 34.99), (-83.11, 35.00), (-82.78, 35.09), (-82.60, 35.15),
    (-82.38, 35.21), (-82.27, 35.20), (-81.97, 35.09), (-81.90, 35.14),
    (-81.73, 35.18), (-81.35, 35.16), (-81.04, 35.15), (-80.93, 35.11),
    (-80.78, 34.96), (-80.78, 34.82), (-79.67, 34.81), (-78.55, 33.87),
    (-78.07, 33.89), (-77.95, 33.94), (-77.93, 34.12), (-77.69, 34.29),
    (-77.44, 34.44), (-77.31, 34.56), (-77.06, 34.68), (-76.75, 34.74),
    (-76.49, 34.82), (-76.35, 34.90), (-76.26, 34.91), (-76.08, 35.00),
    (-75.87, 35.16), (-75.79, 35.19), (-75.75, 35.19), (-75.72, 35.27),
    (-75.72, 35.41), (-75.74, 35.55), (-75.94, 35.79), (-75.98, 35.89),
    (-76.02, 35.96), (-76.07, 36.03), (-76.15, 36.10), (-76.13, 36.15),
    (-76.03, 36.19), (-75.95, 36.19), (-75.87, 36.29), (-75.79, 36.41),
    (-75.79, 36.55), (-76.92, 36.55), (-78.51, 36.54), (-80.03, 36.54),
    (-80.44, 36.55), (-80.61, 36.56), (-80.84, 36.56), (-81.65, 36.59),
    (-81.93, 36.59), (-82.03, 36.59), (-82.08, 36.59), (-82.22, 36.59),
    (-82.41, 36.59), (-82.60, 36.60), (-82.84, 36.60), (-83.07, 36.59),
    (-83.28, 36.60), (-83.53, 36.60), (-83.67, 36.60), (-83.91, 36.59),
    (-84.07, 36.59), (-84.22, 36.59), (-84.32, 36.59), (-84.32, 36.29),
    (-84.32, 35.99), (-84.32, 35.69), (-84.32, 35.39), (-84.32, 35.22),
]


def get_nc_boundary() -> List[Tuple[float, float]]:
    """
    Get North Carolina state boundary coordinates.
    
    Attempts to load from TIGER shapefile first, falls back to
    simplified boundary if shapefile not available.
    
    Returns:
        List of (longitude, latitude) tuples
    """
    coords = get_state_boundary_coords("North Carolina")
    if coords is not None:
        return coords
    
    logger.warning("Using fallback NC boundary (simplified)")
    return NC_BOUNDARY_FALLBACK


# Backward compatibility alias
NC_BOUNDARY = NC_BOUNDARY_FALLBACK
