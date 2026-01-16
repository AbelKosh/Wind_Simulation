#!/usr/bin/env python
"""
Test script to visualize severe weather wind data only.

This script generates a heatmap using ONLY the wind speed data from
severe weather reports (Local Storm Reports), without any grid data
from the Weather Company historical API.

Use this to compare:
1. Storm-only visualization (this script)
2. Grid-only visualization (--no-augment-storm-winds flag)
3. Combined visualization (default behavior)

Usage:
    python test_storm_winds_only.py
"""

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy import ndimage

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path setup - avoid circular import by importing directly
from config.settings import get_settings
from src.data.grid import PRESET_REGIONS, GridBounds, get_state_geometry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# EF rating wind speed mapping (copied from geo_heatmap.py to avoid import issues)
EF_WIND_SPEEDS = {
    'EF0': 75,
    'EF1': 98,
    'EF2': 123,
    'EF3': 150,
    'EF4': 183,
    'EF5': 220,
}


def extract_storm_wind_points_local(storm_data, include_tornadoes=True):
    """
    Extract wind speed data points from storm reports.
    
    Returns a list of (latitude, longitude, wind_speed_mph) tuples.
    """
    wind_points = []
    
    for report in storm_data.reports:
        # Direct wind reports (thunderstorm wind, high wind)
        if report.is_wind_event and not report.is_tornado:
            if report.magnitude and report.magnitude_unit == 'mph':
                wind_points.append((
                    report.latitude,
                    report.longitude,
                    report.magnitude
                ))
        
        # Tornado wind estimates from EF rating
        elif report.is_tornado and include_tornadoes and report.tornado_path:
            ef_rating = report.tornado_path.ef_rating
            if ef_rating and ef_rating in EF_WIND_SPEEDS:
                wind_speed = EF_WIND_SPEEDS[ef_rating]
                
                # Add points along the tornado path
                for point in report.tornado_path.path_points:
                    wind_points.append((
                        point.latitude,
                        point.longitude,
                        wind_speed
                    ))
    
    return wind_points


def create_storm_only_heatmap(
    bounds: GridBounds,
    start_date: date,
    end_date: date,
    output_path: Path,
    resolution: float = 0.1,
    demo_mode: bool = True,
):
    """
    Create a heatmap using ONLY severe weather wind data.
    
    Args:
        bounds: Geographic bounds
        start_date: Start date
        end_date: End date
        output_path: Output file path
        resolution: Grid resolution for interpolation
        demo_mode: Whether to use demo data
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Import here to avoid circular import
    from src.api.weather_client import WeatherClient
    
    # Fetch storm data
    client = WeatherClient(demo_mode=demo_mode)
    storm_data = client.get_local_storm_reports(
        lat_min=bounds.lat_min,
        lat_max=bounds.lat_max,
        lon_min=bounds.lon_min,
        lon_max=bounds.lon_max,
        start_date=start_date,
        end_date=end_date,
        include_tornado_paths=True,
    )
    
    logger.info(f"Fetched {storm_data.count} storm reports, {storm_data.tornado_count} tornadoes")
    
    # Extract wind points using local function
    wind_points = extract_storm_wind_points_local(storm_data, include_tornadoes=True)
    
    if not wind_points:
        logger.error("No wind points extracted from storm data")
        return
    
    logger.info(f"Extracted {len(wind_points)} wind speed points from storm reports")
    
    # Print wind points for reference
    print("\n" + "=" * 70)
    print("STORM WIND DATA POINTS")
    print("=" * 70)
    for lat, lon, speed in sorted(wind_points, key=lambda x: -x[2]):
        print(f"  {speed:6.1f} mph at ({lat:.3f}°N, {lon:.3f}°W)")
    print("=" * 70 + "\n")
    
    # Create arrays
    lats = np.array([p[0] for p in wind_points])
    lons = np.array([p[1] for p in wind_points])
    winds = np.array([p[2] for p in wind_points])
    
    # Create interpolation grid
    grid_lats = np.linspace(bounds.lat_min, bounds.lat_max, 
                            int((bounds.lat_max - bounds.lat_min) / resolution))
    grid_lons = np.linspace(bounds.lon_min, bounds.lon_max,
                            int((bounds.lon_max - bounds.lon_min) / resolution))
    
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    
    # Interpolate using RBF-like approach (inverse distance weighting)
    # Since we have sparse points, we'll use nearest + some smoothing
    points = np.array([lats, lons]).T
    grid_points = np.array([lat_grid.ravel(), lon_grid.ravel()]).T
    
    try:
        # Use linear interpolation with fill
        wind_grid = griddata(points, winds, grid_points, method='linear')
        wind_grid = wind_grid.reshape(lat_grid.shape)
        
        # Fill NaNs with nearest
        nearest = griddata(points, winds, grid_points, method='nearest')
        nearest = nearest.reshape(lat_grid.shape)
        wind_grid = np.where(np.isnan(wind_grid), nearest, wind_grid)
        
        # Apply gaussian smoothing to spread the influence
        wind_grid = ndimage.gaussian_filter(wind_grid, sigma=2.0)
        
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        return
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set extent
    margin = 0.1
    ax.set_extent([
        bounds.lon_min - margin,
        bounds.lon_max + margin,
        bounds.lat_min - margin,
        bounds.lat_max + margin
    ], crs=ccrs.PlateCarree())
    
    # Add basemap
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#666666', linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.STATES, edgecolor='#999999', linewidth=0.3, zorder=1)
    
    # Create meshgrid for pcolormesh
    lon_step = (grid_lons[-1] - grid_lons[0]) / (len(grid_lons) - 1)
    lat_step = (grid_lats[-1] - grid_lats[0]) / (len(grid_lats) - 1)
    
    lon_edges = np.concatenate([
        [grid_lons[0] - lon_step/2],
        (grid_lons[:-1] + grid_lons[1:]) / 2,
        [grid_lons[-1] + lon_step/2]
    ])
    lat_edges = np.concatenate([
        [grid_lats[0] - lat_step/2],
        (grid_lats[:-1] + grid_lats[1:]) / 2,
        [grid_lats[-1] + lat_step/2]
    ])
    
    LON, LAT = np.meshgrid(lon_edges, lat_edges)
    
    # Custom colormap for severe wind speeds (higher range)
    vmin, vmax = 40, 120  # Focus on severe wind speeds
    
    severe_colors = [
        (40, '#FFFF00'),   # Yellow - 40 mph
        (50, '#FFA500'),   # Orange - 50 mph
        (60, '#FF4500'),   # OrangeRed - 60 mph
        (70, '#FF0000'),   # Red - 70 mph
        (80, '#DC143C'),   # Crimson - 80 mph
        (90, '#8B0000'),   # DarkRed - 90 mph
        (100, '#800080'),  # Purple - 100 mph
        (120, '#4B0082'),  # Indigo - 120+ mph
    ]
    
    colors = []
    boundaries = []
    for speed, color in severe_colors:
        if vmin <= speed <= vmax:
            boundaries.append((speed - vmin) / (vmax - vmin))
            colors.append(color)
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'severe_wind', list(zip(boundaries, colors))
    )
    
    # Plot heatmap
    mesh = ax.pcolormesh(
        LON, LAT, wind_grid,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='flat',
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    
    # Add state boundary
    geometry = get_state_geometry("North Carolina")
    if geometry is not None:
        from cartopy.feature import ShapelyFeature
        nc_feature = ShapelyFeature(
            [geometry], ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='black',
            linewidth=2.0,
        )
        ax.add_feature(nc_feature, zorder=10)
    
    # Plot actual storm report locations
    for lat, lon, speed in wind_points:
        ax.scatter(
            lon, lat,
            s=100,
            c='white',
            edgecolors='black',
            linewidths=1.5,
            marker='o',
            transform=ccrs.PlateCarree(),
            zorder=15,
        )
        # Add wind speed label
        ax.annotate(
            f'{speed:.0f}',
            (lon, lat),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            color='black',
            transform=ccrs.PlateCarree(),
            zorder=16,
        )
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Wind Speed (mph)', fontsize=12)
    
    # Title
    ax.set_title(
        f"Storm Wind Reports ONLY - {bounds.name}\n"
        f"{start_date} to {end_date}\n"
        f"({len(wind_points)} storm wind observations)",
        fontsize=14,
        fontweight='bold',
    )
    
    # Add legend for EF scale reference
    legend_text = "EF Wind Estimates:\n"
    for ef, speed in sorted(EF_WIND_SPEEDS.items()):
        legend_text += f"  {ef}: ~{speed} mph\n"
    
    ax.text(
        0.02, 0.02, legend_text.strip(),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        zorder=20,
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info(f"Saved storm-only heatmap to {output_path}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STORM WIND STATISTICS")
    print("=" * 70)
    print(f"  Total storm reports: {storm_data.count}")
    print(f"  Tornado reports: {storm_data.tornado_count}")
    print(f"  Wind speed points: {len(wind_points)}")
    print(f"  Min wind speed: {winds.min():.1f} mph")
    print(f"  Max wind speed: {winds.max():.1f} mph")
    print(f"  Mean wind speed: {winds.mean():.1f} mph")
    print("=" * 70)


def main():
    """Main entry point."""
    print("=" * 70)
    print("STORM WINDS ONLY VISUALIZATION TEST")
    print("=" * 70)
    print("\nThis script generates a heatmap using ONLY severe weather")
    print("wind data (Local Storm Reports), without grid data.\n")
    
    # Use NC bounds and Jan 8-9, 2024 dates
    bounds = PRESET_REGIONS['north_carolina']
    start_date = date(2024, 1, 8)
    end_date = date(2024, 1, 9)
    
    output_dir = project_root / "output" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate storm-only heatmap
    create_storm_only_heatmap(
        bounds=bounds,
        start_date=start_date,
        end_date=end_date,
        output_path=output_dir / "storm_winds_only.png",
        resolution=0.05,  # Higher resolution for smoother interpolation
        demo_mode=True,
    )
    
    print(f"\n✓ Storm-only heatmap saved to: {output_dir / 'storm_winds_only.png'}")
    print("\nTo compare with other visualizations, run:")
    print("  # Combined (grid + storm): python -m src.cli --demo --format png")
    print("  # Grid only: python -m src.cli --demo --format png --no-augment-storm-winds")
    print("\nOpen all three images to compare the visualizations.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
