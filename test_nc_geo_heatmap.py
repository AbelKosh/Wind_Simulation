#!/usr/bin/env python
"""
Test script: Generate geographic heatmap for North Carolina.

This script tests the map-based heatmap visualization by:
1. Creating a grid of points across North Carolina
2. Fetching wind data for each point (demo mode)
3. Generating static (PNG) and interactive (HTML) map visualizations
4. Exporting data to CSV for external analysis

For more options, use the CLI directly:
    python -m src.cli --help
"""

import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.cli import run_heatmap
from src.data.grid import PRESET_REGIONS


def main():
    """Generate geographic wind heatmap for NC using demo data."""
    print("=" * 70)
    print("Geographic Wind Heatmap Test - North Carolina")
    print("=" * 70)
    
    # Configuration
    start_date = date(2024, 1, 7)
    end_date = date(2024, 1, 9)
    resolution = 0.1  # Use 0.1 for faster test; use 0.05 for full resolution
    
    # Get North Carolina bounds
    nc_bounds = PRESET_REGIONS["north_carolina"]
    
    print(f"\nRegion: {nc_bounds.name}")
    print(f"Bounds: ({nc_bounds.lat_min}°N to {nc_bounds.lat_max}°N, "
          f"{nc_bounds.lon_min}°W to {nc_bounds.lon_max}°W)")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Resolution: {resolution}°")
    
    # Create output directory
    output_dir = project_root / "output" / "nc_geo_heatmap"
    print(f"Output directory: {output_dir}")
    print("\nUsing DEMO MODE (spatially coherent synthetic data)")
    
    try:
        results = run_heatmap(
            start_date=start_date,
            end_date=end_date,
            bounds=nc_bounds,
            resolution=resolution,
            output_dir=output_dir,
            output_format='all',
            demo_mode=True,
            use_cache=True,
            smoothing_factor=4,
            alpha=0.5,
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nGenerated files:")
        for fmt, path in results.items():
            print(f"  • {path}")
        
        if 'html' in results:
            print(f"\nTo view the interactive map, open the HTML file in a browser:")
            print(f"  open {results['html']}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
