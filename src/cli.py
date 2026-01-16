#!/usr/bin/env python
"""
Command-line interface for Wind Simulation heatmap generation.

This module provides a CLI for generating geographic wind speed heatmaps
from Weather Company API data.

Usage:
    python -m src.cli --start-date 2024-01-07 --end-date 2024-01-09
    python -m src.cli --lat-min 35.0 --lat-max 36.5 --lon-min -80 --lon-max -75 --resolution 0.1
    python -m src.cli --demo --format all
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from config.settings import get_settings
from src.data.grid import PRESET_REGIONS, GridBounds, estimate_grid_size
from src.data.grid_fetcher import fetch_grid_data_with_progress
from src.api.weather_client import WeatherClient
from src.visualization.geo_heatmap import (
    save_geo_heatmap_static,
    save_geo_heatmap_interactive,
    export_grid_to_csv,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_str}'. Use YYYY-MM-DD format."
        )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate geographic wind speed heatmaps from Weather Company API data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate heatmap for North Carolina with demo data
  python -m src.cli --demo

  # Custom date range
  python -m src.cli --start-date 2024-01-07 --end-date 2024-01-09 --demo

  # Custom geographic bounds
  python -m src.cli --lat-min 35.0 --lat-max 36.5 --lon-min -80 --lon-max -75 --demo

  # High resolution for detailed analysis
  python -m src.cli --resolution 0.05 --demo

  # Output only CSV data
  python -m src.cli --format csv --demo
        """
    )
    
    # Date range arguments
    date_group = parser.add_argument_group('Date Range')
    date_group.add_argument(
        '--start-date',
        type=parse_date,
        default=None,
        help='Start date for wind data (YYYY-MM-DD). Default: 3 days ago'
    )
    date_group.add_argument(
        '--end-date',
        type=parse_date,
        default=None,
        help='End date for wind data (YYYY-MM-DD). Default: yesterday'
    )
    
    # Geographic bounds arguments
    geo_group = parser.add_argument_group('Geographic Bounds')
    geo_group.add_argument(
        '--lat-min',
        type=float,
        default=None,
        help='Minimum latitude (southern boundary)'
    )
    geo_group.add_argument(
        '--lat-max',
        type=float,
        default=None,
        help='Maximum latitude (northern boundary)'
    )
    geo_group.add_argument(
        '--lon-min',
        type=float,
        default=None,
        help='Minimum longitude (western boundary)'
    )
    geo_group.add_argument(
        '--lon-max',
        type=float,
        default=None,
        help='Maximum longitude (eastern boundary)'
    )
    geo_group.add_argument(
        '--region',
        type=str,
        choices=list(PRESET_REGIONS.keys()),
        default='north_carolina',
        help='Use a preset region. Ignored if custom bounds are provided.'
    )
    
    # Resolution and output arguments
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--resolution',
        type=float,
        default=None,
        help='Grid resolution in degrees (smaller = higher detail). Default: 0.05'
    )
    output_group.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for generated files. Default: output/<region>'
    )
    output_group.add_argument(
        '--format',
        type=str,
        choices=['png', 'html', 'csv', 'all'],
        default='all',
        help='Output format(s) to generate. Default: all'
    )
    output_group.add_argument(
        '--smoothing',
        type=int,
        default=4,
        help='Smoothing factor for heatmap interpolation (1=none, 4=default)'
    )
    output_group.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Heatmap transparency (0-1, lower = more transparent). Default: 0.5'
    )
    
    # Mode arguments
    mode_group = parser.add_argument_group('Mode')
    mode_group.add_argument(
        '--demo',
        action='store_true',
        help='Use demo mode with synthetic data (no API key required)'
    )
    mode_group.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of API responses'
    )
    mode_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Severe weather arguments
    severe_group = parser.add_argument_group('Severe Weather Overlay')
    severe_group.add_argument(
        '--severe-weather',
        action='store_true',
        default=True,
        help='Overlay Local Storm Reports on heatmap (default: enabled)'
    )
    severe_group.add_argument(
        '--no-severe-weather',
        action='store_true',
        help='Disable severe weather overlay'
    )
    severe_group.add_argument(
        '--tornado-paths',
        action='store_true',
        default=True,
        help='Show tornado path polygons (default: enabled)'
    )
    severe_group.add_argument(
        '--no-tornado-paths',
        action='store_true',
        help='Disable tornado path polygons'
    )
    severe_group.add_argument(
        '--augment-storm-winds',
        action='store_true',
        default=True,
        help='Incorporate storm wind speeds into heatmap interpolation (default: enabled)'
    )
    severe_group.add_argument(
        '--no-augment-storm-winds',
        action='store_true',
        help='Disable storm wind augmentation - only use grid data for heatmap'
    )
    severe_group.add_argument(
        '--event-types',
        type=str,
        default=None,
        help='Comma-separated list of event types to show (tornado,thunderstorm_wind,hail,high_wind). Default: all'
    )
    
    return parser


def run_heatmap(
    start_date: date,
    end_date: date,
    bounds: GridBounds,
    resolution: float,
    output_dir: Path,
    output_format: str = 'all',
    demo_mode: bool = False,
    use_cache: bool = True,
    smoothing_factor: int = 4,
    alpha: float = 0.5,
    include_severe_weather: bool = True,
    show_tornado_paths: bool = True,
    event_types: Optional[list] = None,
    augment_with_storm_winds: bool = True,
) -> dict:
    """
    Generate wind speed heatmap(s) for the specified parameters.
    
    Args:
        start_date: Start date for wind data
        end_date: End date for wind data
        bounds: Geographic bounding box
        resolution: Grid resolution in degrees
        output_dir: Directory for output files
        output_format: Output format ('png', 'html', 'csv', or 'all')
        demo_mode: Whether to use demo data
        use_cache: Whether to cache API responses
        smoothing_factor: Interpolation upsampling factor
        alpha: Heatmap transparency
        include_severe_weather: Whether to fetch and overlay Local Storm Reports
        show_tornado_paths: Whether to draw tornado path polygons
        event_types: List of event types to include (None = all)
        augment_with_storm_winds: Whether to incorporate storm wind data into heatmap
        
    Returns:
        Dictionary with paths to generated files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Estimate grid size
    grid_size = estimate_grid_size(bounds, resolution)
    logger.info(f"Grid size: {grid_size['lat_points']} lat × {grid_size['lon_points']} lon = "
                f"{grid_size['total']} points")
    
    # Initialize client
    client = WeatherClient(demo_mode=demo_mode)
    mode_str = "DEMO MODE" if demo_mode else "API MODE"
    logger.info(f"Using {mode_str}")
    
    # Fetch grid data
    logger.info("Fetching wind data for grid points...")
    grid_data = fetch_grid_data_with_progress(
        bounds=bounds,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        client=client,
        use_cache=use_cache,
    )
    
    # Print statistics
    stats = grid_data.get_statistics()
    logger.info(f"Data fetched successfully!")
    logger.info(f"  Valid points: {stats['valid_points']}/{stats['total_points']} "
                f"({stats['coverage_pct']:.1f}%)")
    logger.info(f"  Max wind speed: {stats['max_wind_speed']:.1f} mph at "
                f"({stats['max_location'][0]:.2f}°N, {stats['max_location'][1]:.2f}°W)")
    logger.info(f"  Avg wind speed: {stats['avg_wind_speed']:.1f} mph")
    
    # Fetch severe weather reports if enabled
    storm_data = None
    if include_severe_weather:
        logger.info("Fetching severe weather reports...")
        storm_data = client.get_local_storm_reports(
            lat_min=bounds.lat_min,
            lat_max=bounds.lat_max,
            lon_min=bounds.lon_min,
            lon_max=bounds.lon_max,
            start_date=start_date,
            end_date=end_date,
            event_types=event_types,
            include_tornado_paths=show_tornado_paths,
        )
        logger.info(f"  Storm reports: {storm_data.count} total, "
                   f"{storm_data.tornado_count} tornadoes")
    
    # Generate outputs
    results = {}
    region_name = bounds.name or "custom_region"
    base_name = region_name.lower().replace(' ', '_').replace(',', '')
    
    if output_format in ['png', 'all']:
        logger.info("Generating static heatmap (PNG)...")
        png_path = save_geo_heatmap_static(
            grid_data,
            output_path=output_dir / f"{base_name}_heatmap.png",
            title=f"Peak Wind Speed - {region_name}\n{start_date} to {end_date}",
            dpi=150,
            smoothing_factor=smoothing_factor,
            alpha=alpha,
            storm_data=storm_data,
            show_tornado_paths=show_tornado_paths,
            show_wind_reports=True,
            augment_with_storm_winds=augment_with_storm_winds,
        )
        results['png'] = png_path
        logger.info(f"  Saved: {png_path}")
    
    if output_format in ['html', 'all']:
        logger.info("Generating interactive heatmap (HTML)...")
        html_path = save_geo_heatmap_interactive(
            grid_data,
            output_path=output_dir / f"{base_name}_heatmap.html",
            title=f"{region_name} Wind Speed",
            storm_data=storm_data,
            show_tornado_paths=show_tornado_paths,
            show_wind_reports=True,
            augment_with_storm_winds=augment_with_storm_winds,
        )
        results['html'] = html_path
        logger.info(f"  Saved: {html_path}")
    
    if output_format in ['csv', 'all']:
        logger.info("Exporting wind data (CSV)...")
        csv_path = export_grid_to_csv(
            grid_data,
            output_path=output_dir / f"{base_name}_data.csv",
            include_metadata=True,
        )
        results['csv'] = csv_path
        logger.info(f"  Saved: {csv_path}")
    
    return results


def main(argv: Optional[list] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get settings
    settings = get_settings()
    
    # Determine dates - default to Jan 8-9, 2024 storm event
    if args.start_date is None:
        # Default to January 8, 2024 for the severe weather event
        args.start_date = date(2024, 1, 8)
    if args.end_date is None:
        # Default to January 9, 2024 for the severe weather event
        args.end_date = date(2024, 1, 9)
    
    # Validate date range
    if args.start_date > args.end_date:
        logger.error("Start date must be before or equal to end date")
        return 1
    
    # Determine bounds
    if all(v is not None for v in [args.lat_min, args.lat_max, args.lon_min, args.lon_max]):
        # Custom bounds
        try:
            bounds = GridBounds(
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
                name="Custom Region"
            )
        except ValueError as e:
            logger.error(f"Invalid bounds: {e}")
            return 1
    else:
        # Use preset region
        bounds = PRESET_REGIONS.get(args.region)
        if bounds is None:
            logger.error(f"Unknown region: {args.region}")
            return 1
    
    # Determine resolution
    resolution = args.resolution or settings.default_resolution
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        region_slug = (bounds.name or args.region).lower().replace(' ', '_').replace(',', '')
        output_dir = project_root / "output" / region_slug
    
    # Determine demo mode
    demo_mode = args.demo or settings.demo_mode or not settings.has_api_key
    
    # Determine severe weather settings
    include_severe_weather = args.severe_weather and not args.no_severe_weather
    show_tornado_paths = args.tornado_paths and not args.no_tornado_paths
    augment_storm_winds = args.augment_storm_winds and not args.no_augment_storm_winds
    event_types = args.event_types.split(',') if args.event_types else None
    
    # Print configuration
    print("=" * 70)
    print("Wind Speed Heatmap Generator")
    print("=" * 70)
    print(f"\nRegion: {bounds.name}")
    print(f"Bounds: ({bounds.lat_min}°N to {bounds.lat_max}°N, "
          f"{bounds.lon_min}°W to {bounds.lon_max}°W)")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Resolution: {resolution}°")
    print(f"Output Directory: {output_dir}")
    print(f"Format: {args.format}")
    print(f"Mode: {'Demo' if demo_mode else 'API'}")
    print(f"Severe Weather Overlay: {'Enabled' if include_severe_weather else 'Disabled'}")
    if include_severe_weather:
        print(f"  Tornado Paths: {'Enabled' if show_tornado_paths else 'Disabled'}")
        print(f"  Storm Wind Augmentation: {'Enabled' if augment_storm_winds else 'Disabled'}")
        if event_types:
            print(f"  Event Types: {', '.join(event_types)}")
    print()
    
    try:
        results = run_heatmap(
            start_date=args.start_date,
            end_date=args.end_date,
            bounds=bounds,
            resolution=resolution,
            output_dir=output_dir,
            output_format=args.format,
            demo_mode=demo_mode,
            use_cache=not args.no_cache,
            smoothing_factor=args.smoothing,
            alpha=args.alpha,
            include_severe_weather=include_severe_weather,
            show_tornado_paths=show_tornado_paths,
            event_types=event_types,
            augment_with_storm_winds=augment_storm_winds,
        )
        
        print("\n" + "=" * 70)
        print("COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated files:")
        for fmt, path in results.items():
            print(f"  • {path}")
        
        if 'html' in results:
            print(f"\nTo view the interactive map, open:")
            print(f"  open {results['html']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
