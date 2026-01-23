"""
Geographic heatmap visualization for wind data.

This module provides functions for creating map-based heatmaps
showing wind speed intensity across geographic regions.

Uses Cartopy for accurate map projections and basemaps.
Supports both static (PNG) and interactive (HTML) outputs.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

from src.data.grid import (
    GridData, GridBounds, PRESET_REGIONS,
    get_state_geometry, get_nc_boundary
)
from src.data.models import (
    SevereWeatherResponse, StormReport, TornadoPath, StormEventType
)
from src.visualization.styles import (
    FIGURE_SIZES,
    DEFAULT_DPI,
    FONT_SIZES,
    style_context,
)

logger = logging.getLogger(__name__)


# Wind speed color scale (similar to weather maps)
WIND_SPEED_COLORS = [
    (0, '#f0f9e8'),      # 0-5 mph: very light green
    (5, '#ccebc5'),      # 5-10 mph: light green
    (10, '#a8ddb5'),     # 10-15 mph: green
    (15, '#7bccc4'),     # 15-20 mph: teal
    (20, '#4eb3d3'),     # 20-25 mph: light blue
    (25, '#2b8cbe'),     # 25-30 mph: blue
    (30, '#0868ac'),     # 30-40 mph: dark blue
    (40, '#f1b6da'),     # 40-50 mph: pink
    (50, '#e31a1c'),     # 50-60 mph: red
    (60, '#800026'),     # 60+ mph: dark red
]


def create_wind_colormap(vmin: float = 0, vmax: float = 60) -> mcolors.LinearSegmentedColormap:
    """
    Create a custom colormap for wind speeds.
    
    Args:
        vmin: Minimum wind speed
        vmax: Maximum wind speed
        
    Returns:
        Matplotlib colormap
    """
    colors = []
    boundaries = []
    
    for speed, color in WIND_SPEED_COLORS:
        if speed <= vmax:
            boundaries.append(speed / vmax)
            colors.append(color)
    
    if boundaries[-1] < 1.0:
        boundaries.append(1.0)
        colors.append(WIND_SPEED_COLORS[-1][1])
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'wind_speed',
        list(zip(boundaries, colors))
    )
    
    return cmap


# EF rating color mapping
EF_COLORS = {
    'EF0': '#00FF00',  # Green
    'EF1': '#FFFF00',  # Yellow
    'EF2': '#FFA500',  # Orange
    'EF3': '#FF0000',  # Red
    'EF4': '#8B0000',  # Dark red
    'EF5': '#800080',  # Purple
}

# Storm event markers and colors
STORM_MARKERS = {
    'tornado': {'marker': '^', 'color': 'red', 'size': 150, 'label': 'Tornado'},
    'thunderstorm_wind': {'marker': 'o', 'color': 'orange', 'size': 80, 'label': 'T-Storm Wind'},
    'hail': {'marker': 's', 'color': 'cyan', 'size': 60, 'label': 'Hail'},
    'high_wind': {'marker': 'd', 'color': 'blue', 'size': 70, 'label': 'High Wind'},
}

# EF rating to estimated peak wind speed (mph) - using midpoint of range
# EF0: 65-85, EF1: 86-110, EF2: 111-135, EF3: 136-165, EF4: 166-200, EF5: >200
EF_WIND_SPEEDS = {
    'EF0': 75,   # Midpoint of 65-85 mph
    'EF1': 98,   # Midpoint of 86-110 mph
    'EF2': 123,  # Midpoint of 111-135 mph
    'EF3': 150,  # Midpoint of 136-165 mph
    'EF4': 183,  # Midpoint of 166-200 mph
    'EF5': 220,  # Conservative estimate for >200 mph
}


def extract_storm_wind_points(
    storm_data: SevereWeatherResponse,
    include_tornadoes: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Extract wind speed data points from storm reports.
    
    Returns a list of (latitude, longitude, wind_speed_mph) tuples that
    can be used to augment grid data for interpolation.
    
    Args:
        storm_data: SevereWeatherResponse with storm reports
        include_tornadoes: Whether to include estimated tornado winds
        
    Returns:
        List of (lat, lon, wind_speed) tuples
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
                logger.debug(
                    f"Storm wind point: {report.location_name} - "
                    f"{report.magnitude} mph at ({report.latitude:.3f}, {report.longitude:.3f})"
                )
        
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
                
                logger.debug(
                    f"Tornado wind point: {report.location_name} - "
                    f"{ef_rating} (~{wind_speed} mph)"
                )
    
    logger.info(
        f"Extracted {len(wind_points)} storm wind points from "
        f"{len(storm_data.reports)} reports"
    )
    return wind_points


def augment_grid_with_storm_winds(
    grid_data: GridData,
    storm_data: SevereWeatherResponse,
    include_tornadoes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment grid data with storm wind observations for better interpolation.
    
    This function extracts wind speed data from severe weather reports and
    combines it with the regular grid data. For each location, the MAXIMUM
    wind speed value is kept (grid vs storm) - values are never added together.
    
    The additional storm wind points will influence the bicubic interpolation,
    showing higher wind speeds at locations where extreme winds were observed.
    
    Args:
        grid_data: GridData object with regular grid wind data
        storm_data: SevereWeatherResponse with storm reports
        include_tornadoes: Whether to include estimated tornado wind speeds
        
    Returns:
        Tuple of (all_lats, all_lons, all_winds) with max wind at each location
    """
    # Get grid data
    lats = grid_data.lat_array
    lons = grid_data.lon_array
    wind_array = grid_data.to_2d_array('peak_wind_speed')
    
    # Create arrays of all grid points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_lats = lat_grid.ravel()
    grid_lons = lon_grid.ravel()
    grid_winds = wind_array.ravel()
    
    # Filter out NaN values from grid data
    valid_mask = ~np.isnan(grid_winds)
    grid_lats = grid_lats[valid_mask]
    grid_lons = grid_lons[valid_mask]
    grid_winds = grid_winds[valid_mask]
    
    # Extract storm wind points
    storm_points = extract_storm_wind_points(storm_data, include_tornadoes)
    
    if not storm_points:
        logger.info("No storm wind points to augment")
        return grid_lats, grid_lons, grid_winds
    
    # Convert to arrays
    storm_lats = np.array([p[0] for p in storm_points])
    storm_lons = np.array([p[1] for p in storm_points])
    storm_winds = np.array([p[2] for p in storm_points])
    
    # Filter storm points to be within grid bounds
    bounds = grid_data.grid_bounds
    in_bounds = (
        (storm_lats >= bounds.lat_min) & (storm_lats <= bounds.lat_max) &
        (storm_lons >= bounds.lon_min) & (storm_lons <= bounds.lon_max)
    )
    storm_lats = storm_lats[in_bounds]
    storm_lons = storm_lons[in_bounds]
    storm_winds = storm_winds[in_bounds]
    
    if len(storm_winds) == 0:
        logger.info("No storm points within grid bounds")
        return grid_lats, grid_lons, grid_winds
    
    logger.info(
        f"Augmenting grid with {len(storm_lats)} storm wind points "
        f"(max: {storm_winds.max():.0f} mph)"
    )
    
    # Use a dictionary to track max wind at each approximate location
    # Round to grid resolution to group nearby points
    resolution = grid_data.resolution
    location_winds = {}
    
    # First, add all grid points
    for lat, lon, wind in zip(grid_lats, grid_lons, grid_winds):
        # Use exact grid coordinates as keys
        key = (round(lat, 6), round(lon, 6))
        location_winds[key] = wind
    
    # Then, for storm points, snap to nearest grid cell and take max
    grid_lat_set = set(np.round(grid_lats, 6))
    grid_lon_set = set(np.round(grid_lons, 6))
    
    storm_points_added = 0
    storm_points_upgraded = 0
    
    for s_lat, s_lon, s_wind in zip(storm_lats, storm_lons, storm_winds):
        # Find nearest grid point
        nearest_lat = min(grid_lat_set, key=lambda x: abs(x - s_lat))
        nearest_lon = min(grid_lon_set, key=lambda x: abs(x - s_lon))
        
        # Check if storm point is close enough to a grid point (within resolution)
        if abs(nearest_lat - s_lat) <= resolution and abs(nearest_lon - s_lon) <= resolution:
            # Snap to grid - take maximum value
            key = (round(nearest_lat, 6), round(nearest_lon, 6))
            if key in location_winds:
                if s_wind > location_winds[key]:
                    logger.debug(
                        f"Upgrading grid point ({nearest_lat:.3f}, {nearest_lon:.3f}) "
                        f"from {location_winds[key]:.1f} to {s_wind:.1f} mph"
                    )
                    location_winds[key] = s_wind
                    storm_points_upgraded += 1
            else:
                location_winds[key] = s_wind
                storm_points_added += 1
        else:
            # Storm point is not near a grid point - add as new point
            key = (round(s_lat, 6), round(s_lon, 6))
            if key in location_winds:
                location_winds[key] = max(location_winds[key], s_wind)
            else:
                location_winds[key] = s_wind
                storm_points_added += 1
    
    logger.info(
        f"Storm wind integration: {storm_points_upgraded} grid points upgraded, "
        f"{storm_points_added} new points added"
    )
    
    # Convert back to arrays
    all_lats = np.array([k[0] for k in location_winds.keys()])
    all_lons = np.array([k[1] for k in location_winds.keys()])
    all_winds = np.array(list(location_winds.values()))
    
    return all_lats, all_lons, all_winds


def overlay_storm_reports(
    ax,
    storm_data: SevereWeatherResponse,
    show_tornado_paths: bool = True,
    show_wind_reports: bool = True,
    transform=None,
) -> None:
    """
    Overlay storm reports on an existing Cartopy axes.
    
    Args:
        ax: Matplotlib axes with Cartopy projection
        storm_data: SevereWeatherResponse with storm reports
        show_tornado_paths: Whether to draw tornado path polygons
        show_wind_reports: Whether to show wind event markers
        transform: Cartopy transform (default PlateCarree)
    """
    import cartopy.crs as ccrs
    
    if transform is None:
        transform = ccrs.PlateCarree()
    
    # Track which types we've plotted for legend
    plotted_types = set()
    
    # Draw tornado paths first (under markers)
    if show_tornado_paths:
        for report in storm_data.reports:
            if report.is_tornado and report.tornado_path:
                path = report.tornado_path
                if len(path.path_points) >= 2:
                    # Get EF color
                    ef_color = EF_COLORS.get(path.ef_rating, '#FF0000')
                    
                    # Draw path line with width proportional to max width
                    lons = [pt.longitude for pt in path.path_points]
                    lats = [pt.latitude for pt in path.path_points]
                    
                    # Line width based on EF rating
                    ef_widths = {'EF0': 2, 'EF1': 3, 'EF2': 4, 'EF3': 5, 'EF4': 6, 'EF5': 7}
                    line_width = ef_widths.get(path.ef_rating, 3)
                    
                    ax.plot(
                        lons, lats,
                        color=ef_color,
                        linewidth=line_width,
                        solid_capstyle='round',
                        transform=transform,
                        zorder=5,
                        alpha=0.9,
                    )
                    
                    # Add path outline for visibility
                    ax.plot(
                        lons, lats,
                        color='black',
                        linewidth=line_width + 1,
                        solid_capstyle='round',
                        transform=transform,
                        zorder=4,
                        alpha=0.5,
                    )
                    
                    # Mark start/end points
                    ax.scatter(
                        lons[0], lats[0],
                        marker='o', s=50, c=ef_color, edgecolors='black',
                        linewidths=1, transform=transform, zorder=6
                    )
                    ax.scatter(
                        lons[-1], lats[-1],
                        marker='x', s=50, c=ef_color, edgecolors='black',
                        linewidths=2, transform=transform, zorder=6
                    )
    
    # Plot storm event markers
    if show_wind_reports:
        for report in storm_data.reports:
            event_type = report.event_type.lower()
            
            if event_type not in STORM_MARKERS:
                event_type = 'thunderstorm_wind'  # Default
            
            style = STORM_MARKERS[event_type]
            
            # For tornadoes, use EF color
            if report.is_tornado and report.tornado_path:
                color = EF_COLORS.get(report.tornado_path.ef_rating, style['color'])
            else:
                color = style['color']
            
            # Size based on magnitude for wind events
            size = style['size']
            if report.magnitude and event_type in ['thunderstorm_wind', 'high_wind']:
                size = style['size'] * (1 + (report.magnitude - 50) / 50)
                size = max(50, min(200, size))
            
            ax.scatter(
                report.longitude, report.latitude,
                marker=style['marker'],
                s=size,
                c=color,
                edgecolors='black' if report.is_tornado else 'white',
                linewidths=1,
                transform=transform,
                zorder=7,
                alpha=0.9,
            )
            
            plotted_types.add(event_type)
    
    return plotted_types


def overlay_storm_reports_folium(
    m,
    storm_data: SevereWeatherResponse,
    show_tornado_paths: bool = True,
    show_wind_reports: bool = True,
) -> None:
    """
    Overlay storm reports on a Folium map.
    
    Args:
        m: Folium Map object
        storm_data: SevereWeatherResponse with storm reports
        show_tornado_paths: Whether to draw tornado paths
        show_wind_reports: Whether to show wind markers
    """
    import folium
    
    # Draw tornado paths
    if show_tornado_paths:
        for report in storm_data.reports:
            if report.is_tornado and report.tornado_path:
                path = report.tornado_path
                if len(path.path_points) >= 2:
                    ef_color = EF_COLORS.get(path.ef_rating, 'red')
                    
                    coords = [[pt.latitude, pt.longitude] for pt in path.path_points]
                    
                    # Popup with tornado info
                    popup_html = f"""
                    <b>{path.ef_rating or 'Unknown'} Tornado</b><br>
                    Length: {path.length_miles:.1f} mi<br>
                    Max Width: {path.max_width_yards:.0f} yds<br>
                    Fatalities: {path.fatalities}<br>
                    Injuries: {path.injuries}<br>
                    {report.description or ''}
                    """
                    
                    folium.PolyLine(
                        coords,
                        color=ef_color,
                        weight=5,
                        opacity=0.8,
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"{path.ef_rating} Tornado Path"
                    ).add_to(m)
                    
                    # Start marker
                    folium.CircleMarker(
                        coords[0],
                        radius=8,
                        color='black',
                        fill=True,
                        fill_color=ef_color,
                        popup="Touchdown"
                    ).add_to(m)
    
    # Add storm markers
    if show_wind_reports:
        for report in storm_data.reports:
            if report.is_tornado and show_tornado_paths:
                continue  # Already shown as path
            
            # Determine icon
            if report.is_tornado:
                icon = folium.Icon(color='red', icon='warning-sign', prefix='glyphicon')
            elif report.event_type == StormEventType.THUNDERSTORM_WIND:
                icon = folium.Icon(color='orange', icon='cloud', prefix='glyphicon')
            elif report.event_type == StormEventType.HAIL:
                icon = folium.Icon(color='blue', icon='asterisk', prefix='glyphicon')
            else:
                icon = folium.Icon(color='gray', icon='info-sign', prefix='glyphicon')
            
            popup_html = f"""
            <b>{report.event_type.replace('_', ' ').title()}</b><br>
            Location: {report.location_name or 'Unknown'}, {report.state}<br>
            Time: {report.timestamp.strftime('%Y-%m-%d %H:%M')} UTC<br>
            """
            if report.magnitude:
                popup_html += f"Magnitude: {report.magnitude} {report.magnitude_unit}<br>"
            if report.description:
                popup_html += f"<br>{report.description}"
            
            folium.Marker(
                [report.latitude, report.longitude],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{report.event_type}: {report.magnitude or 'N/A'} {report.magnitude_unit or ''}"
            ).add_to(m)


def add_storm_legend(ax, plotted_types: set, ef_ratings: List[str] = None) -> None:
    """
    Add a legend for storm report symbols.
    
    Args:
        ax: Matplotlib axes
        plotted_types: Set of event types that were plotted
        ef_ratings: List of EF ratings to include in legend
    """
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Add tornado EF ratings if applicable
    if 'tornado' in plotted_types and ef_ratings:
        for ef in ef_ratings:
            if ef in EF_COLORS:
                legend_elements.append(
                    Line2D([0], [0], marker='^', color='w', 
                           markerfacecolor=EF_COLORS[ef], markersize=10,
                           markeredgecolor='black', label=f'{ef} Tornado')
                )
    
    # Add other event types
    for event_type in plotted_types:
        if event_type != 'tornado' and event_type in STORM_MARKERS:
            style = STORM_MARKERS[event_type]
            legend_elements.append(
                Line2D([0], [0], marker=style['marker'], color='w',
                       markerfacecolor=style['color'], markersize=8,
                       markeredgecolor='white', label=style['label'])
            )
    
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=9,
            framealpha=0.9,
            title='Storm Reports'
        )


def _smooth_data(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    smoothing_factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth gridded data using bicubic interpolation.
    
    Args:
        data: 2D array of wind data (shape: n_lats x n_lons)
        lats: 1D array of latitudes
        lons: 1D array of longitudes
        smoothing_factor: Upsampling factor (e.g., 4 = 4x resolution)
        
    Returns:
        Tuple of (smoothed_data, new_lats, new_lons)
    """
    if smoothing_factor <= 1:
        return data, lats, lons
    
    # Create higher resolution lat/lon grids
    new_lats = np.linspace(lats.min(), lats.max(), len(lats) * smoothing_factor)
    new_lons = np.linspace(lons.min(), lons.max(), len(lons) * smoothing_factor)
    
    # Create meshgrid of original coordinates
    orig_lon_grid, orig_lat_grid = np.meshgrid(lons, lats)
    new_lon_grid, new_lat_grid = np.meshgrid(new_lons, new_lats)
    
    # Get valid (non-NaN) points
    valid_mask = ~np.isnan(data)
    points = np.array([orig_lat_grid[valid_mask], orig_lon_grid[valid_mask]]).T
    values = data[valid_mask]
    
    if len(values) < 4:
        logger.warning("Not enough valid points for interpolation")
        return data, lats, lons
    
    # Interpolate to new grid
    new_points = np.array([new_lat_grid.ravel(), new_lon_grid.ravel()]).T
    
    try:
        smoothed = griddata(points, values, new_points, method='cubic')
        smoothed = smoothed.reshape(new_lat_grid.shape)
        
        # Fill any remaining NaNs with nearest neighbor
        if np.any(np.isnan(smoothed)):
            nearest = griddata(points, values, new_points, method='nearest')
            nearest = nearest.reshape(new_lat_grid.shape)
            smoothed = np.where(np.isnan(smoothed), nearest, smoothed)
        
        # Apply slight gaussian smoothing for extra polish
        smoothed = ndimage.gaussian_filter(smoothed, sigma=0.5)
        
    except Exception as e:
        logger.warning(f"Interpolation failed, using original data: {e}")
        return data, lats, lons
    
    return smoothed, new_lats, new_lons


def _smooth_data_with_storm_augmentation(
    grid_data: GridData,
    storm_data: Optional[SevereWeatherResponse],
    smoothing_factor: int = 4,
    include_tornado_winds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth gridded data with storm wind augmentation using bicubic interpolation.
    
    This function combines regular grid wind data with storm wind observations
    (from severe weather reports) before performing interpolation. The result
    is a smoother visualization that accurately represents extreme wind events.
    
    Args:
        grid_data: GridData object with regular grid wind data
        storm_data: Optional SevereWeatherResponse with storm reports
        smoothing_factor: Upsampling factor (e.g., 4 = 4x resolution)
        include_tornado_winds: Whether to include estimated tornado wind speeds
        
    Returns:
        Tuple of (smoothed_data, new_lats, new_lons)
    """
    lats = grid_data.lat_array
    lons = grid_data.lon_array
    
    # If no storm data, fall back to regular smoothing
    if storm_data is None or len(storm_data.reports) == 0:
        wind_array = grid_data.to_2d_array('peak_wind_speed')
        return _smooth_data(wind_array, lats, lons, smoothing_factor)
    
    # Get augmented data points (grid + storm winds)
    all_lats, all_lons, all_winds = augment_grid_with_storm_winds(
        grid_data, storm_data, include_tornado_winds
    )
    
    if len(all_winds) < 4:
        logger.warning("Not enough valid points for interpolation")
        wind_array = grid_data.to_2d_array('peak_wind_speed')
        return wind_array, lats, lons
    
    # Create target grid at higher resolution
    if smoothing_factor <= 1:
        new_lats = lats
        new_lons = lons
    else:
        new_lats = np.linspace(lats.min(), lats.max(), len(lats) * smoothing_factor)
        new_lons = np.linspace(lons.min(), lons.max(), len(lons) * smoothing_factor)
    
    new_lon_grid, new_lat_grid = np.meshgrid(new_lons, new_lats)
    
    # Create points array for interpolation
    points = np.array([all_lats, all_lons]).T
    new_points = np.array([new_lat_grid.ravel(), new_lon_grid.ravel()]).T
    
    try:
        # Use cubic interpolation with augmented points
        smoothed = griddata(points, all_winds, new_points, method='cubic')
        smoothed = smoothed.reshape(new_lat_grid.shape)
        
        # Fill any remaining NaNs with nearest neighbor
        if np.any(np.isnan(smoothed)):
            nearest = griddata(points, all_winds, new_points, method='nearest')
            nearest = nearest.reshape(new_lat_grid.shape)
            smoothed = np.where(np.isnan(smoothed), nearest, smoothed)
        
        # Apply slight gaussian smoothing for extra polish
        smoothed = ndimage.gaussian_filter(smoothed, sigma=0.5)
        
        logger.info(
            f"Augmented interpolation complete: "
            f"grid max={grid_data.to_2d_array('peak_wind_speed').max():.1f} mph, "
            f"augmented max={smoothed.max():.1f} mph"
        )
        
    except Exception as e:
        logger.warning(f"Augmented interpolation failed, using regular data: {e}")
        wind_array = grid_data.to_2d_array('peak_wind_speed')
        return _smooth_data(wind_array, lats, lons, smoothing_factor)
    
    return smoothed, new_lats, new_lons


def plot_geo_heatmap_static(
    grid_data: GridData,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_boundary: bool = True,
    show_colorbar: bool = True,
    show_max_marker: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.5,
    smoothing_factor: int = 4,
    use_basemap: bool = True,
    storm_data: Optional[SevereWeatherResponse] = None,
    show_tornado_paths: bool = True,
    show_wind_reports: bool = True,
    augment_with_storm_winds: bool = True,
    show_storm_markers: bool = True,
) -> plt.Figure:
    """
    Create a static geographic heatmap of wind speeds using Cartopy.
    
    Args:
        grid_data: GridData object with wind data
        title: Plot title
        figsize: Figure size (width, height) in inches
        show_boundary: Whether to show state boundary
        show_colorbar: Whether to show colorbar
        show_max_marker: Whether to mark the maximum wind location
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        alpha: Transparency of heatmap overlay (0-1, lower = more transparent)
        smoothing_factor: Interpolation upsampling factor (1 = no smoothing)
        use_basemap: Whether to use map tile background
        storm_data: Optional SevereWeatherResponse with storm reports to overlay
        show_tornado_paths: Whether to draw tornado path polygons
        show_wind_reports: Whether to show wind event markers
        augment_with_storm_winds: Whether to incorporate storm wind data into heatmap
        show_storm_markers: Whether to display storm report markers on the map
        
    Returns:
        Matplotlib Figure object
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    with style_context():
        # Get data arrays
        lats = grid_data.lat_array
        lons = grid_data.lon_array
        wind_data = grid_data.to_2d_array('peak_wind_speed')
        
        # Validate data
        valid_data = wind_data[~np.isnan(wind_data)]
        if len(valid_data) == 0:
            logger.warning("No valid data for geo heatmap")
            fig, ax = plt.subplots(figsize=figsize or (12, 8))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return fig
        
        # Apply smoothing (with or without storm wind augmentation)
        if storm_data and augment_with_storm_winds:
            # Use storm-augmented interpolation
            wind_data, lats, lons = _smooth_data_with_storm_augmentation(
                grid_data, storm_data, smoothing_factor, 
                include_tornado_winds=show_tornado_paths
            )
        elif smoothing_factor > 1:
            # Standard smoothing without storm augmentation
            wind_data, lats, lons = _smooth_data(
                wind_data, lats, lons, smoothing_factor
            )
        
        # Determine color scale
        data_min = np.nanmin(wind_data)
        data_max = np.nanmax(wind_data)
        vmin = vmin if vmin is not None else max(0, data_min - 2)
        vmax = vmax if vmax is not None else min(80, data_max + 5)
        
        # Create figure with Cartopy projection
        figsize = figsize or (14, 10)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Set extent
        bounds = grid_data.grid_bounds
        margin = 0.1
        ax.set_extent([
            bounds.lon_min - margin,
            bounds.lon_max + margin,
            bounds.lat_min - margin,
            bounds.lat_max + margin
        ], crs=ccrs.PlateCarree())
        
        # Add basemap features
        if use_basemap:
            # Add terrain/land features
            ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', zorder=0)
            ax.add_feature(cfeature.LAKES, facecolor='#e6f3ff', alpha=0.5, zorder=0)
            ax.add_feature(cfeature.RIVERS, edgecolor='#a6cee3', linewidth=0.5, zorder=0)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#666666', linewidth=0.8, zorder=1)
            ax.add_feature(cfeature.BORDERS, edgecolor='#999999', linestyle='--', linewidth=0.5, zorder=1)
            ax.add_feature(cfeature.STATES, edgecolor='#999999', linewidth=0.3, zorder=1)
        
        # Create meshgrid for pcolormesh
        lon_step = (lons[-1] - lons[0]) / (len(lons) - 1) if len(lons) > 1 else 0.05
        lat_step = (lats[-1] - lats[0]) / (len(lats) - 1) if len(lats) > 1 else 0.05
        
        lon_edges = np.concatenate([
            [lons[0] - lon_step/2],
            (lons[:-1] + lons[1:]) / 2,
            [lons[-1] + lon_step/2]
        ])
        lat_edges = np.concatenate([
            [lats[0] - lat_step/2],
            (lats[:-1] + lats[1:]) / 2,
            [lats[-1] + lat_step/2]
        ])
        
        LON, LAT = np.meshgrid(lon_edges, lat_edges)
        
        # Create colormap
        cmap = create_wind_colormap(vmin, vmax)
        
        # Plot heatmap with transparency
        mesh = ax.pcolormesh(
            LON, LAT, wind_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='flat',
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        
        # Add state boundary from TIGER shapefile
        if show_boundary:
            geometry = get_state_geometry("North Carolina")
            if geometry is not None:
                try:
                    from cartopy.feature import ShapelyFeature
                    boundary_feature = ShapelyFeature(
                        [geometry], ccrs.PlateCarree(),
                        facecolor='none',
                        edgecolor='black',
                        linewidth=2,
                    )
                    ax.add_feature(boundary_feature, zorder=3)
                except Exception as e:
                    logger.warning(f"Could not add boundary feature: {e}")
                    # Fall back to manual plotting
                    boundary = get_nc_boundary()
                    if boundary:
                        boundary_lons = [p[0] for p in boundary]
                        boundary_lats = [p[1] for p in boundary]
                        ax.plot(boundary_lons, boundary_lats, 'k-', linewidth=2,
                               transform=ccrs.PlateCarree(), zorder=3)
            else:
                # Use fallback boundary
                boundary = get_nc_boundary()
                if boundary:
                    boundary_lons = [p[0] for p in boundary]
                    boundary_lats = [p[1] for p in boundary]
                    ax.plot(boundary_lons, boundary_lats, 'k-', linewidth=2,
                           transform=ccrs.PlateCarree(), zorder=3)
        
        # Mark maximum wind location
        if show_max_marker:
            stats = grid_data.get_statistics()
            if stats.get('max_location'):
                max_lat, max_lon = stats['max_location']
                ax.scatter(
                    max_lon, max_lat,
                    s=200, c='red', marker='*', edgecolors='white',
                    linewidths=2, zorder=10,
                    transform=ccrs.PlateCarree(),
                    label=f"Max: {stats['max_wind_speed']:.1f} mph"
                )
        
        # Add colorbar
        if show_colorbar:
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label(f'Peak Wind Speed ({grid_data.units})', fontsize=12)
            
            # Add threshold lines
            for t in [10, 20, 30, 40, 50]:
                if vmin <= t <= vmax:
                    cbar.ax.axhline(y=t, color='white', linewidth=0.5, alpha=0.5)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        else:
            bounds = grid_data.grid_bounds
            region_name = bounds.name or "Region"
            ax.set_title(
                f'Peak Wind Speed - {region_name}\n{grid_data.target_date}',
                fontsize=14, fontweight='bold', pad=10
            )
        
        # Add legend
        if show_max_marker:
            ax.legend(loc='upper left', fontsize=10)
        
        # Add statistics annotation
        stats = grid_data.get_statistics()
        stats_text = (
            f"Resolution: {grid_data.resolution}°\n"
            f"Points: {stats['valid_points']}/{stats['total_points']}\n"
            f"Max: {stats.get('max_wind_speed', 'N/A'):.1f} mph\n"
            f"Avg: {stats.get('avg_wind_speed', 'N/A'):.1f} mph"
        )
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            zorder=10
        )
        
        # Overlay storm reports if provided and markers are enabled
        if storm_data and storm_data.count > 0 and show_storm_markers:
            import cartopy.crs as ccrs
            plotted_types = overlay_storm_reports(
                ax, storm_data,
                show_tornado_paths=show_tornado_paths,
                show_wind_reports=show_wind_reports,
                transform=ccrs.PlateCarree()
            )
            
            # Add storm legend
            ef_ratings = list(set(
                r.tornado_path.ef_rating for r in storm_data.reports 
                if r.tornado_path and r.tornado_path.ef_rating
            ))
            if plotted_types:
                add_storm_legend(ax, plotted_types, ef_ratings)
            
            # Update stats text with storm count
            storm_text = f"\nStorm Reports: {storm_data.count}"
            if storm_data.tornado_count > 0:
                storm_text += f" ({storm_data.tornado_count} tornadoes)"
            ax.text(
                0.02, 0.15, storm_text.strip(),
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                zorder=10
            )
        
        fig.tight_layout()
        
        return fig


def _get_boundary_for_region(bounds: GridBounds) -> Optional[List[Tuple[float, float]]]:
    """
    Get state boundary coordinates for a region.
    
    Uses TIGER shapefile for NC, returns None for other regions.
    """
    nc_bounds = PRESET_REGIONS.get('north_carolina')
    if nc_bounds and bounds.lat_min < nc_bounds.lat_max and bounds.lat_max > nc_bounds.lat_min:
        return get_nc_boundary()
    
    return None


def plot_geo_heatmap_interactive(
    grid_data: GridData,
    title: Optional[str] = None,
    tiles: str = 'OpenStreetMap',
    zoom_start: int = 7,
    storm_data: Optional[SevereWeatherResponse] = None,
    show_tornado_paths: bool = True,
    show_wind_reports: bool = True,
    augment_with_storm_winds: bool = True,
    show_storm_markers: bool = True,
) -> 'folium.Map':
    """
    Create an interactive geographic heatmap using Folium.
    
    Args:
        grid_data: GridData object with wind data
        title: Map title (shown in popup)
        tiles: Map tile provider
        zoom_start: Initial zoom level
        storm_data: Optional SevereWeatherResponse with storm reports to overlay
        show_tornado_paths: Whether to draw tornado path polygons
        show_wind_reports: Whether to show wind event markers
        augment_with_storm_winds: Whether to incorporate storm wind data into heatmap
        show_storm_markers: Whether to display storm report markers on the map
        
    Returns:
        Folium Map object
    """
    import folium
    from folium.plugins import HeatMap
    
    bounds = grid_data.grid_bounds
    center = bounds.center
    
    # Create map
    m = folium.Map(
        location=[center[0], center[1]],
        zoom_start=zoom_start,
        tiles=tiles,
    )
    
    # Prepare heat data from grid
    heat_data = []
    max_wind = 0
    max_point = None
    max_lat, max_lon = None, None
    
    for point in grid_data.points:
        if point.has_data and point.peak_wind_speed:
            intensity = min(1.0, point.peak_wind_speed / 60)
            heat_data.append([point.latitude, point.longitude, intensity])
            
            if point.peak_wind_speed > max_wind:
                max_wind = point.peak_wind_speed
                max_point = point
    
    # Augment with storm wind data
    if storm_data and augment_with_storm_winds:
        storm_points = extract_storm_wind_points(
            storm_data, include_tornadoes=show_tornado_paths
        )
        for lat, lon, wind_speed in storm_points:
            # Check if within bounds
            if (bounds.lat_min <= lat <= bounds.lat_max and
                bounds.lon_min <= lon <= bounds.lon_max):
                # Higher intensity for storm winds (normalized to 100mph max for severe)
                intensity = min(1.0, wind_speed / 100)
                heat_data.append([lat, lon, intensity])
                
                # Track max wind including storm data
                if wind_speed > max_wind:
                    max_wind = wind_speed
                    max_lat, max_lon = lat, lon
                    max_point = None  # Storm wind, not a grid point
        
        if storm_points:
            logger.info(f"Added {len(storm_points)} storm wind points to interactive heatmap")
    
    if not heat_data:
        logger.warning("No valid data for interactive heatmap")
        return m
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=18,
        radius=25,
        blur=15,
        gradient={
            0.2: 'blue',
            0.4: 'cyan',
            0.6: 'lime',
            0.8: 'yellow',
            1.0: 'red'
        },
    ).add_to(m)
    
    # Add marker for maximum wind
    if max_point:
        popup_text = f"<b>Maximum Wind Speed</b><br>Speed: {max_point.peak_wind_speed:.1f} mph<br>"
        if max_point.peak_gust:
            popup_text += f"Gust: {max_point.peak_gust:.1f} mph<br>"
        popup_text += f"Location: ({max_point.latitude:.4f}, {max_point.longitude:.4f})"
        
        folium.Marker(
            [max_point.latitude, max_point.longitude],
            popup=folium.Popup(popup_text, max_width=200),
            icon=folium.Icon(color='red', icon='flag'),
            tooltip=f"Max: {max_point.peak_wind_speed:.1f} mph"
        ).add_to(m)
    elif max_lat is not None:
        # Max from storm wind data
        popup_text = f"<b>Maximum Wind Speed (Storm Report)</b><br>Speed: {max_wind:.1f} mph<br>"
        popup_text += f"Location: ({max_lat:.4f}, {max_lon:.4f})"
        
        folium.Marker(
            [max_lat, max_lon],
            popup=folium.Popup(popup_text, max_width=200),
            icon=folium.Icon(color='darkred', icon='warning-sign', prefix='glyphicon'),
            tooltip=f"Max (Storm): {max_wind:.1f} mph"
        ).add_to(m)
    
    # Add boundary polygon for NC from TIGER
    geometry = get_state_geometry("North Carolina")
    if geometry is not None:
        try:
            from shapely.geometry import mapping
            folium.GeoJson(
                mapping(geometry),
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'black',
                    'weight': 2,
                },
                tooltip='North Carolina'
            ).add_to(m)
        except Exception as e:
            logger.warning(f"Could not add GeoJSON boundary: {e}")
            # Fall back to coordinate list
            boundary = get_nc_boundary()
            if boundary:
                boundary_coords = [[lat, lon] for lon, lat in boundary]
                folium.Polygon(
                    locations=boundary_coords,
                    color='black',
                    weight=2,
                    fill=False,
                    tooltip='State Boundary'
                ).add_to(m)
    else:
        boundary = _get_boundary_for_region(bounds)
        if boundary:
            boundary_coords = [[lat, lon] for lon, lat in boundary]
            folium.Polygon(
                locations=boundary_coords,
                color='black',
                weight=2,
                fill=False,
                tooltip='State Boundary'
            ).add_to(m)
    
    # Add title
    if title:
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
                    background-color: white; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3); font-size: 14px;">
            <b>{title}</b><br>
            Date: {grid_data.target_date}<br>
            Max Wind: {max_wind:.1f} mph
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3); font-size: 12px;">
        <b>Wind Speed (mph)</b><br>
        <i style="background: blue; width: 20px; height: 10px; display: inline-block;"></i> 0-15<br>
        <i style="background: cyan; width: 20px; height: 10px; display: inline-block;"></i> 15-25<br>
        <i style="background: lime; width: 20px; height: 10px; display: inline-block;"></i> 25-35<br>
        <i style="background: yellow; width: 20px; height: 10px; display: inline-block;"></i> 35-50<br>
        <i style="background: red; width: 20px; height: 10px; display: inline-block;"></i> 50+
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add storm overlays if provided and markers are enabled
    if storm_data and storm_data.count > 0 and show_storm_markers:
        overlay_storm_reports_folium(
            m, storm_data,
            show_tornado_paths=show_tornado_paths,
            show_wind_reports=show_wind_reports
        )
        
        # Add storm legend
        storm_legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: white; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3); font-size: 12px;">
            <b>Storm Reports</b><br>
            <i style="background: #00FF00; width: 20px; height: 3px; display: inline-block;"></i> EF0<br>
            <i style="background: #FFFF00; width: 20px; height: 3px; display: inline-block;"></i> EF1<br>
            <i style="background: #FFA500; width: 20px; height: 3px; display: inline-block;"></i> EF2<br>
            <i style="background: #FF0000; width: 20px; height: 3px; display: inline-block;"></i> EF3+<br>
            <span style="color: orange;">●</span> T-Storm Wind
        </div>
        '''
        m.get_root().html.add_child(folium.Element(storm_legend_html))
    
    return m


def save_geo_heatmap_static(
    grid_data: GridData,
    output_path: Path,
    dpi: int = 150,
    **kwargs,
) -> Path:
    """
    Generate and save a static geo heatmap to file.
    
    Args:
        grid_data: GridData with wind data
        output_path: Output file path
        dpi: Image DPI
        **kwargs: Additional arguments for plot_geo_heatmap_static
        
    Returns:
        Path to saved file
    """
    fig = plot_geo_heatmap_static(grid_data, **kwargs)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info(f"Saved static geo heatmap to {output_path}")
    return output_path


def save_geo_heatmap_interactive(
    grid_data: GridData,
    output_path: Path,
    **kwargs,
) -> Path:
    """
    Generate and save an interactive geo heatmap to HTML file.
    
    Args:
        grid_data: GridData with wind data
        output_path: Output file path
        **kwargs: Additional arguments for plot_geo_heatmap_interactive
        
    Returns:
        Path to saved file
    """
    m = plot_geo_heatmap_interactive(grid_data, **kwargs)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    m.save(str(output_path))
    
    logger.info(f"Saved interactive geo heatmap to {output_path}")
    return output_path


def export_grid_to_csv(
    grid_data: GridData,
    output_path: Path,
    include_metadata: bool = True,
) -> Path:
    """
    Export grid wind data to CSV format.
    
    Creates a CSV file with geographic coordinates and peak wind speeds
    that can be used for external analysis and visualization tools.
    
    Args:
        grid_data: GridData object with wind data
        output_path: Output file path (should end in .csv)
        include_metadata: Whether to include header comments with metadata
        
    Returns:
        Path to saved CSV file
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if include_metadata:
            bounds = grid_data.grid_bounds
            stats = grid_data.get_statistics()
            f.write(f"# Wind Data Export\n")
            f.write(f"# Region: {bounds.name or 'Custom'}\n")
            f.write(f"# Date: {grid_data.target_date}\n")
            f.write(f"# Resolution: {grid_data.resolution} degrees\n")
            f.write(f"# Bounds: ({bounds.lat_min}, {bounds.lon_min}) to ({bounds.lat_max}, {bounds.lon_max})\n")
            f.write(f"# Valid Points: {stats['valid_points']}/{stats['total_points']}\n")
            f.write(f"# Units: {grid_data.units}\n")
            f.write(f"# Source: {grid_data.source}\n")
            f.write(f"#\n")
        
        writer.writerow([
            'latitude',
            'longitude', 
            'peak_wind_speed',
            'avg_wind_speed',
            'peak_gust',
            'peak_direction',
            'observation_count',
        ])
        
        for point in grid_data.points:
            if point.has_data:
                writer.writerow([
                    round(point.latitude, 6),
                    round(point.longitude, 6),
                    round(point.peak_wind_speed, 1) if point.peak_wind_speed else '',
                    round(point.avg_wind_speed, 1) if point.avg_wind_speed else '',
                    round(point.peak_gust, 1) if point.peak_gust else '',
                    round(point.peak_direction, 0) if point.peak_direction else '',
                    point.observation_count,
                ])
    
    logger.info(f"Exported grid data to CSV: {output_path}")
    return output_path
