"""
Visualization styles and constants.

This module defines consistent styling for all wind visualizations,
including color palettes, figure sizes, and matplotlib configurations.

All styles are designed to be colorblind-friendly and suitable for
publication-quality output.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Tuple, List

# =============================================================================
# Figure Sizes
# =============================================================================

# Standard figure sizes (width, height) in inches
FIGURE_SIZES: Dict[str, Tuple[float, float]] = {
    "heatmap": (10, 8),
    "summary": (8, 6),
    "small": (6, 4),
    "large": (14, 8),
}

# Default DPI for exports
DEFAULT_DPI = 300

# =============================================================================
# Color Schemes
# =============================================================================

# Primary colormap for wind speed (blue=low, red=high)
WIND_COLORMAP = "RdYlBu_r"  # Reversed Red-Yellow-Blue

# Alternative colormaps
COLORMAPS = {
    "default": "RdYlBu_r",
    "sequential": "YlOrRd",      # Yellow-Orange-Red for heatmaps
    "diverging": "coolwarm",     # For anomalies
    "categorical": "Set2",       # For categories
}

# Wind speed color scale (for custom mappings)
WIND_SPEED_COLORS = {
    "calm": "#3288bd",       # Blue (0-5 mph)
    "light": "#66c2a5",      # Teal (5-15 mph)
    "moderate": "#abdda4",   # Light green (15-25 mph)
    "fresh": "#e6f598",      # Yellow-green (25-35 mph)
    "strong": "#fee08b",     # Yellow (35-45 mph)
    "gale": "#fdae61",       # Orange (45-55 mph)
    "storm": "#f46d43",      # Red-orange (55-65 mph)
    "violent": "#d53e4f",    # Red (65-75 mph)
    "hurricane": "#9e0142",  # Dark red (75+ mph)
}

# Colorblind-friendly palette for categorical data
CATEGORICAL_COLORS = [
    "#0077BB",  # Blue
    "#33BBEE",  # Cyan
    "#009988",  # Teal
    "#EE7733",  # Orange
    "#CC3311",  # Red
    "#EE3377",  # Magenta
    "#BBBBBB",  # Grey
]

# =============================================================================
# Typography
# =============================================================================

FONT_SIZES = {
    "title": 14,
    "subtitle": 12,
    "axis_label": 11,
    "tick_label": 10,
    "legend": 10,
    "annotation": 9,
}

FONT_FAMILY = "sans-serif"

# =============================================================================
# Plot Style Configuration
# =============================================================================

def get_plot_style() -> Dict:
    """
    Get matplotlib rcParams for consistent styling.
    
    Returns:
        Dictionary of rcParams settings
    """
    return {
        # Figure
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.dpi": 100,
        
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["axis_label"],
        
        # Grid
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        
        # Ticks
        "xtick.labelsize": FONT_SIZES["tick_label"],
        "ytick.labelsize": FONT_SIZES["tick_label"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Legend
        "legend.fontsize": FONT_SIZES["legend"],
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        
        # Font
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZES["tick_label"],
    }


def apply_style():
    """Apply the wind simulation plot style globally."""
    plt.rcParams.update(get_plot_style())


def style_context():
    """
    Context manager for applying style temporarily.
    
    Usage:
        with style_context():
            fig, ax = plt.subplots()
            # ... plotting code
    """
    return plt.rc_context(get_plot_style())


# =============================================================================
# Color Utilities
# =============================================================================

def get_wind_color(speed: float, max_speed: float = 75.0) -> str:
    """
    Get color for wind speed value.
    
    Args:
        speed: Wind speed in mph
        max_speed: Maximum expected speed (for normalization)
        
    Returns:
        Hex color string
    """
    # Define thresholds
    thresholds = [
        (5, "calm"),
        (15, "light"),
        (25, "moderate"),
        (35, "fresh"),
        (45, "strong"),
        (55, "gale"),
        (65, "storm"),
        (75, "violent"),
    ]
    
    for threshold, category in thresholds:
        if speed <= threshold:
            return WIND_SPEED_COLORS[category]
    
    return WIND_SPEED_COLORS["hurricane"]


def create_wind_colormap(
    vmin: float = 0,
    vmax: float = 75,
    n_colors: int = 256
) -> mcolors.LinearSegmentedColormap:
    """
    Create a custom colormap for wind speeds.
    
    Args:
        vmin: Minimum wind speed
        vmax: Maximum wind speed
        n_colors: Number of colors in colormap
        
    Returns:
        Matplotlib colormap
    """
    colors = list(WIND_SPEED_COLORS.values())
    return mcolors.LinearSegmentedColormap.from_list(
        "wind_speed", colors, N=n_colors
    )


def get_wind_speed_cmap():
    """Get the default wind speed colormap."""
    return plt.get_cmap(WIND_COLORMAP)


# =============================================================================
# Annotation Helpers
# =============================================================================

def format_wind_speed(speed: float, unit: str = "mph", decimals: int = 1) -> str:
    """Format wind speed with unit."""
    if speed is None:
        return "N/A"
    return f"{speed:.{decimals}f} {unit}"


def format_direction(degrees: float) -> str:
    """Format wind direction as cardinal."""
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    index = round(degrees / 22.5) % 16
    return f"{directions[index]} ({degrees:.0f}°)"


# =============================================================================
# Initialize on import
# =============================================================================

# Apply default style
apply_style()
