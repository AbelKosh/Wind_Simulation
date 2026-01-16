"""
Visualization module for geographic wind heatmaps.

This module provides functions for creating map-based visualizations
showing peak wind speeds across geographic regions.

Exports:
    - plot_geo_heatmap_static: Create static matplotlib heatmap
    - plot_geo_heatmap_interactive: Create interactive Folium map
    - save_geo_heatmap_static: Save static heatmap to PNG
    - save_geo_heatmap_interactive: Save interactive heatmap to HTML
    - export_grid_to_csv: Export grid data to CSV format
"""

from src.visualization.geo_heatmap import (
    plot_geo_heatmap_static,
    plot_geo_heatmap_interactive,
    save_geo_heatmap_static,
    save_geo_heatmap_interactive,
    export_grid_to_csv,
)
from src.visualization.styles import (
    WIND_COLORMAP,
    FIGURE_SIZES,
    DEFAULT_DPI,
    style_context,
)

__all__ = [
    # Geographic heatmap functions
    "plot_geo_heatmap_static",
    "plot_geo_heatmap_interactive",
    "save_geo_heatmap_static",
    "save_geo_heatmap_interactive",
    "export_grid_to_csv",
    # Style utilities
    "WIND_COLORMAP",
    "FIGURE_SIZES",
    "DEFAULT_DPI",
    "style_context",
]
