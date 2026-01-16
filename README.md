# Wind Speed Geographic Heatmap Tool

A Python application for fetching historical wind data from the Weather Company (IBM) API and generating geographic heatmap visualizations showing peak wind speeds across regions. Designed for researchers studying wind patterns and pairing with wind damage data.

## 🌪️ Features

- **Geographic Heatmaps**: Map-based visualizations showing peak wind speeds across regions
- **Severe Weather Integration**: Local Storm Reports (LSR) for tornado and thunderstorm wind data
- **Storm Wind Augmentation**: Combine grid and severe weather data using "max wins" logic
- **Tornado Path Visualization**: EF-rated tornado paths with color-coded overlays
- **Cartopy Basemaps**: Professional map projections with land, ocean, rivers, and state boundaries
- **TIGER Shapefiles**: Accurate state boundaries from Census Bureau data
- **Smoothed Visualization**: Bicubic interpolation for smooth gradient heatmaps
- **Static PNG Output**: High-resolution heatmaps for reports and publications
- **Interactive HTML Maps**: Zoomable, pannable Folium-based maps with storm report popups
- **CSV Data Export**: Export coordinates and wind speeds for external analysis tools
- **Command-Line Interface**: Full CLI for scripted/automated usage
- **Grid-Based Fetching**: Collect data across configurable geographic grids (up to 0.05° resolution)
- **Smart Caching**: File-based caching to minimize API calls
- **Demo Mode**: Test without an API key using realistic synthetic data (includes Jan 8-9, 2024 NC storm)

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Output Files](#output-files)
- [API Reference](#api-reference)
- [Preset Regions](#preset-regions)
- [Project Structure](#project-structure)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Weather Company API key

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Wind_Simulation.git
   cd Wind_Simulation
   ```

2. **Create a conda environment** (recommended)
   ```bash
   conda create -n wind_simulation python=3.11
   conda activate wind_simulation
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies include:
   - `cartopy` - Map projections and basemaps
   - `geopandas` - Geographic data handling
   - `scipy` - Data interpolation/smoothing
   - `folium` - Interactive maps
   - `matplotlib` - Static visualizations

4. **Configure environment** (optional, for API access)
   ```bash
   cp .env.example .env
   # Edit .env and add your Weather Company API key
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`):

```env
# Weather Company API Key (optional - demo mode available)
WEATHER_API_KEY=your_api_key_here
```

### Demo Mode

If no API key is configured, the application runs in **demo mode**:
- Generates realistic, spatially coherent synthetic wind data
- Includes coastal effects, storm patterns, and diurnal variation
- Includes realistic severe weather data from the January 8-9, 2024 NC storm event
- Demo storm data includes EF1 tornadoes (Claremont, New Bern, Harkers Island) and thunderstorm wind reports
- All features work normally - useful for testing and development

## 🎯 Quick Start

### Command-Line Interface (Recommended)

The easiest way to generate heatmaps is via the CLI:

```bash
# Generate heatmap for North Carolina with demo data
python -m src.cli --demo

# Custom date range
python -m src.cli --start-date 2024-01-07 --end-date 2024-01-09 --demo

# Custom geographic bounds
python -m src.cli --lat-min 35.0 --lat-max 36.5 --lon-min -80 --lon-max -75 --demo

# High resolution (0.05° ≈ 5.5km)
python -m src.cli --resolution 0.05 --demo

# Output only CSV data
python -m src.cli --format csv --demo

# See all options
python -m src.cli --help
```

### CLI Options

| Option | Description | Default |
|--------|-------------|----------|
| `--start-date` | Start date (YYYY-MM-DD) | 2024-01-08 |
| `--end-date` | End date (YYYY-MM-DD) | 2024-01-09 |
| `--lat-min/max` | Custom latitude bounds | NC preset |
| `--lon-min/max` | Custom longitude bounds | NC preset |
| `--region` | Preset region name | north_carolina |
| `--resolution` | Grid resolution in degrees | 0.25 |
| `--format` | Output format (png/html/csv/all) | all |
| `--smoothing` | Interpolation factor (1-8) | 4 |
| `--alpha` | Heatmap transparency (0-1) | 0.5 |
| `--demo` | Use synthetic data | False |
| `--output-dir` | Output directory | output/<region> |
| `--severe-weather` | Enable severe weather overlay | True |
| `--no-severe-weather` | Disable severe weather overlay | - |
| `--tornado-paths` | Show tornado path polygons | True |
| `--no-tornado-paths` | Disable tornado paths | - |
| `--augment-storm-winds` | Incorporate storm winds into heatmap | True |
| `--no-augment-storm-winds` | Use only grid data for heatmap | - |
| `--event-types` | Filter storm types (comma-separated) | all |

### Storm Wind Augmentation

When `--augment-storm-winds` is enabled (default), the tool combines historical grid data with severe weather observations:

```bash
# Default: grid + storm winds combined (max wins)
python -m src.cli --demo

# Compare different visualizations:
python -m src.cli --demo --no-augment-storm-winds  # Grid data only
python test_storm_winds_only.py                      # Storm data only
```

**How it works:**
1. Fetch historical wind data at regular grid points (e.g., 0.25° resolution)
2. Fetch Local Storm Reports (tornado, thunderstorm wind, high wind events)
3. For tornadoes, estimate wind speeds from EF rating (EF1 ≈ 98 mph)
4. For each grid cell, keep the **maximum** wind value (grid vs storm) - values are never summed
5. Apply bicubic interpolation for smooth visualization

**Example output (demo mode):**
```
Grid max: 32.8 mph
Storm max: 98 mph (EF1 tornado)
Augmented max: 105.6 mph (after interpolation)
```

### Test Scripts

Run the test scripts for quick demos:

```bash
# Standard heatmap test
python test_nc_geo_heatmap.py

# Storm winds only visualization (for comparison)
python test_storm_winds_only.py
```

### Programmatic Usage

```python
from datetime import date
from src.data.grid import PRESET_REGIONS
from src.cli import run_heatmap

# Using the run_heatmap function
results = run_heatmap(
    start_date=date(2024, 1, 7),
    end_date=date(2024, 1, 9),
    bounds=PRESET_REGIONS["north_carolina"],
    resolution=0.1,
    output_dir=Path("output/my_analysis"),
    output_format='all',
    demo_mode=True,
    smoothing_factor=4,
    alpha=0.5,
)

print(f"Generated: {results}")
```

Or use individual components:

```python
from datetime import date
from pathlib import Path
from src.data.grid import PRESET_REGIONS
from src.data.grid_fetcher import fetch_grid_data_with_progress
from src.api.weather_client import WeatherClient
from src.visualization.geo_heatmap import (
    save_geo_heatmap_static,
    save_geo_heatmap_interactive,
    export_grid_to_csv,
)

# Initialize client (demo_mode=True for testing without API key)
client = WeatherClient(demo_mode=True)

# Get region bounds
nc_bounds = PRESET_REGIONS["north_carolina"]

# Fetch grid data
grid_data = fetch_grid_data_with_progress(
    bounds=nc_bounds,
    start_date=date(2024, 1, 7),
    end_date=date(2024, 1, 9),
    resolution=0.1,  # degrees
    client=client,
)

# Generate outputs with smoothing and transparency
save_geo_heatmap_static(
    grid_data, 
    Path("output/heatmap.png"),
    smoothing_factor=4,  # 4x interpolation
    alpha=0.5,           # Semi-transparent over basemap
)
save_geo_heatmap_interactive(grid_data, Path("output/heatmap.html"))
export_grid_to_csv(grid_data, Path("output/wind_data.csv"))
```

## 📁 Output Files

Each run produces three output files:

### 1. Static Heatmap (PNG)
- High-resolution geographic heatmap with Cartopy projection
- Basemap with land, ocean, lakes, rivers, and coastlines
- Accurate state boundaries from Census TIGER shapefiles
- Smoothed heatmap overlay (bicubic interpolation)
- Semi-transparent overlay for geographic context
- Color scale indicating wind speed intensity
- Statistics annotation (max, avg, resolution)

### 2. Interactive Map (HTML)
- Folium-based zoomable/pannable map
- Click markers for detailed information
- Legend showing wind speed scale
- Open in any web browser

### 3. CSV Data Export
```csv
# Wind Data Export
# Region: North Carolina
# Date: 2024-01-02
# Resolution: 0.25 degrees
latitude,longitude,peak_wind_speed,avg_wind_speed,peak_gust,peak_direction,observation_count
35.55,-75.50,25.3,18.2,32.1,225,24
35.55,-75.25,23.8,17.5,30.2,230,24
...
```

The CSV includes:
- Geographic coordinates (latitude, longitude)
- Peak wind speed at each location
- Average wind speed
- Peak gust speed
- Predominant wind direction
- Number of observations

## 🗺️ Preset Regions

| Name | Description | Bounds |
|------|-------------|--------|
| `north_carolina` | Full state of NC | 33.8°N-36.6°N, 84.3°W-75.4°W |
| `outer_banks` | Outer Banks, NC | 34.5°N-36.5°N, 76.5°W-75.3°W |
| `florida` | State of Florida | 24.5°N-31.0°N, 87.6°W-80.0°W |
| `texas_coast` | Texas Gulf Coast | 26.0°N-30.0°N, 97.5°W-93.5°W |
| `northeast_us` | Northeast US | 38.5°N-45.0°N, 80.5°W-66.9°W |

### Custom Regions

```python
from src.data.grid import GridBounds

custom_bounds = GridBounds(
    lat_min=34.0,
    lat_max=36.0,
    lon_min=-80.0,
    lon_max=-78.0,
    name="Custom Region"
)
```

## 🔌 API Reference

### Weather Company API

This application uses two Weather Company (IBM) APIs:

#### Historical Weather API
- **Endpoint**: `GET /v3/wx/observations/hourly/history`
- **Purpose**: Fetch hourly wind observations for grid points
- **Rate Limits**: 100 requests per minute (configurable)
- **Automatic retry**: Exponential backoff on rate limit errors

#### Severe Weather API (Local Storm Reports)
- **Endpoint**: `GET /v2/alerts/lsr`
- **Purpose**: Fetch tornado, thunderstorm wind, and hail reports
- **Event Types**: tornado, thunderstorm_wind, hail, high_wind, flash_flood, wildfire
- **Data Returned**: Location, timestamp, magnitude (wind speed/hail size), EF rating for tornadoes

### EF Rating Wind Speed Mapping

| EF Rating | Wind Speed Range | Estimated Value Used |
|-----------|------------------|----------------------|
| EF0 | 65-85 mph | 75 mph |
| EF1 | 86-110 mph | 98 mph |
| EF2 | 111-135 mph | 123 mph |
| EF3 | 136-165 mph | 150 mph |
| EF4 | 166-200 mph | 183 mph |
| EF5 | >200 mph | 220 mph |

### Grid Resolution

| Resolution | Points/Degree | Example (NC) | Approx. Distance |
|------------|--------------|--------------|------------------|
| 0.5° | 4 | ~120 points | ~55 km |
| 0.25° | 16 | ~480 points | ~28 km |
| 0.1° | 100 | ~2,700 points | ~11 km |
| 0.05° | 400 | ~4,800 points | ~5.5 km |

Default resolution is **0.05°** for maximum detail. Higher resolution = more detail but more API calls and longer processing time.

## 📂 Project Structure

```
Wind_Simulation/
├── src/
│   ├── cli.py                   # Command-line interface
│   ├── api/
│   │   ├── cache.py             # API response caching
│   │   └── weather_client.py    # Weather + Severe Weather APIs
│   ├── data/
│   │   ├── models.py            # Pydantic models (wind + storm data)
│   │   ├── processor.py         # Data processing utilities
│   │   ├── grid.py              # Grid structures & TIGER shapefile loading
│   │   └── grid_fetcher.py      # Grid data fetching
│   └── visualization/
│       ├── geo_heatmap.py       # Cartopy heatmap, storm overlays, CSV export
│       └── styles.py            # Visualization styling
├── data/
│   ├── cache/                   # Cached API responses
│   └── shapefiles/              # Cached TIGER shapefiles
├── output/
│   ├── north_carolina/          # Default output directory
│   └── comparison/              # Comparison visualizations
├── config/
│   └── settings.py              # Configuration settings
├── test_nc_geo_heatmap.py       # Test script for heatmaps
├── test_storm_winds_only.py     # Test script for storm-only visualization
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Troubleshooting

### "No API key configured"
The application will use demo mode with synthetic data. For real data, add your API key to `.env`.

### "Rate limit exceeded"
The application automatically handles rate limits with backoff. For large grids, consider:
- Using lower resolution (0.5° instead of 0.25°)
- Enabling caching (`use_cache=True`)
- Waiting between runs

### High memory usage
For large grids (>1000 points), the application may use significant memory. Consider:
- Processing in smaller regional chunks
- Using lower resolution

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read the project rules in `.copilot/rules/project-rules.md` before submitting changes.
