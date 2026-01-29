# Wind Simulation Project Rules

## Project Overview
This project creates a geographic wind speed heatmap visualization tool that pulls historical wind data from the Weather Company (IBM) API, processes it across a grid of geographic coordinates, and displays peak wind speeds as map-based heatmaps for specified regions and time windows. The tool also integrates severe weather data from Local Storm Reports (LSR) to augment wind visualization with extreme wind events like tornado and thunderstorm wind observations. The tool exports both visual outputs (PNG/HTML maps) and CSV data files that can be used with external analysis tools and paired with proprietary wind damage data for research purposes.

---

## 1. Model Behavior Rules

### 1.1 General Development Principles
- **Simplicity First**: Always choose the simpler solution. This tool is for researchers, not software engineers.
- **Fail Gracefully**: All errors should be caught and display user-friendly messages, never raw stack traces.
- **Sensible Defaults**: Every configurable option should have a reasonable default value.
- **Offline Resilience**: Once data is fetched and cached, all visualization features should work without an internet connection.

### 1.2 Code Organization Rules
- **Single Responsibility**: Each module/file should do ONE thing well.
- **Flat Structure**: Avoid deep nesting. Maximum 2 levels of directories.
- **Explicit Over Implicit**: Name variables, functions, and files descriptively.
- **Comments for Why, Not What**: Code should be self-documenting; comments explain reasoning.

### 1.3 Data Handling Rules
- **Never Modify Raw Data**: Always work with copies; preserve original API responses.
- **Validate Early**: Check data integrity immediately after fetching.
- **Cache Aggressively**: Store fetched data locally to minimize API calls.
- **Grid-Based Processing**: Process data across geographic grids for spatial coverage.

---

## 2. API Usage Rules

### 2.1 Weather Company API Specific Rules
- **API Key Security**: 
  - NEVER hardcode API keys in source code
  - Use environment variables (`WEATHER_API_KEY`) or a local `.env` file
  - Add `.env` to `.gitignore` immediately
  
- **Rate Limiting**:
  - Implement exponential backoff for retries (1s, 2s, 4s, 8s, max 32s)
  - Track API calls per minute/day
  - Default: max 100 calls/minute

- **Request Best Practices**:
  - Always include timeout (default: 30 seconds)
  - Use session objects for connection pooling
  - Log all API requests and responses (sanitize keys)

### 2.2 Error Handling Rules
```
HTTP 200: Success - process data
HTTP 400: Bad Request - check parameters, show user-friendly message
HTTP 401: Unauthorized - prompt for valid API key
HTTP 403: Forbidden - check subscription/permissions
HTTP 404: Not Found - location may be invalid
HTTP 429: Rate Limited - wait and retry with backoff
HTTP 5xx: Server Error - retry with backoff, then fail gracefully
```

### 2.3 Data Validation Rules
- **Required Fields**: Every API response must contain: timestamp, wind_speed
- **Optional Fields**: wind_direction, wind_gust (handle missing gracefully)
- **Range Validation**:
  - Wind speed: 0-200 mph (flag outliers)
  - Wind direction: 0-360 degrees
  - Latitude: -90 to 90
  - Longitude: -180 to 180
- **Null Handling**: Replace null wind values with NaN, never 0

### 2.4 Caching Rules
- **Cache Key Format**: `{lat}_{lon}_{start_date}_{end_date}.json`
- **Cache Location**: `./data/cache/` directory
- **Cache Expiry**: 24 hours for recent data (< 7 days old), never for historical
- **Cache Validation**: Verify file integrity before using cached data

---

## 3. Feature Scope

### 3.1 Core Features (Implemented)
- **Command-Line Interface**: Full argparse CLI for scripted usage (`python -m src.cli`)
- **Geographic Grid Fetching**: Fetch wind data across a grid of lat/lon coordinates (up to 0.01° resolution, though 0.025° is recommended for performance)
- **NOAA Storm Events Integration**: Historical severe weather data from NOAA Storm Events Database (automatic download and caching)
- **Storm Wind Augmentation**: Combine grid and storm wind data using "max wins" logic (highest value preserved, never summed)
- **Tornado Wind Estimation**: EF rating to wind speed mapping (EF0: 75 mph, EF1: 98 mph, etc.)
- **Static Heatmap (PNG)**: Cartopy-based geographic heatmap with basemaps and TIGER state boundaries
- **Interactive Heatmap (HTML)**: Folium-based zoomable/pannable map with storm report overlays
- **Smoothed Visualization**: Bicubic interpolation for smooth gradient heatmaps
- **Semi-Transparent Overlay**: Configurable alpha for heatmap over basemap
- **TIGER Shapefiles**: Accurate state boundaries from Census Bureau data (loaded via geopandas)
- **CSV Export**: Export coordinates and peak wind speeds for external analysis
- **Smart Caching**: File-based caching to minimize API calls and NOAA downloads

### 3.2 CLI Flags Reference
| Flag | Default | Description |
|------|---------|-------------|
| `--start-date` | Yesterday | Start date for wind data (YYYY-MM-DD) |
| `--end-date` | Today | End date for wind data (YYYY-MM-DD) |
| `--region` | nc | Preset region (nc, nc_coast, etc.) |
| `--resolution` | 0.05 | Grid resolution in degrees |
| `--format` | all | Output format (png, html, csv, all) |
| `--verbose` | False | Enable debug logging |
| `--show-storm-markers` | False | Display storm markers and tornado paths on map |
| `--no-augment-storm-winds` | False | Disable storm wind augmentation (only use grid data) |
| `--no-max-marker` | False | Hide the star marker at peak wind location |
| `--smoothing` | 4 | Interpolation upsampling factor |
| `--alpha` | 0.5 | Heatmap transparency |

### 3.3 Out of Scope
- Time series plots
- Wind rose diagrams
- Statistical summaries
- Streamlit dashboard
- Real-time streaming data
- Predictive modeling

### 3.4 Visualization Rules
- **Single Visualization Type**: Geographic map-based heatmap only
- **Cartopy Basemaps**: Use Cartopy for map projections with land/ocean/river features
- **TIGER Boundaries**: Use Census Bureau TIGER shapefiles for accurate state outlines
- **Smoothing**: Apply bicubic interpolation (default 4x) for smooth gradients
- **Transparency**: Default alpha of 0.5 for heatmap over basemap
- **Accessible Colors**: Use colorblind-friendly color scales
- **Clear Labels**: All maps must include legends, scales, and location markers
- **Export Quality**: Static images at minimum 150 DPI
- **Storm Report Overlays**: Tornado paths colored by EF rating, storm markers by event type
- **Max-Wins Data Fusion**: When combining grid and storm data, keep only the highest wind value per location (never sum)

### 3.5 Severe Weather Data Rules
- **NOAA Storm Events**: Fetch from NOAA Storm Events Database (https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/)
- **Supported Event Types**: Tornado, Thunderstorm Wind, High Wind, Hail, Flash Flood
- **Data Delay**: NOAA data has approximately 120-day publication delay
- **EF Rating Wind Mapping**: EF0=75mph, EF1=98mph, EF2=123mph, EF3=150mph, EF4=183mph, EF5=220mph
- **Tornado Path Points**: Add wind estimates at both touchdown and liftoff locations
- **Grid Augmentation**: Snap storm observations to nearest grid cell, take maximum value
- **Default Behavior**: Severe weather overlay and augmentation enabled by default

---

## 4. Output Files

### 4.1 Required Outputs
Every run should produce:
1. **Static Heatmap (PNG)**: High-resolution geographic heatmap
2. **Interactive Heatmap (HTML)**: Folium-based map viewable in browser
3. **CSV Data File**: Coordinates and peak wind speeds for external use

### 4.2 CSV Format
```csv
# Wind Data Export
# Region: North Carolina
# Date: 2024-01-07 to 2024-01-09
# Resolution: 0.05 degrees
# Bounds: (33.8, -84.3) to (36.6, -75.4)
# Valid Points: 444/444
# Units: mph
# Source: weather_company
#
latitude,longitude,peak_wind_speed,avg_wind_speed,peak_gust,peak_direction,observation_count
35.55,-75.50,25.3,18.2,32.1,225,72
...
```

---

## 5. Code Style Rules

### 5.1 Python Specific
- **Version**: Python 3.9+
- **Formatting**: Black formatter (line length 88)
- **Linting**: Flake8 with default settings
- **Type Hints**: Required for all function signatures
- **Imports**: Use isort, standard library → third party → local

### 5.2 Naming Conventions
```python
# Files: lowercase_with_underscores.py
# Classes: PascalCase
# Functions: lowercase_with_underscores
# Constants: UPPERCASE_WITH_UNDERSCORES
# Variables: lowercase_with_underscores
```

### 5.3 Project Structure
```
Wind_Simulation/
├── src/
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface
│   ├── api/
│   │   ├── __init__.py
│   │   ├── cache.py             # API response caching
│   │   ├── weather_client.py    # Weather Company API client
│   │   └── noaa_client.py       # NOAA Storm Events client
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py            # Pydantic models (wind + storm data)
│   │   ├── processor.py         # Unit conversion only
│   │   ├── grid.py              # Grid structures + TIGER loading
│   │   └── grid_fetcher.py
│   └── visualization/
│       ├── __init__.py
│       ├── geo_heatmap.py       # Cartopy heatmap + storm overlays + CSV
│       └── styles.py
├── data/
│   ├── cache/                   # Cached Weather API responses
│   ├── noaa_cache/              # Cached NOAA Storm Events files
│   ├── shapefiles/              # Cached TIGER shapefiles
│   └── exports/
├── output/
│   ├── north_carolina/          # Default output directory
│   └── comparison/              # Comparison visualizations
├── config/
│   └── settings.py
├── tests/
├── test_nc_geo_heatmap.py       # Test script for heatmaps
├── test_storm_winds_only.py     # Test script for storm-only visualization
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 6. Testing Rules

### 6.1 Test Requirements
- **Unit Tests**: For all data processing functions
- **Integration Tests**: For API client (with mocked responses)

### 6.2 Test Data
- Use fixtures with known wind data values
- Include edge cases: zero wind, maximum wind, missing data
- Default test region: North Carolina

---

## 7. Documentation Rules

### 7.1 README Requirements
- Project description (1 paragraph)
- Installation instructions (step by step)
- Configuration (API key setup)
- CLI usage with examples
- Programmatic usage examples
- Output file descriptions
- Grid resolution reference table

### 7.2 Inline Documentation
- Module docstring at top of each file
- Function docstrings with Args, Returns, Raises
- Complex logic should have explanatory comments
