# How to Use the Wind Speed Heatmap Tool

A step-by-step guide to setting up and running the Wind Speed Geographic Heatmap Tool.

---

## Table of Contents

1. [Installing Conda](#1-installing-conda)
2. [Environment Setup](#2-environment-setup)
3. [Quick Start Commands](#3-quick-start-commands)
4. [Understanding the Output](#4-understanding-the-output)
5. [Common Use Cases](#5-common-use-cases)
6. [Input Format Reference](#6-input-format-reference)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Installing Conda

Conda is a package and environment manager that simplifies installing Python dependencies, especially scientific libraries like Cartopy and GeoPandas.

### Option A: Install Miniconda (Recommended)

Miniconda is a lightweight installer that includes only conda and Python.

#### macOS

```bash
# Download Miniconda installer
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  # For Apple Silicon (M1/M2/M3)
# OR
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh  # For Intel Macs

# Run the installer
bash Miniconda3-latest-MacOSX-*.sh

# Follow the prompts:
# - Press Enter to review the license
# - Type 'yes' to accept the license
# - Press Enter to confirm the installation location
# - Type 'yes' to initialize conda

# Restart your terminal or run:
source ~/.zshrc
```

#### Linux

```bash
# Download Miniconda installer
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow the prompts, then restart your terminal or run:
source ~/.bashrc
```

#### Windows

1. Download the installer from: https://docs.conda.io/en/latest/miniconda.html
2. Run the `.exe` installer
3. Follow the installation wizard
4. Open "Anaconda Prompt" from the Start menu

### Option B: Install Anaconda (Full Distribution)

Anaconda includes conda plus 250+ pre-installed scientific packages (larger download).

1. Download from: https://www.anaconda.com/download
2. Run the installer for your operating system
3. Follow the installation wizard

### Verify Conda Installation

```bash
# Check conda is installed
conda --version
# Should output something like: conda 24.x.x

# Update conda to latest version
conda update conda -y
```

---

## 2. Environment Setup

### Prerequisites

- Conda installed (see [Section 1](#1-installing-conda) if not)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Wind_Simulation.git
cd Wind_Simulation
```

### Step 2: Create the Conda Environment

The project includes an `environment.yml` file with all required dependencies:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate wind_simulation
```

If you encounter issues, you can also create the environment manually:

```bash
# Create a new environment with Python 3.11
conda create -n wind_simulation python=3.11 -y
conda activate wind_simulation

# Install core dependencies
conda install -c conda-forge cartopy geopandas scipy folium matplotlib numpy pydantic requests python-dotenv -y
```

### Step 3: Verify the Installation

```bash
# Test the CLI help
python -m src.cli --help
```

You should see a list of available options.

### Step 4: Configure API Key (Required)

To use the Weather Company API for grid wind data:

```bash
# Create a .env file
cp .env.example .env

# Edit .env and add your API key
# WEATHER_API_KEY=your_api_key_here
```

**Note:** An API key is required for grid wind data. NOAA Storm Events data (for severe weather reports) does not require an API key and is downloaded automatically.

---

## 3. Quick Start Commands

### Basic Usage

```bash
# Basic run - generates PNG, HTML, and CSV outputs
python -m src.cli

# With verbose logging to see progress
python -m src.cli --verbose
```

### Customize Date Range

```bash
# Specific date range
python -m src.cli --start-date 2024-01-07 --end-date 2024-01-09
```

### Adjust Resolution

```bash
# Low resolution (fast, ~2,700 points for NC)
python -m src.cli --resolution 0.1

# Medium resolution (balanced, ~10,800 points)
python -m src.cli --resolution 0.05

# High resolution (slow, ~40,000 points, recommended max)
python -m src.cli --resolution 0.025
```

### Choose Output Format

```bash
# Only PNG (static image)
python -m src.cli --format png

# Only HTML (interactive map)
python -m src.cli --format html

# Only CSV (data export)
python -m src.cli --format csv

# All formats (default)
python -m src.cli --format all
```

### Control Storm Data

```bash
# Default: Storm data augments the heatmap values, but markers are hidden
python -m src.cli

# Show storm markers (tornado paths, thunderstorm wind locations)
python -m src.cli --show-storm-markers

# Disable storm wind augmentation (only show base grid data)
python -m src.cli --no-augment-storm-winds

# Hide the star marker at the peak wind location
python -m src.cli --no-max-marker
```

---

## 4. Understanding the Output

After running, you'll find output files in `output/<region_name>/`:

### PNG Heatmap

- Geographic map with semi-transparent wind speed overlay
- Color scale: blue (low) → green → yellow → red (high)
- State boundaries from Census TIGER shapefiles
- Statistics box showing max/avg wind speeds
- Star marker (optional) showing peak wind location

### HTML Interactive Map

- Zoomable/pannable map (open in web browser)
- Heat overlay showing wind intensity
- Click for location details
- Wind speed legend

### CSV Data Export

```csv
# Wind Data Export
# Region: North Carolina
# Date: 2024-01-08 to 2024-01-09
latitude,longitude,peak_wind_speed,avg_wind_speed,peak_gust,peak_direction,observation_count
35.55,-75.50,25.3,18.2,32.1,225,48
...
```

---

## 5. Common Use Cases

### Use Case 1: High-Resolution Heatmap for Report

```bash
python -m src.cli \
    --start-date 2024-01-08 \
    --end-date 2024-01-09 \
    --resolution 0.025 \
    --format png \
    --no-max-marker
```

### Use Case 2: Interactive Map with Storm Markers

```bash
python -m src.cli \
    --start-date 2024-01-08 \
    --end-date 2024-01-09 \
    --format html \
    --show-storm-markers
```

### Use Case 3: Export Data Only (for External Analysis)

```bash
python -m src.cli \
    --resolution 0.05 \
    --format csv
```

### Use Case 4: Custom Region (Not North Carolina)

```bash
python -m src.cli \
    --lat-min 25.0 \
    --lat-max 31.0 \
    --lon-min -87.5 \
    --lon-max -80.0 \
    --resolution 0.1
```

### Use Case 5: Compare Grid Data vs Storm-Augmented Data

```bash
# Run with storm augmentation (default)
python -m src.cli --output-dir output/with_storm

# Run without storm augmentation
python -m src.cli --no-augment-storm-winds --output-dir output/grid_only
```

---

## 6. Input Format Reference

### Date Format

Dates must be in **YYYY-MM-DD** format:

```bash
--start-date 2024-01-08
--end-date 2024-01-09
```

### Geographic Bounds

Coordinates use decimal degrees:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--lat-min` | Southern boundary | 33.8 |
| `--lat-max` | Northern boundary | 36.6 |
| `--lon-min` | Western boundary | -84.3 |
| `--lon-max` | Eastern boundary | -75.4 |

**Note:** Longitudes in the Western Hemisphere are **negative**.

### Preset Regions

Instead of custom bounds, use preset region names:

```bash
--region north_carolina    # Full state of NC (default)
--region outer_banks       # NC Outer Banks
--region florida           # State of Florida
--region texas_coast       # Texas Gulf Coast
--region northeast_us      # Northeast US
```

### Resolution Values

| Resolution | Grid Size (NC) | Approximate Points | Speed |
|------------|----------------|-------------------|-------|
| 0.25° | ~27 km | ~270 | Very Fast |
| 0.1° | ~11 km | ~2,700 | Fast |
| 0.05° | ~5.5 km | ~10,800 | Moderate |
| 0.025° | ~2.7 km | ~40,000 | Slow |
| 0.01° | ~1.1 km | ~250,000 | Very Slow |

**Recommendation:** Use 0.025° for high-quality output. Avoid 0.01° unless you have specific needs.

---

## 7. Troubleshooting

### "ModuleNotFoundError: No module named 'cartopy'"

**Solution:** Install cartopy via conda:

```bash
conda activate wind_simulation
conda install -c conda-forge cartopy -y
```

### "ModuleNotFoundError: No module named 'geopandas'"

**Solution:** Install geopandas via conda:

```bash
conda activate wind_simulation
conda install -c conda-forge geopandas -y
```

### "WARNING - geopandas not installed, using fallback boundary"

**Solution:** Same as above - install geopandas.

### Heatmap shows low max wind speed despite storm events

**Cause:** Storm data augmentation was disabled or not working.

**Solution:** Ensure you're NOT using `--no-augment-storm-winds`:

```bash
# Correct - augmentation is enabled by default
python -m src.cli

# Check output for: "Augmented interpolation complete: grid max=X mph, augmented max=Y mph"
```

### Process hangs or is very slow

**Cause:** Resolution is too high for the region size.

**Solution:** Use a lower resolution:

```bash
# Start with 0.1° to test
python -m src.cli --resolution 0.1
```

### API rate limit exceeded

**Cause:** Too many API calls in short time period.

**Solution:** Reduce resolution or wait between runs:

```bash
python -m src.cli --resolution 0.1
```

### NOAA data not available for recent dates

**Cause:** NOAA Storm Events data has approximately a 120-day publication delay.

**Solution:** Use older dates or wait for data to become available. The application will show a warning and continue with empty storm data.

---

## Need More Help?

- Check the [README.md](README.md) for detailed API documentation
- Review [.copilot/rules/project-rules.md](.copilot/rules/project-rules.md) for project conventions
- Run `python -m src.cli --help` for complete CLI reference


conda run -n wind_simulation python -m src.cli --start-date 2024-01-08 --end-date 2024-01-09 --region outer_banks --resolution 0.025 --format png --no-max-marker