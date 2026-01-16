"""
Data processing module for wind simulation.

This module provides functions for transforming wind data from API responses
into analysis-ready formats, including unit conversion functionality.

Usage:
    from src.data.processor import response_to_dataframe, convert_wind_speed
    
    df = response_to_dataframe(wind_response)
"""

import logging
import math
from typing import Optional

import pandas as pd

from src.data.models import WindDataResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Unit Conversion Constants
# =============================================================================

# Conversion factors to meters per second (base unit)
WIND_SPEED_TO_MS = {
    "mph": 0.44704,        # miles per hour
    "m/s": 1.0,            # meters per second (base)
    "km/h": 0.277778,      # kilometers per hour
    "kph": 0.277778,       # alias for km/h
    "knots": 0.514444,     # nautical miles per hour
    "kts": 0.514444,       # alias for knots
}

# Human-readable unit labels
UNIT_LABELS = {
    "mph": "mph",
    "m/s": "m/s",
    "km/h": "km/h",
    "kph": "km/h",
    "knots": "knots",
    "kts": "knots",
}


# =============================================================================
# DataFrame Conversion
# =============================================================================

def response_to_dataframe(response: WindDataResponse) -> pd.DataFrame:
    """
    Convert WindDataResponse to pandas DataFrame.
    
    Creates a DataFrame with columns for all observation fields,
    sorted by timestamp.
    
    Args:
        response: WindDataResponse from API client
        
    Returns:
        DataFrame with columns: timestamp, wind_speed, wind_direction,
        wind_gust, temperature, cardinal_direction
    """
    if response.is_empty:
        logger.warning("Empty response - returning empty DataFrame")
        return pd.DataFrame(columns=[
            "timestamp", "wind_speed", "wind_direction", 
            "wind_gust", "temperature", "cardinal_direction"
        ])
    
    # Extract data from observations
    data = []
    for obs in response.observations:
        data.append({
            "timestamp": obs.timestamp,
            "wind_speed": obs.wind_speed,
            "wind_direction": obs.wind_direction,
            "wind_gust": obs.wind_gust,
            "temperature": obs.temperature,
            "cardinal_direction": obs.get_cardinal_direction(),
        })
    
    df = pd.DataFrame(data)
    
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Add metadata as attributes
    df.attrs["latitude"] = response.location.latitude
    df.attrs["longitude"] = response.location.longitude
    df.attrs["units"] = response.units
    df.attrs["source"] = response.source
    
    logger.debug(f"Created DataFrame with {len(df)} rows")
    
    return df


# =============================================================================
# Unit Conversion
# =============================================================================

def convert_wind_speed(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert wind speed between units.
    
    Supported units: mph, m/s, km/h (kph), knots (kts)
    
    Args:
        value: Wind speed value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted wind speed value
        
    Raises:
        ValueError: If unit not supported
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if from_unit not in WIND_SPEED_TO_MS:
        raise ValueError(f"Unknown source unit: {from_unit}. Supported: {list(WIND_SPEED_TO_MS.keys())}")
    if to_unit not in WIND_SPEED_TO_MS:
        raise ValueError(f"Unknown target unit: {to_unit}. Supported: {list(WIND_SPEED_TO_MS.keys())}")
    
    if from_unit == to_unit:
        return value
    
    if math.isnan(value):
        return value
    
    # Convert to m/s first, then to target
    value_ms = value * WIND_SPEED_TO_MS[from_unit]
    value_target = value_ms / WIND_SPEED_TO_MS[to_unit]
    
    return value_target


def convert_dataframe_units(
    df: pd.DataFrame,
    to_unit: str,
    from_unit: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert wind speed columns in DataFrame to different units.
    
    Converts both wind_speed and wind_gust columns.
    
    Args:
        df: DataFrame with wind data
        to_unit: Target unit
        from_unit: Source unit (uses DataFrame attrs or 'mph' if not specified)
        
    Returns:
        DataFrame with converted values (copy)
    """
    df = df.copy()
    
    # Determine source unit
    if from_unit is None:
        from_unit = df.attrs.get("units", "imperial")
        if from_unit == "imperial":
            from_unit = "mph"
        elif from_unit == "metric":
            from_unit = "m/s"
    
    # Convert wind_speed
    if "wind_speed" in df.columns:
        df["wind_speed"] = df["wind_speed"].apply(
            lambda x: convert_wind_speed(x, from_unit, to_unit) if pd.notna(x) else x
        )
    
    # Convert wind_gust
    if "wind_gust" in df.columns:
        df["wind_gust"] = df["wind_gust"].apply(
            lambda x: convert_wind_speed(x, from_unit, to_unit) if pd.notna(x) else x
        )
    
    # Update attrs
    df.attrs["units"] = to_unit
    
    return df


def get_unit_label(unit: str) -> str:
    """Get display label for unit."""
    return UNIT_LABELS.get(unit.lower(), unit)
