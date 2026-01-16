"""
Configuration settings for Wind Simulation.

This module provides type-safe configuration management using Pydantic.
Settings are loaded from environment variables and .env file.

Usage:
    from config.settings import get_settings
    
    settings = get_settings()
    print(settings.weather_api_key)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Get the project root directory (parent of config/)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings with validation.
    
    Settings are loaded from environment variables, with fallback to .env file.
    All paths are relative to the project root unless absolute.
    """
    
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==========================================================================
    # API Configuration
    # ==========================================================================
    weather_api_key: str = Field(
        default="",
        description="Weather Company API key for data access"
    )
    
    weather_api_base_url: str = Field(
        default="https://api.weather.com",
        description="Base URL for Weather Company API"
    )
    
    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    max_requests_per_minute: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum API requests per minute"
    )
    
    max_requests_per_day: int = Field(
        default=50000,
        ge=1,
        description="Maximum API requests per day"
    )
    
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="API request timeout in seconds"
    )
    
    # ==========================================================================
    # Directory Configuration
    # ==========================================================================
    cache_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "cache",
        description="Directory for cached API responses"
    )
    
    export_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "exports",
        description="Directory for exported CSV files"
    )
    
    output_dir: Path = Field(
        default=PROJECT_ROOT / "output",
        description="Directory for generated images"
    )
    
    # ==========================================================================
    # Output Settings
    # ==========================================================================
    default_units: str = Field(
        default="imperial",
        description="Default unit system: 'imperial' (mph) or 'metric' (m/s)"
    )
    
    image_dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for exported images"
    )
    
    # ==========================================================================
    # Grid Resolution
    # ==========================================================================
    default_resolution: float = Field(
        default=0.05,
        ge=0.01,
        le=1.0,
        description="Default grid resolution in degrees (0.05° ≈ 5.5km)"
    )
    
    # ==========================================================================
    # Demo Mode
    # ==========================================================================
    demo_mode: bool = Field(
        default=False,
        description="Use demo data instead of making API calls"
    )
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("default_units")
    @classmethod
    def validate_units(cls, v: str) -> str:
        """Validate unit system."""
        valid_units = {"imperial", "metric"}
        v_lower = v.lower()
        if v_lower not in valid_units:
            raise ValueError(f"Units must be one of: {valid_units}")
        return v_lower
    
    @field_validator("cache_dir", "export_dir", "output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v) -> Path:
        """Convert string paths to Path objects and resolve relative paths."""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            v = PROJECT_ROOT / v
        return v
    
    # ==========================================================================
    # Properties
    # ==========================================================================
    @property
    def has_api_key(self) -> bool:
        """Check if a valid API key is configured."""
        return bool(self.weather_api_key and self.weather_api_key != "your_api_key_here")
    
    @property
    def units_label(self) -> str:
        """Get human-readable units label."""
        return "mph" if self.default_units == "imperial" else "m/s"
    
    # ==========================================================================
    # Methods
    # ==========================================================================
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [self.cache_dir, self.export_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self) -> str:
        """
        Get the API key, raising an error if not configured.
        
        Raises:
            ValueError: If API key is not configured and not in demo mode
        """
        if self.demo_mode:
            return "DEMO_MODE"
        
        if not self.has_api_key:
            raise ValueError(
                "Weather API key not configured!\n"
                "Please set WEATHER_API_KEY in your .env file.\n"
                "See .env.example for instructions.\n"
                "Alternatively, set DEMO_MODE=true to use demo data."
            )
        return self.weather_api_key


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings instance (cached for performance)
    
    Note:
        Uses lru_cache to avoid re-reading .env file on every call.
        Call get_settings.cache_clear() to reload settings if needed.
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Force reload settings from environment.
    
    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# Preset Locations (commonly used coordinates)
# =============================================================================
PRESET_LOCATIONS = {
    "Outer Banks, NC": {"latitude": 35.2585, "longitude": -75.5277},
    "Cape Hatteras, NC": {"latitude": 35.2225, "longitude": -75.6352},
    "Nags Head, NC": {"latitude": 35.9571, "longitude": -75.6241},
    "Kill Devil Hills, NC": {"latitude": 36.0307, "longitude": -75.6760},
    "Kitty Hawk, NC": {"latitude": 36.0726, "longitude": -75.7057},
    "Ocracoke, NC": {"latitude": 35.1146, "longitude": -75.9810},
    "Miami Beach, FL": {"latitude": 25.7907, "longitude": -80.1300},
    "Galveston, TX": {"latitude": 29.3013, "longitude": -94.7977},
}


# =============================================================================
# Wind Speed Thresholds (for visualization and alerts)
# =============================================================================
WIND_THRESHOLDS = {
    "calm": 0,           # 0 mph
    "light": 5,          # 5 mph
    "moderate": 15,      # 15 mph
    "fresh": 25,         # 25 mph
    "strong": 35,        # 35 mph
    "gale": 47,          # 47 mph
    "storm": 55,         # 55 mph
    "violent_storm": 64, # 64 mph
    "hurricane": 74,     # 74 mph (Category 1)
}
