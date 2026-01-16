"""
Data models for wind simulation.

This module defines Pydantic models for wind observation data,
locations, time ranges, and API responses.

All models include validation to ensure data integrity.
"""

from datetime import datetime, date
from typing import Optional, List
import math

from pydantic import BaseModel, Field, field_validator, model_validator


class Location(BaseModel):
    """
    Geographic location with latitude and longitude.
    
    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        name: Optional human-readable location name
    """
    
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    name: Optional[str] = Field(default=None, description="Optional location name")
    
    @property
    def geocode(self) -> str:
        """Format as API-compatible geocode string."""
        return f"{self.latitude},{self.longitude}"
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name} ({self.latitude:.4f}, {self.longitude:.4f})"
        return f"({self.latitude:.4f}, {self.longitude:.4f})"


class TimeRange(BaseModel):
    """
    Time range with start and end dates/datetimes.
    
    Attributes:
        start_datetime: Start of the time range
        end_datetime: End of the time range (must be after start)
    """
    
    start_datetime: datetime = Field(..., description="Start of time range")
    end_datetime: datetime = Field(..., description="End of time range")
    
    @model_validator(mode="after")
    def validate_range(self) -> "TimeRange":
        """Ensure end is after start."""
        if self.end_datetime <= self.start_datetime:
            raise ValueError("end_datetime must be after start_datetime")
        return self
    
    @property
    def start_date(self) -> date:
        """Get start as date."""
        return self.start_datetime.date()
    
    @property
    def end_date(self) -> date:
        """Get end as date."""
        return self.end_datetime.date()
    
    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        delta = self.end_datetime - self.start_datetime
        return delta.total_seconds() / 3600
    
    @property
    def duration_days(self) -> float:
        """Get duration in days."""
        return self.duration_hours / 24
    
    @classmethod
    def from_dates(cls, start_date: date, end_date: date) -> "TimeRange":
        """
        Create TimeRange from date objects.
        
        Start will be at 00:00:00, end will be at 23:59:59.
        """
        return cls(
            start_datetime=datetime.combine(start_date, datetime.min.time()),
            end_datetime=datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
        )


class WindObservation(BaseModel):
    """
    Single wind observation at a point in time.
    
    Attributes:
        timestamp: Observation time (UTC)
        wind_speed: Wind speed in configured units (usually mph)
        wind_direction: Wind direction in degrees (0-360, 0=North)
        wind_gust: Optional gust speed
        temperature: Optional temperature
        cardinal_direction: Optional cardinal direction string (N, NE, etc.)
    """
    
    timestamp: datetime = Field(..., description="Observation timestamp (UTC)")
    wind_speed: float = Field(..., ge=0, description="Wind speed")
    wind_direction: Optional[float] = Field(
        default=None, 
        ge=0, 
        le=360, 
        description="Wind direction in degrees (0=North)"
    )
    wind_gust: Optional[float] = Field(default=None, ge=0, description="Wind gust speed")
    temperature: Optional[float] = Field(default=None, description="Temperature")
    cardinal_direction: Optional[str] = Field(default=None, description="Cardinal direction")
    
    @field_validator("wind_speed", mode="before")
    @classmethod
    def handle_null_wind_speed(cls, v):
        """Handle null/None wind speed values."""
        if v is None:
            return 0.0  # Default to 0 for missing wind speed
        if isinstance(v, float) and math.isnan(v):
            return 0.0
        return v
    
    @field_validator("wind_gust", mode="before")
    @classmethod
    def handle_null_wind_gust(cls, v):
        """Handle null/None wind gust values."""
        if v is None:
            return None  # Keep as None for optional field
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
    
    @field_validator("wind_direction", mode="before")
    @classmethod
    def normalize_direction(cls, v):
        """Normalize wind direction to 0-360 range."""
        if v is None:
            return None
        v = float(v)
        if math.isnan(v):
            return None
        # Normalize to 0-360
        return v % 360
    
    @property
    def is_valid(self) -> bool:
        """Check if observation has valid wind data."""
        return self.wind_speed is not None and self.wind_speed >= 0
    
    @property
    def has_gust(self) -> bool:
        """Check if gust data is available."""
        return self.wind_gust is not None and self.wind_gust >= 0
    
    def get_cardinal_direction(self) -> Optional[str]:
        """
        Get cardinal direction from degrees.
        
        Returns 16-point compass direction (N, NNE, NE, etc.)
        """
        if self.cardinal_direction:
            return self.cardinal_direction
        
        if self.wind_direction is None:
            return None
        
        directions = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ]
        index = round(self.wind_direction / 22.5) % 16
        return directions[index]


class WindDataResponse(BaseModel):
    """
    Complete wind data response containing multiple observations.
    
    Attributes:
        observations: List of wind observations
        location: Location for the data
        time_range: Time range covered
        fetch_timestamp: When data was fetched from API
        units: Unit system used (imperial/metric)
        source: Data source identifier
    """
    
    observations: List[WindObservation] = Field(
        default_factory=list,
        description="List of wind observations"
    )
    location: Location = Field(..., description="Data location")
    time_range: TimeRange = Field(..., description="Time range covered")
    fetch_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When data was fetched"
    )
    units: str = Field(default="imperial", description="Unit system")
    source: str = Field(default="weather_company", description="Data source")
    
    @property
    def count(self) -> int:
        """Number of observations."""
        return len(self.observations)
    
    @property
    def valid_count(self) -> int:
        """Number of valid (non-NaN) observations."""
        return sum(1 for obs in self.observations if obs.is_valid)
    
    @property
    def is_empty(self) -> bool:
        """Check if response contains no observations."""
        return self.count == 0
    
    @property
    def has_valid_data(self) -> bool:
        """Check if response contains at least one valid observation."""
        return self.valid_count > 0
    
    def get_max_wind_speed(self) -> Optional[float]:
        """Get maximum wind speed from observations."""
        valid_speeds = [obs.wind_speed for obs in self.observations if obs.is_valid]
        return max(valid_speeds) if valid_speeds else None
    
    def get_max_gust(self) -> Optional[float]:
        """Get maximum gust speed from observations."""
        valid_gusts = [obs.wind_gust for obs in self.observations if obs.has_gust]
        return max(valid_gusts) if valid_gusts else None
    
    def filter_valid(self) -> "WindDataResponse":
        """Return new response with only valid observations."""
        return WindDataResponse(
            observations=[obs for obs in self.observations if obs.is_valid],
            location=self.location,
            time_range=self.time_range,
            fetch_timestamp=self.fetch_timestamp,
            units=self.units,
            source=self.source,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "observations": [obs.model_dump() for obs in self.observations],
            "location": self.location.model_dump(),
            "time_range": {
                "start_datetime": self.time_range.start_datetime.isoformat(),
                "end_datetime": self.time_range.end_datetime.isoformat(),
            },
            "fetch_timestamp": self.fetch_timestamp.isoformat(),
            "units": self.units,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WindDataResponse":
        """Create from dictionary (JSON deserialization)."""
        # Parse time range
        time_range = TimeRange(
            start_datetime=datetime.fromisoformat(data["time_range"]["start_datetime"]),
            end_datetime=datetime.fromisoformat(data["time_range"]["end_datetime"]),
        )
        
        # Parse observations
        observations = []
        for obs_data in data.get("observations", []):
            # Handle timestamp parsing
            if isinstance(obs_data.get("timestamp"), str):
                obs_data["timestamp"] = datetime.fromisoformat(obs_data["timestamp"])
            observations.append(WindObservation(**obs_data))
        
        return cls(
            observations=observations,
            location=Location(**data["location"]),
            time_range=time_range,
            fetch_timestamp=datetime.fromisoformat(data.get("fetch_timestamp", datetime.utcnow().isoformat())),
            units=data.get("units", "imperial"),
            source=data.get("source", "weather_company"),
        )


class StormEventType:
    """Storm event type constants from Local Storm Reports."""
    TORNADO = "tornado"
    THUNDERSTORM_WIND = "thunderstorm_wind"
    HAIL = "hail"
    FLASH_FLOOD = "flash_flood"
    FUNNEL_CLOUD = "funnel_cloud"
    WATERSPOUT = "waterspout"
    HIGH_WIND = "high_wind"
    MARINE_TSTM_WIND = "marine_thunderstorm_wind"


class TornadoPathPoint(BaseModel):
    """
    Single point along a tornado path.
    
    Attributes:
        latitude: Point latitude
        longitude: Point longitude
        timestamp: Optional timestamp at this point
    """
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: Optional[datetime] = None


class TornadoPath(BaseModel):
    """
    Tornado path polygon/polyline.
    
    Attributes:
        path_points: List of points defining the tornado track
        ef_rating: Enhanced Fujita rating (EF0-EF5)
        start_time: Tornado start time
        end_time: Tornado end time
        max_width_yards: Maximum width in yards
        length_miles: Path length in miles
        fatalities: Number of fatalities
        injuries: Number of injuries
    """
    path_points: List[TornadoPathPoint] = Field(default_factory=list)
    ef_rating: Optional[str] = Field(default=None, description="EF0-EF5 rating")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_width_yards: Optional[float] = None
    length_miles: Optional[float] = None
    fatalities: int = 0
    injuries: int = 0
    
    @property
    def start_location(self) -> Optional[TornadoPathPoint]:
        """Get tornado touchdown point."""
        return self.path_points[0] if self.path_points else None
    
    @property
    def end_location(self) -> Optional[TornadoPathPoint]:
        """Get tornado liftoff point."""
        return self.path_points[-1] if self.path_points else None
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Get tornado duration in minutes."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return None


class StormReport(BaseModel):
    """
    Local Storm Report (LSR) from Weather Company API.
    
    Attributes:
        event_type: Type of storm event (tornado, thunderstorm_wind, hail, etc.)
        latitude: Event latitude
        longitude: Event longitude
        timestamp: Event timestamp (UTC)
        magnitude: Event magnitude (wind speed in mph, hail size in inches, etc.)
        magnitude_unit: Unit for magnitude (mph, in, etc.)
        description: Text description of the event
        source: Report source (trained spotter, law enforcement, etc.)
        location_name: Named location (city, county)
        state: State abbreviation
        county: County name
        tornado_path: Optional tornado path data
    """
    event_type: str = Field(..., description="Storm event type")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: datetime = Field(..., description="Event time (UTC)")
    magnitude: Optional[float] = Field(default=None, description="Event magnitude")
    magnitude_unit: Optional[str] = Field(default=None, description="Magnitude unit")
    description: Optional[str] = Field(default=None, description="Event description")
    source: Optional[str] = Field(default=None, description="Report source")
    location_name: Optional[str] = Field(default=None, description="Named location")
    state: Optional[str] = Field(default=None, description="State abbreviation")
    county: Optional[str] = Field(default=None, description="County name")
    tornado_path: Optional[TornadoPath] = Field(default=None, description="Tornado path if applicable")
    
    @property
    def is_tornado(self) -> bool:
        """Check if this is a tornado report."""
        return self.event_type.lower() == StormEventType.TORNADO
    
    @property
    def is_wind_event(self) -> bool:
        """Check if this is any type of wind event."""
        return self.event_type.lower() in [
            StormEventType.TORNADO,
            StormEventType.THUNDERSTORM_WIND,
            StormEventType.HIGH_WIND,
            StormEventType.MARINE_TSTM_WIND,
        ]
    
    @property
    def location(self) -> Location:
        """Get event location as Location object."""
        return Location(
            latitude=self.latitude,
            longitude=self.longitude,
            name=self.location_name
        )


class SevereWeatherResponse(BaseModel):
    """
    Response containing multiple storm reports.
    
    Attributes:
        reports: List of storm reports
        time_range: Time range covered
        bounds: Geographic bounds queried
        fetch_timestamp: When data was fetched
    """
    reports: List[StormReport] = Field(default_factory=list)
    time_range: TimeRange = Field(..., description="Time range covered")
    bounds: Optional[dict] = Field(default=None, description="Geographic bounds")
    fetch_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def count(self) -> int:
        """Total number of reports."""
        return len(self.reports)
    
    @property
    def tornado_count(self) -> int:
        """Number of tornado reports."""
        return sum(1 for r in self.reports if r.is_tornado)
    
    @property
    def wind_event_count(self) -> int:
        """Number of wind-related events."""
        return sum(1 for r in self.reports if r.is_wind_event)
    
    def filter_by_type(self, event_type: str) -> List[StormReport]:
        """Get reports filtered by event type."""
        return [r for r in self.reports if r.event_type.lower() == event_type.lower()]
    
    def filter_by_state(self, state: str) -> List[StormReport]:
        """Get reports filtered by state."""
        return [r for r in self.reports if r.state and r.state.upper() == state.upper()]
    
    def get_tornado_paths(self) -> List[TornadoPath]:
        """Get all tornado paths from tornado reports."""
        return [r.tornado_path for r in self.reports if r.tornado_path is not None]


class DataQualityReport(BaseModel):
    """
    Report on data quality issues.
    
    Attributes:
        total_observations: Total number of observations
        valid_observations: Number with valid wind speed
        missing_wind_speed: Number with missing/NaN wind speed
        missing_direction: Number with missing direction
        outliers: Number of suspected outliers (speed > 200 mph)
        issues: List of specific issue descriptions
    """
    
    total_observations: int = 0
    valid_observations: int = 0
    missing_wind_speed: int = 0
    missing_direction: int = 0
    outliers: int = 0
    issues: List[str] = Field(default_factory=list)
    
    @property
    def quality_score(self) -> float:
        """
        Calculate overall data quality score (0-100).
        
        100 = perfect data, lower = more issues.
        """
        if self.total_observations == 0:
            return 0.0
        
        valid_pct = (self.valid_observations / self.total_observations) * 100
        outlier_penalty = min(self.outliers * 5, 20)  # Max 20% penalty for outliers
        
        return max(0, valid_pct - outlier_penalty)
    
    @property
    def is_acceptable(self) -> bool:
        """Check if data quality is acceptable (>= 80%)."""
        return self.quality_score >= 80
