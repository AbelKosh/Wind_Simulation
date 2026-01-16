"""
Weather Company API client.

This module provides the main interface for fetching wind data from
the Weather Company (IBM) API with proper error handling, rate limiting,
caching, and data validation.

Usage:
    from src.api.weather_client import WeatherClient
    
    client = WeatherClient()
    response = client.get_historical_wind(
        latitude=35.2585,
        longitude=-75.5277,
        start_date=date(2026, 1, 10),
        end_date=date(2026, 1, 12)
    )
"""

import logging
import time
import random
import math
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import get_settings, Settings
from src.api.cache import CacheManager
from src.data.models import (
    WindObservation,
    WindDataResponse,
    Location,
    TimeRange,
    DataQualityReport,
    StormReport,
    SevereWeatherResponse,
    TornadoPath,
    TornadoPathPoint,
    StormEventType,
)

logger = logging.getLogger(__name__)


class WeatherAPIError(Exception):
    """Base exception for Weather API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class RateLimitError(WeatherAPIError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(WeatherAPIError):
    """Raised when API authentication fails."""
    pass


class InvalidLocationError(WeatherAPIError):
    """Raised when location coordinates are invalid."""
    pass


class WeatherClient:
    """
    Client for the Weather Company API.
    
    Handles authentication, rate limiting, caching, and data validation
    for fetching historical wind data.
    
    Attributes:
        api_key: Weather Company API key
        base_url: API base URL
        session: Requests session with retry configuration
        cache: Cache manager for storing responses
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
        cache_dir: Optional[Path] = None,
        demo_mode: Optional[bool] = None,
    ):
        """
        Initialize the Weather API client.
        
        Args:
            api_key: API key (uses settings if not provided)
            settings: Settings object (uses default if not provided)
            cache_dir: Override cache directory
            demo_mode: Force demo mode if True
        """
        self.settings = settings or get_settings()
        
        # Determine demo mode
        if demo_mode is True:
            self.demo_mode = True
            self.api_key = "DEMO_MODE"
        elif demo_mode is False:
            self.demo_mode = False
            # Get API key
            if api_key:
                self.api_key = api_key
            else:
                self.api_key = self.settings.get_api_key()
        else:
            # Auto-detect based on settings
            if self.settings.demo_mode:
                self.demo_mode = True
                self.api_key = "DEMO_MODE"
            elif api_key:
                self.api_key = api_key
                self.demo_mode = False
            else:
                self.api_key = self.settings.get_api_key()
                self.demo_mode = not self.settings.has_api_key
        
        self.base_url = self.settings.weather_api_base_url.rstrip("/")
        
        # Initialize rate limiter
        self._request_times: deque = deque(maxlen=self.settings.max_requests_per_minute)
        
        # Initialize session with retry logic
        self.session = self._create_session()
        
        # Initialize cache manager
        self.cache = CacheManager()
        
        logger.info(f"WeatherClient initialized (demo_mode={self.demo_mode})")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry configuration."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        max_requests = self.settings.max_requests_per_minute
        
        # Remove requests older than 1 minute
        while self._request_times and self._request_times[0] < now - 60:
            self._request_times.popleft()
        
        # Wait if at limit
        if len(self._request_times) >= max_requests:
            sleep_time = 60 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
    
    def _record_request(self) -> None:
        """Record a request for rate limiting."""
        self._request_times.append(time.time())
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        retries: int = 3,
    ) -> dict:
        """
        Make an API request with rate limiting and error handling.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            retries: Number of retries for transient errors
            
        Returns:
            JSON response as dictionary
            
        Raises:
            WeatherAPIError: For API errors
            AuthenticationError: For auth failures
            RateLimitError: For rate limit exceeded
        """
        # Check rate limit
        self._wait_for_rate_limit()
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Add API key to params
        params["apiKey"] = self.api_key
        
        # Log request (sanitize API key)
        safe_params = {k: v for k, v in params.items() if k != "apiKey"}
        logger.debug(f"API Request: {endpoint} params={safe_params}")
        
        last_error = None
        for attempt in range(retries):
            try:
                self._record_request()
                
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.settings.request_timeout,
                )
                
                # Handle errors
                self._handle_error(response)
                
                # Success
                data = response.json()
                logger.debug(f"API Response: {len(str(data))} bytes")
                return data
                
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                
                if attempt < retries - 1:
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
        
        raise WeatherAPIError(f"Request failed after {retries} attempts: {last_error}")
    
    def _handle_error(self, response: requests.Response) -> None:
        """
        Handle HTTP error responses.
        
        Args:
            response: Requests response object
            
        Raises:
            Appropriate exception based on status code
        """
        if response.ok:
            return
        
        status = response.status_code
        
        try:
            error_data = response.json()
        except Exception:
            error_data = {"raw": response.text[:500]}
        
        error_messages = {
            400: "Bad request - check your parameters",
            401: "Authentication failed - check your API key",
            403: "Access forbidden - check your API subscription",
            404: "Location not found - verify coordinates",
            429: "Rate limit exceeded - please wait before retrying",
        }
        
        message = error_messages.get(status, f"API error (HTTP {status})")
        
        if status == 401:
            raise AuthenticationError(message, status, error_data)
        elif status == 429:
            raise RateLimitError(message, status, error_data)
        elif status == 404:
            raise InvalidLocationError(message, status, error_data)
        else:
            raise WeatherAPIError(message, status, error_data)
    
    def _validate_response(self, raw_data: dict) -> DataQualityReport:
        """
        Validate API response and generate quality report.
        
        Args:
            raw_data: Raw API response
            
        Returns:
            DataQualityReport with validation results
        """
        report = DataQualityReport()
        observations = raw_data.get("observations", [])
        report.total_observations = len(observations)
        
        for obs in observations:
            # Check wind speed
            wspd = obs.get("wspd")
            if wspd is None:
                report.missing_wind_speed += 1
            elif isinstance(wspd, (int, float)) and not math.isnan(wspd):
                report.valid_observations += 1
                
                # Check for outliers (> 200 mph is suspicious)
                if wspd > 200:
                    report.outliers += 1
                    report.issues.append(f"Outlier wind speed: {wspd} mph at {obs.get('valid_time_gmt')}")
            else:
                report.missing_wind_speed += 1
            
            # Check direction
            wdir = obs.get("wdir")
            if wdir is None:
                report.missing_direction += 1
        
        # Log quality issues
        if not report.is_acceptable:
            logger.warning(f"Data quality below threshold: {report.quality_score:.1f}%")
            for issue in report.issues[:5]:  # Log first 5 issues
                logger.warning(f"  - {issue}")
        
        return report
    
    def _normalize_observation(self, obs: dict) -> WindObservation:
        """
        Normalize a single observation from API format.
        
        Args:
            obs: Raw observation dictionary from API
            
        Returns:
            WindObservation model
        """
        # Parse timestamp
        timestamp_gmt = obs.get("valid_time_gmt")
        if timestamp_gmt:
            timestamp = datetime.utcfromtimestamp(timestamp_gmt)
        else:
            # Fallback to ISO format if available
            timestamp = datetime.fromisoformat(obs.get("timestamp", datetime.utcnow().isoformat()))
        
        # Get wind data with proper null handling
        wind_speed = obs.get("wspd")
        if wind_speed is None:
            wind_speed = float('nan')
        
        wind_direction = obs.get("wdir")
        wind_gust = obs.get("gust")
        temperature = obs.get("temp")
        cardinal = obs.get("wdir_cardinal")
        
        return WindObservation(
            timestamp=timestamp,
            wind_speed=float(wind_speed) if wind_speed is not None else float('nan'),
            wind_direction=float(wind_direction) if wind_direction is not None else None,
            wind_gust=float(wind_gust) if wind_gust is not None else None,
            temperature=float(temperature) if temperature is not None else None,
            cardinal_direction=cardinal,
        )
    
    def _generate_demo_data(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> WindDataResponse:
        """
        Generate realistic demo data for testing without API.
        
        Creates synthetic wind data with realistic patterns:
        - Spatial coherence (nearby locations have similar wind)
        - Coastal enhancement (stronger winds near coasts)
        - Diurnal variation (stronger afternoon winds)
        - Random variation
        - Occasional gusts
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date
            end_date: End date
            
        Returns:
            WindDataResponse with synthetic data
        """
        logger.info("Generating demo data (no API call)")
        
        observations = []
        current = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.max.time())
        
        # Create a deterministic seed based on location for spatial coherence
        # Use a hash that creates smooth spatial variation
        loc_seed = int(abs(latitude * 100) * 1000 + abs(longitude * 100)) % (2**31)
        random.seed(loc_seed)
        
        # Spatial wind pattern using simplified Perlin-like noise
        # This creates a smooth spatial field that varies across geography
        spatial_wind_factor = self._get_spatial_wind_factor(latitude, longitude, start_date)
        
        # Base wind speed with spatial variation (8-18 mph range for typical day)
        base_wind = 8 + 10 * spatial_wind_factor
        
        # Coastal enhancement: Stronger winds near the coast (approximated by longitude for East Coast)
        # East coast: stronger winds as longitude increases (closer to -75° = coast)
        if longitude > -85 and longitude < -70:  # East coast region
            coast_factor = 1 + 0.3 * (1 - abs(longitude + 75) / 10)
            coast_factor = max(1.0, min(1.4, coast_factor))
        else:
            coast_factor = 1.0
        
        base_wind *= coast_factor
        
        # Add regional weather pattern (simulate a low pressure system)
        # Creates a "storm center" that affects nearby points (more subtle effect)
        storm_lat, storm_lon = self._get_storm_center(start_date)
        dist_to_storm = math.sqrt((latitude - storm_lat)**2 + (longitude - storm_lon)**2)
        storm_factor = max(0.7, min(1.5, 1.5 - dist_to_storm / 5))  # More subtle storm effect
        
        while current <= end:
            hour = current.hour
            
            # Diurnal pattern: stronger in afternoon (peak around 3 PM)
            diurnal_factor = 1.0 + 0.25 * math.sin((hour - 6) * math.pi / 12)
            
            # Random hourly variation (smaller for spatial coherence)
            random_factor = random.uniform(0.9, 1.1)
            
            # Calculate wind speed
            wind_speed = base_wind * diurnal_factor * random_factor * storm_factor
            
            # Add occasional stronger gusts/events
            if random.random() < 0.03:  # 3% chance
                wind_speed *= random.uniform(1.2, 1.5)
            
            # Ensure reasonable range (typical non-storm day: 5-40 mph)
            wind_speed = max(3, min(45, wind_speed))
            
            # Wind direction - prevailing direction varies by region
            # Outer banks: Predominantly SW to NE winds
            base_direction = 225 + random.uniform(-30, 30)  # SW-ish
            
            # Storm influence on direction
            if dist_to_storm < 5:
                # Wind spirals around low pressure (counterclockwise in NH)
                angle_to_storm = math.atan2(storm_lat - latitude, storm_lon - longitude)
                storm_dir = math.degrees(angle_to_storm) + 90  # Perpendicular, counterclockwise
                base_direction = storm_dir + random.uniform(-20, 20)
            
            direction = base_direction % 360
            
            # Gust (usually 1.2-1.5x sustained, higher during storms)
            gust_multiplier = random.uniform(1.2, 1.6) if dist_to_storm > 3 else random.uniform(1.4, 2.0)
            if random.random() < 0.85:  # 85% have gust data
                gust = wind_speed * gust_multiplier
            else:
                gust = None
            
            # Temperature (cooler near coast, varies with wind)
            temp_base = 55 - (latitude - 35) * 3  # Cooler further north
            temp_base -= coast_factor * 5  # Cooler near coast
            temp_diurnal = 10 * math.sin((hour - 6) * math.pi / 12)  # Warmer afternoon
            temperature = temp_base + temp_diurnal + random.uniform(-3, 3)
            
            observations.append(WindObservation(
                timestamp=current,
                wind_speed=round(wind_speed, 1),
                wind_direction=round(direction, 0),
                wind_gust=round(gust, 1) if gust else None,
                temperature=round(temperature, 1),
            ))
            
            current += timedelta(hours=1)
        
        return WindDataResponse(
            observations=observations,
            location=Location(latitude=latitude, longitude=longitude),
            time_range=TimeRange.from_dates(start_date, end_date),
            units="imperial",
            source="demo_data",
        )
    
    def _get_spatial_wind_factor(self, lat: float, lon: float, target_date: date) -> float:
        """
        Generate a spatially coherent wind factor using simplified noise.
        
        Creates smooth spatial variation so nearby points have similar values.
        """
        # Use sine waves at different frequencies for pseudo-noise
        # This creates a smooth, spatially varying field
        scale1 = 0.5  # Large scale features
        scale2 = 1.5  # Medium scale
        scale3 = 4.0  # Small scale detail
        
        # Date influences the pattern (different each day)
        day_offset = target_date.toordinal() * 0.1
        
        noise = (
            0.4 * math.sin(lat * scale1 + lon * scale1 * 0.7 + day_offset) +
            0.3 * math.sin(lat * scale2 * 1.3 + lon * scale2 + day_offset * 2) +
            0.2 * math.sin(lat * scale3 + lon * scale3 * 0.9 + day_offset * 3) +
            0.1 * math.sin((lat + lon) * scale3 * 1.5)
        )
        
        # Normalize to 0-1 range
        return (noise + 1) / 2
    
    def _get_storm_center(self, target_date: date) -> tuple:
        """
        Get a simulated storm center location for a given date.
        
        Creates a consistent "weather system" that affects the region.
        """
        # Storm center varies by date but is deterministic
        random.seed(target_date.toordinal())
        
        # Storm somewhere in the Southeast US
        storm_lat = random.uniform(33, 37)
        storm_lon = random.uniform(-82, -76)
        
        return storm_lat, storm_lon
    
    def get_historical_wind(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
    ) -> WindDataResponse:
        """
        Fetch historical wind data for a location and time range.
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            start_date: Start date for data
            end_date: End date for data (inclusive)
            use_cache: Whether to use cached data if available
            
        Returns:
            WindDataResponse with observations
            
        Raises:
            WeatherAPIError: For API errors
            ValueError: For invalid parameters
        """
        # Validate inputs
        if not -90 <= latitude <= 90:
            raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180.")
        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")
        
        # Check cache first
        if use_cache:
            cache_key = self.cache.get_cache_key(latitude, longitude, start_date, end_date)
            
            if self.cache.is_cache_valid(cache_key, end_date):
                cached = self.cache.get_cached_data(cache_key)
                if cached and "data" in cached:
                    logger.info(f"Using cached data for {cache_key}")
                    return WindDataResponse.from_dict(cached["data"])
        
        # Demo mode - generate synthetic data
        if self.demo_mode:
            response = self._generate_demo_data(latitude, longitude, start_date, end_date)
            
            # Cache demo data too
            if use_cache:
                self.cache.save_to_cache(cache_key, response.to_dict())
            
            return response
        
        # Make API request
        logger.info(f"Fetching wind data: ({latitude}, {longitude}) from {start_date} to {end_date}")
        
        # Weather Company API endpoint for historical hourly data
        endpoint = "/v3/wx/observations/history/hourly"
        
        params = {
            "geocode": f"{latitude},{longitude}",
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "units": "e" if self.settings.default_units == "imperial" else "m",
            "format": "json",
        }
        
        raw_data = self._make_request(endpoint, params)
        
        # Validate response
        quality_report = self._validate_response(raw_data)
        
        # Parse observations
        observations = []
        for obs in raw_data.get("observations", []):
            try:
                observations.append(self._normalize_observation(obs))
            except Exception as e:
                logger.warning(f"Failed to parse observation: {e}")
        
        # Sort by timestamp
        observations.sort(key=lambda x: x.timestamp)
        
        # Build response
        response = WindDataResponse(
            observations=observations,
            location=Location(latitude=latitude, longitude=longitude),
            time_range=TimeRange.from_dates(start_date, end_date),
            units=self.settings.default_units,
            source="weather_company",
        )
        
        # Cache response
        if use_cache:
            self.cache.save_to_cache(cache_key, response.to_dict())
        
        logger.info(f"Retrieved {response.valid_count}/{response.count} valid observations")
        
        return response
    
    def get_local_storm_reports(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: date,
        end_date: date,
        event_types: Optional[List[str]] = None,
        include_tornado_paths: bool = True,
    ) -> SevereWeatherResponse:
        """
        Fetch Local Storm Reports (LSR) for a bounding box and time range.
        
        Args:
            lat_min: Minimum latitude (southern boundary)
            lat_max: Maximum latitude (northern boundary)
            lon_min: Minimum longitude (western boundary)
            lon_max: Maximum longitude (eastern boundary)
            start_date: Start date for reports
            end_date: End date for reports (inclusive)
            event_types: Filter by event types (tornado, thunderstorm_wind, hail, etc.)
            include_tornado_paths: Whether to include tornado path polygons
            
        Returns:
            SevereWeatherResponse with storm reports
            
        Raises:
            WeatherAPIError: For API errors
        """
        # Demo mode - generate synthetic storm reports
        if self.demo_mode:
            return self._generate_demo_storm_reports(
                lat_min, lat_max, lon_min, lon_max,
                start_date, end_date, event_types, include_tornado_paths
            )
        
        # Weather Company LSR API endpoint
        endpoint = "/v2/alerts/lsr"
        
        params = {
            "geocode": f"{(lat_min + lat_max) / 2},{(lon_min + lon_max) / 2}",
            "bbox": f"{lat_max},{lon_min},{lat_min},{lon_max}",
            "startDate": start_date.strftime("%Y%m%d"),
            "endDate": end_date.strftime("%Y%m%d"),
            "format": "json",
        }
        
        if event_types:
            params["eventType"] = ",".join(event_types)
        
        logger.info(f"Fetching storm reports: ({lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E) "
                   f"from {start_date} to {end_date}")
        
        raw_data = self._make_request(endpoint, params)
        
        # Parse storm reports
        reports = []
        for report_data in raw_data.get("localStormReports", []):
            try:
                report = self._parse_storm_report(report_data, include_tornado_paths)
                if report:
                    reports.append(report)
            except Exception as e:
                logger.warning(f"Failed to parse storm report: {e}")
        
        # Filter by event types if specified
        if event_types:
            reports = [r for r in reports if r.event_type.lower() in [t.lower() for t in event_types]]
        
        response = SevereWeatherResponse(
            reports=reports,
            time_range=TimeRange.from_dates(start_date, end_date),
            bounds={"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        )
        
        logger.info(f"Retrieved {response.count} storm reports ({response.tornado_count} tornadoes)")
        
        return response
    
    def _parse_storm_report(
        self,
        data: dict,
        include_tornado_path: bool = True
    ) -> Optional[StormReport]:
        """
        Parse a single storm report from API response.
        
        Args:
            data: Raw report data from API
            include_tornado_path: Whether to parse tornado path data
            
        Returns:
            StormReport or None if parsing fails
        """
        try:
            # Parse timestamp
            timestamp_str = data.get("eventTime") or data.get("valid_time_gmt")
            if isinstance(timestamp_str, (int, float)):
                timestamp = datetime.utcfromtimestamp(timestamp_str)
            else:
                timestamp = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
            
            # Get location
            lat = float(data.get("lat") or data.get("latitude", 0))
            lon = float(data.get("lon") or data.get("longitude", 0))
            
            # Get event type
            event_type = data.get("eventType", "unknown").lower()
            
            # Parse magnitude
            magnitude = data.get("magnitude")
            if magnitude is not None:
                magnitude = float(magnitude)
            
            # Parse tornado path if applicable
            tornado_path = None
            if include_tornado_path and event_type == "tornado":
                tornado_path = self._parse_tornado_path(data)
            
            return StormReport(
                event_type=event_type,
                latitude=lat,
                longitude=lon,
                timestamp=timestamp,
                magnitude=magnitude,
                magnitude_unit=data.get("magnitudeUnit", "mph"),
                description=data.get("description") or data.get("remarks"),
                source=data.get("source"),
                location_name=data.get("location") or data.get("city"),
                state=data.get("state") or data.get("stateCode"),
                county=data.get("county"),
                tornado_path=tornado_path,
            )
        except Exception as e:
            logger.warning(f"Failed to parse storm report: {e}")
            return None
    
    def _parse_tornado_path(self, data: dict) -> Optional[TornadoPath]:
        """
        Parse tornado path data from storm report.
        
        Args:
            data: Raw report data with tornado path info
            
        Returns:
            TornadoPath or None
        """
        try:
            path_data = data.get("tornadoPath") or data.get("path")
            
            if not path_data:
                # Create simple path from start/end coordinates if available
                start_lat = data.get("startLat") or data.get("lat")
                start_lon = data.get("startLon") or data.get("lon")
                end_lat = data.get("endLat")
                end_lon = data.get("endLon")
                
                path_points = []
                if start_lat and start_lon:
                    path_points.append(TornadoPathPoint(
                        latitude=float(start_lat),
                        longitude=float(start_lon)
                    ))
                if end_lat and end_lon:
                    path_points.append(TornadoPathPoint(
                        latitude=float(end_lat),
                        longitude=float(end_lon)
                    ))
                
                if not path_points:
                    return None
                    
                path_data = {"points": path_points}
            
            # Parse path points
            points = []
            if isinstance(path_data, list):
                for pt in path_data:
                    points.append(TornadoPathPoint(
                        latitude=float(pt.get("lat", pt.get("latitude", 0))),
                        longitude=float(pt.get("lon", pt.get("longitude", 0))),
                        timestamp=pt.get("time")
                    ))
            elif isinstance(path_data, dict) and "points" in path_data:
                points = path_data["points"]
            
            return TornadoPath(
                path_points=points,
                ef_rating=data.get("efRating") or data.get("tornado_f_scale"),
                max_width_yards=float(data["maxWidth"]) if data.get("maxWidth") else None,
                length_miles=float(data["pathLength"]) if data.get("pathLength") else None,
                fatalities=int(data.get("fatalities", 0)),
                injuries=int(data.get("injuries", 0)),
            )
        except Exception as e:
            logger.debug(f"Failed to parse tornado path: {e}")
            return None
    
    def _generate_demo_storm_reports(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: date,
        end_date: date,
        event_types: Optional[List[str]] = None,
        include_tornado_paths: bool = True,
    ) -> SevereWeatherResponse:
        """
        Generate realistic demo storm reports for the January 8-9, 2024 event.
        
        This creates synthetic data matching the real tornado outbreak that
        affected North Carolina on January 9, 2024.
        """
        logger.info("Generating demo storm reports for Jan 8-9, 2024 event")
        
        reports = []
        
        # Real tornado data from January 9, 2024 NC event
        # Source: NWS storm reports and Wikipedia
        nc_tornadoes = [
            {
                "name": "Newton-Catawba EF1",
                "start_lat": 35.657, "start_lon": -81.157,
                "end_lat": 35.72, "end_lon": -80.95,
                "ef_rating": "EF1",
                "timestamp": datetime(2024, 1, 9, 17, 27),  # 17:27 UTC
                "length_miles": 9.02,
                "max_width_yards": 250,
                "fatalities": 1,
                "injuries": 4,
                "description": "High-end EF1 tornado touched down in Catawba County near Claremont. "
                              "Multiple manufactured homes seriously damaged. One fatality."
            },
            {
                "name": "New Bern EF1",
                "start_lat": 35.196, "start_lon": -77.056,
                "end_lat": 35.24, "end_lon": -76.95,
                "ef_rating": "EF1",
                "timestamp": datetime(2024, 1, 9, 1, 3),  # 01:03 UTC (Jan 9)
                "length_miles": 5.02,
                "max_width_yards": 125,
                "fatalities": 0,
                "injuries": 0,
                "description": "Agricultural building had metal roofing torn off. "
                              "Multiple trees snapped along the path."
            },
            {
                "name": "Harkers Island EF1",
                "start_lat": 34.6933, "start_lon": -76.5592,
                "end_lat": 34.70, "end_lon": -76.55,
                "ef_rating": "EF1",
                "timestamp": datetime(2024, 1, 9, 2, 9),  # 02:09 UTC
                "length_miles": 0.23,
                "max_width_yards": 75,
                "fatalities": 0,
                "injuries": 0,
                "description": "Tornadic waterspout moved inland. House suffered major roof damage."
            },
            {
                "name": "Harkers Island EF0",
                "start_lat": 34.6982, "start_lon": -76.579,
                "end_lat": 34.705, "end_lon": -76.57,
                "ef_rating": "EF0",
                "timestamp": datetime(2024, 1, 9, 2, 9),  # 02:09 UTC (simultaneous)
                "length_miles": 0.14,
                "max_width_yards": 60,
                "fatalities": 0,
                "injuries": 0,
                "description": "Second tornadic waterspout. Roof damage to multiple homes."
            },
        ]
        
        # High wind reports (thunderstorm wind)
        high_wind_reports = [
            {"lat": 35.7, "lon": -78.8, "magnitude": 58, "location": "Raleigh", 
             "timestamp": datetime(2024, 1, 9, 14, 30), "description": "Trees down, power outages"},
            {"lat": 35.2, "lon": -80.8, "magnitude": 62, "location": "Charlotte",
             "timestamp": datetime(2024, 1, 9, 16, 45), "description": "Widespread damage"},
            {"lat": 36.1, "lon": -79.8, "magnitude": 55, "location": "Greensboro",
             "timestamp": datetime(2024, 1, 9, 13, 15), "description": "Tree limbs down"},
            {"lat": 35.9, "lon": -78.5, "magnitude": 52, "location": "Durham",
             "timestamp": datetime(2024, 1, 9, 14, 0), "description": "Power lines down"},
            {"lat": 34.8, "lon": -77.4, "magnitude": 65, "location": "Jacksonville",
             "timestamp": datetime(2024, 1, 9, 3, 30), "description": "Significant wind damage"},
            {"lat": 35.5, "lon": -77.0, "magnitude": 48, "location": "Greenville",
             "timestamp": datetime(2024, 1, 9, 2, 0), "description": "Trees and limbs down"},
            {"lat": 34.2, "lon": -77.9, "magnitude": 71, "location": "Wilmington",
             "timestamp": datetime(2024, 1, 9, 4, 15), "description": "Severe wind damage, roof damage"},
        ]
        
        # Only include if within bounds
        for tornado in nc_tornadoes:
            if not (lat_min <= tornado["start_lat"] <= lat_max and 
                    lon_min <= tornado["start_lon"] <= lon_max):
                continue
            
            # Create tornado path
            path = None
            if include_tornado_paths:
                path = TornadoPath(
                    path_points=[
                        TornadoPathPoint(latitude=tornado["start_lat"], longitude=tornado["start_lon"]),
                        TornadoPathPoint(latitude=tornado["end_lat"], longitude=tornado["end_lon"]),
                    ],
                    ef_rating=tornado["ef_rating"],
                    start_time=tornado["timestamp"],
                    end_time=tornado["timestamp"] + timedelta(minutes=8),
                    max_width_yards=tornado["max_width_yards"],
                    length_miles=tornado["length_miles"],
                    fatalities=tornado["fatalities"],
                    injuries=tornado["injuries"],
                )
            
            reports.append(StormReport(
                event_type=StormEventType.TORNADO,
                latitude=tornado["start_lat"],
                longitude=tornado["start_lon"],
                timestamp=tornado["timestamp"],
                magnitude=None,  # Tornadoes use EF rating, not wind speed
                magnitude_unit="ef_scale",
                description=tornado["description"],
                source="NWS Damage Survey",
                location_name=tornado["name"].split()[0],
                state="NC",
                tornado_path=path,
            ))
        
        # Add thunderstorm wind reports
        if event_types is None or "thunderstorm_wind" in [t.lower() for t in event_types]:
            for wind in high_wind_reports:
                if not (lat_min <= wind["lat"] <= lat_max and lon_min <= wind["lon"] <= lon_max):
                    continue
                
                reports.append(StormReport(
                    event_type=StormEventType.THUNDERSTORM_WIND,
                    latitude=wind["lat"],
                    longitude=wind["lon"],
                    timestamp=wind["timestamp"],
                    magnitude=wind["magnitude"],
                    magnitude_unit="mph",
                    description=wind["description"],
                    source="Trained Spotter",
                    location_name=wind["location"],
                    state="NC",
                ))
        
        # Filter by event types
        if event_types:
            type_filter = [t.lower() for t in event_types]
            reports = [r for r in reports if r.event_type.lower() in type_filter]
        
        return SevereWeatherResponse(
            reports=reports,
            time_range=TimeRange.from_dates(start_date, end_date),
            bounds={"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        )

    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection successful
            
        Raises:
            AuthenticationError: If API key is invalid
        """
        if self.demo_mode:
            logger.info("Demo mode - skipping connection test")
            return True
        
        try:
            # Make a simple request to test connectivity
            self.get_historical_wind(
                latitude=35.2585,
                longitude=-75.5277,
                start_date=date.today() - timedelta(days=1),
                end_date=date.today() - timedelta(days=1),
                use_cache=False,
            )
            logger.info("API connection test successful")
            return True
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            raise WeatherAPIError(f"Connection test failed: {e}")


# Module-level convenience function
def get_wind_data(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    api_key: Optional[str] = None,
) -> WindDataResponse:
    """
    Convenience function to fetch wind data.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date
        end_date: End date
        api_key: Optional API key (uses settings if not provided)
        
    Returns:
        WindDataResponse with observations
    """
    client = WeatherClient(api_key=api_key)
    return client.get_historical_wind(latitude, longitude, start_date, end_date)
