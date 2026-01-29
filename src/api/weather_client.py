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
    ):
        """
        Initialize the Weather API client.
        
        Args:
            api_key: API key (uses settings if not provided)
            settings: Settings object (uses default if not provided)
            cache_dir: Override cache directory
        """
        self.settings = settings or get_settings()
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self.settings.get_api_key()
        
        self.base_url = self.settings.weather_api_base_url.rstrip("/")
        
        # Initialize rate limiter
        self._request_times: deque = deque(maxlen=self.settings.max_requests_per_minute)
        
        # Initialize session with retry logic
        self.session = self._create_session()
        
        # Initialize cache manager
        self.cache = CacheManager()
        
        # Initialize NOAA client for storm reports (lazy import to avoid circular)
        self._noaa_client = None
        
        logger.info("WeatherClient initialized")
    
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
        
        Uses NOAA Storm Events database for historical data. Note that NOAA
        data has an approximate 120-day publication delay.
        
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
        """
        # Use NOAA Storm Events data for historical storm reports
        if self._noaa_client is None:
            from src.api.noaa_client import NOAAStormClient
            self._noaa_client = NOAAStormClient()
        
        return self._noaa_client.get_storm_reports(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            start_date=start_date,
            end_date=end_date,
            event_types=event_types,
            include_tornado_paths=include_tornado_paths,
        )
    
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
    
    def _parse_storm_report_v2(
        self,
        data: dict,
        include_tornado_path: bool = True
    ) -> Optional[StormReport]:
        """
        Parse a single storm report from v2 stormreports API response.
        
        Based on the Weather Company v2/stormreports API format:
        - datetime_gmt: Unix timestamp
        - latitude/longitude: Coordinates
        - event_type: Event type string (e.g., "HAIL", "TORNADO", "TSTM WND DMG")
        - magnitude: Numeric value (string or null)
        - magnitude_units: "INCHES", "MPH", or null
        
        Args:
            data: Raw report data from API
            include_tornado_path: Whether to parse tornado path data
            
        Returns:
            StormReport or None if parsing fails
        """
        try:
            # Parse timestamp from datetime_gmt (Unix epoch)
            datetime_gmt = data.get("datetime_gmt")
            if datetime_gmt:
                timestamp = datetime.utcfromtimestamp(int(datetime_gmt))
            else:
                # Fallback to datetime_local
                local_time = data.get("datetime_local")
                if local_time:
                    timestamp = datetime.fromisoformat(local_time.replace("Z", "+00:00"))
                else:
                    logger.warning("No timestamp found in storm report")
                    return None
            
            # Get location
            lat = float(data.get("latitude", 0))
            lon = float(data.get("longitude", 0))
            
            if lat == 0 and lon == 0:
                logger.warning("Invalid coordinates in storm report")
                return None
            
            # Get event type - normalize to lowercase
            event_type_raw = data.get("event_type", "unknown")
            event_type = self._normalize_event_type(event_type_raw)
            
            # Parse magnitude
            magnitude = data.get("magnitude")
            if magnitude is not None and magnitude != "":
                try:
                    magnitude = float(magnitude)
                except (ValueError, TypeError):
                    magnitude = None
            else:
                magnitude = None
            
            # Get magnitude unit
            magnitude_unit_raw = data.get("magnitude_units", "")
            if magnitude_unit_raw:
                magnitude_unit = "mph" if magnitude_unit_raw.upper() == "MPH" else magnitude_unit_raw.lower()
            else:
                magnitude_unit = "mph"
            
            # Parse tornado path if applicable
            tornado_path = None
            if include_tornado_path and event_type == "tornado":
                tornado_path = self._parse_tornado_path_v2(data)
            
            return StormReport(
                event_type=event_type,
                latitude=lat,
                longitude=lon,
                timestamp=timestamp,
                magnitude=magnitude,
                magnitude_unit=magnitude_unit,
                description=data.get("comments"),
                source=data.get("source") or data.get("bulletin_source"),
                location_name=data.get("location"),
                state=data.get("state_code"),
                county=data.get("geo_name"),
                tornado_path=tornado_path,
            )
        except Exception as e:
            logger.warning(f"Failed to parse v2 storm report: {e}")
            return None
    
    def _normalize_event_type(self, event_type: str) -> str:
        """
        Normalize event type strings from the API to consistent format.
        
        Args:
            event_type: Raw event type string from API
            
        Returns:
            Normalized event type string
        """
        event_type_upper = event_type.upper().strip()
        
        # Map common event types to normalized names
        event_type_map = {
            "TORNADO": "tornado",
            "TSTM WND DMG": "thunderstorm_wind",
            "TSTM WIND DMG": "thunderstorm_wind",
            "TSTM WND GST": "thunderstorm_wind",
            "TSTM WIND GST": "thunderstorm_wind",
            "THUNDERSTORM WIND": "thunderstorm_wind",
            "THUNDERSTORM WIND DAMAGE": "thunderstorm_wind",
            "THUNDERSTORM WIND GUST": "thunderstorm_wind",
            "HAIL": "hail",
            "HIGH WIND": "high_wind",
            "NON-TSTM WND DMG": "high_wind",
            "NON-TSTM WND GST": "high_wind",
            "FLASH FLOOD": "flash_flood",
            "FLOOD": "flood",
            "HEAVY RAIN": "heavy_rain",
            "SNOW": "snow",
            "HEAVY SNOW": "snow",
            "24 HOUR SNOWFALL": "snow",
            "ICE STORM": "ice_storm",
            "WILDFIRE": "wildfire",
        }
        
        return event_type_map.get(event_type_upper, event_type.lower().replace(" ", "_"))
    
    def _parse_tornado_path_v2(self, data: dict) -> Optional[TornadoPath]:
        """
        Parse tornado path data from v2 storm report.
        
        Note: v2 stormreports API typically doesn't include detailed path data,
        so we create a single-point path from the report location.
        
        Args:
            data: Raw report data
            
        Returns:
            TornadoPath or None
        """
        try:
            lat = float(data.get("latitude", 0))
            lon = float(data.get("longitude", 0))
            
            if lat == 0 and lon == 0:
                return None
            
            # Create single-point path from report location
            path_points = [TornadoPathPoint(latitude=lat, longitude=lon)]
            
            # Try to estimate EF rating from magnitude or severity
            ef_rating = None
            magnitude = data.get("magnitude")
            severity = data.get("severity", 5)
            
            # If magnitude looks like an EF rating
            if magnitude:
                mag_str = str(magnitude).upper()
                if mag_str.startswith("EF"):
                    ef_rating = mag_str
                elif mag_str in ["0", "1", "2", "3", "4", "5"]:
                    ef_rating = f"EF{mag_str}"
            
            # Estimate from severity if no EF rating found
            if not ef_rating and severity:
                severity_int = int(severity)
                if severity_int >= 9:
                    ef_rating = "EF3"
                elif severity_int >= 7:
                    ef_rating = "EF2"
                elif severity_int >= 5:
                    ef_rating = "EF1"
                else:
                    ef_rating = "EF0"
            
            return TornadoPath(
                path_points=path_points,
                ef_rating=ef_rating,
                path_length_miles=None,
                path_width_yards=None,
            )
        except Exception as e:
            logger.warning(f"Failed to parse tornado path from v2 data: {e}")
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

    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection successful
            
        Raises:
            AuthenticationError: If API key is invalid
        """
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
