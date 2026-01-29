"""
NOAA Storm Events client.

This module provides functionality for downloading, caching, and parsing
NOAA Storm Events bulk CSV files from NCEI (National Centers for Environmental
Information).

Data source: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Note: NOAA Storm Events data has an approximate 120-day publication delay.
For recent events, storm data may be incomplete or unavailable.

Usage:
    from src.api.noaa_client import NOAAStormClient
    
    client = NOAAStormClient()
    response = client.get_storm_reports(
        lat_min=33.8,
        lat_max=36.6,
        lon_min=-84.3,
        lon_max=-75.4,
        start_date=date(2024, 1, 8),
        end_date=date(2024, 1, 9),
    )
"""

import csv
import gzip
import io
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Set

import requests

from config.settings import get_settings
from src.data.models import (
    StormReport,
    SevereWeatherResponse,
    TornadoPath,
    TornadoPathPoint,
    TimeRange,
    StormEventType,
)


logger = logging.getLogger(__name__)


# NOAA Storm Events base URL
NOAA_STORM_EVENTS_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

# Approximate delay in days for NOAA data publication
NOAA_DATA_DELAY_DAYS = 120

# EF rating to estimated wind speed mapping (mph)
# Using midpoint of each EF rating range
EF_WIND_SPEEDS = {
    "EF0": 75,   # 65-85 mph
    "EF1": 98,   # 86-110 mph
    "EF2": 123,  # 111-135 mph
    "EF3": 150,  # 136-165 mph
    "EF4": 183,  # 166-200 mph
    "EF5": 220,  # >200 mph
    "EFU": 65,   # Unknown, use minimum
}

# Legacy F-scale mapping
F_WIND_SPEEDS = {
    "F0": 75,
    "F1": 98,
    "F2": 136,
    "F3": 165,
    "F4": 200,
    "F5": 250,
}

# Event type mapping from NOAA CSV to our internal types
NOAA_EVENT_TYPE_MAP = {
    "tornado": StormEventType.TORNADO,
    "thunderstorm wind": StormEventType.THUNDERSTORM_WIND,
    "tstm wind": StormEventType.THUNDERSTORM_WIND,
    "high wind": StormEventType.HIGH_WIND,
    "hail": StormEventType.HAIL,
    "flash flood": StormEventType.FLASH_FLOOD,
    "funnel cloud": StormEventType.FUNNEL_CLOUD,
    "waterspout": StormEventType.WATERSPOUT,
    "marine thunderstorm wind": StormEventType.MARINE_TSTM_WIND,
    "marine tstm wind": StormEventType.MARINE_TSTM_WIND,
}


class NOAAClientError(Exception):
    """Base exception for NOAA client errors."""
    pass


class NOAADownloadError(NOAAClientError):
    """Raised when downloading NOAA data fails."""
    pass


class NOAAParseError(NOAAClientError):
    """Raised when parsing NOAA data fails."""
    pass


class NOAAStormClient:
    """
    Client for fetching historical storm data from NOAA Storm Events database.
    
    Downloads and caches NOAA Storm Events bulk CSV files, then filters
    and parses events based on query parameters.
    
    Attributes:
        cache_dir: Directory for cached NOAA CSV files
        session: Requests session for downloads
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the NOAA Storm Events client.
        
        Args:
            cache_dir: Override cache directory (defaults to settings.noaa_cache_dir)
        """
        settings = get_settings()
        self.cache_dir = cache_dir or settings.noaa_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session with reasonable timeout
        self.session = requests.Session()
        self.session.timeout = 60  # 60 second timeout for large files
        
        logger.info(f"NOAAStormClient initialized (cache: {self.cache_dir})")
    
    def get_storm_reports(
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
        Fetch storm reports for a bounding box and time range from NOAA data.
        
        Args:
            lat_min: Minimum latitude (southern boundary)
            lat_max: Maximum latitude (northern boundary)
            lon_min: Minimum longitude (western boundary)
            lon_max: Maximum longitude (eastern boundary)
            start_date: Start date for reports
            end_date: End date for reports (inclusive)
            event_types: Filter by event types (tornado, thunderstorm_wind, etc.)
            include_tornado_paths: Whether to include tornado path data
            
        Returns:
            SevereWeatherResponse with storm reports
        """
        # Check for data delay warning
        days_ago = (date.today() - end_date).days
        if days_ago < NOAA_DATA_DELAY_DAYS:
            logger.warning(
                f"NOAA Storm Events data has ~{NOAA_DATA_DELAY_DAYS}-day publication delay. "
                f"Data for {end_date} (only {days_ago} days ago) may be incomplete or unavailable."
            )
        
        # Determine which years we need to fetch
        years_needed = self._get_years_in_range(start_date, end_date)
        logger.info(f"Fetching NOAA storm data for years: {sorted(years_needed)}")
        
        # Download/load data for each year
        all_reports = []
        for year in sorted(years_needed):
            try:
                year_reports = self._get_reports_for_year(
                    year=year,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    start_date=start_date,
                    end_date=end_date,
                    event_types=event_types,
                    include_tornado_paths=include_tornado_paths,
                )
                all_reports.extend(year_reports)
            except NOAAClientError as e:
                logger.warning(f"Failed to get NOAA data for {year}: {e}")
                # Continue with other years
        
        response = SevereWeatherResponse(
            reports=all_reports,
            time_range=TimeRange.from_dates(start_date, end_date),
            bounds={
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        )
        
        logger.info(
            f"Retrieved {response.count} storm reports from NOAA "
            f"({response.tornado_count} tornadoes)"
        )
        
        return response
    
    def _get_years_in_range(self, start_date: date, end_date: date) -> Set[int]:
        """Get set of years that overlap with the date range."""
        years = set()
        current = start_date
        while current <= end_date:
            years.add(current.year)
            # Move to next year
            current = date(current.year + 1, 1, 1)
        return years
    
    def _get_reports_for_year(
        self,
        year: int,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: date,
        end_date: date,
        event_types: Optional[List[str]] = None,
        include_tornado_paths: bool = True,
    ) -> List[StormReport]:
        """
        Get storm reports for a specific year, downloading if necessary.
        
        Args:
            year: Year to fetch data for
            lat_min, lat_max, lon_min, lon_max: Bounding box
            start_date, end_date: Date range filter
            event_types: Event type filter
            include_tornado_paths: Whether to parse tornado paths
            
        Returns:
            List of StormReport objects
        """
        # Ensure we have the data file
        csv_path = self._ensure_year_data(year)
        if csv_path is None:
            return []
        
        # Parse and filter the CSV
        return self._parse_csv_file(
            csv_path=csv_path,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            start_date=start_date,
            end_date=end_date,
            event_types=event_types,
            include_tornado_paths=include_tornado_paths,
        )
    
    def _ensure_year_data(self, year: int) -> Optional[Path]:
        """
        Ensure we have the NOAA data file for the given year.
        
        Downloads if not cached. Returns path to the cached file.
        
        Args:
            year: Year to fetch
            
        Returns:
            Path to cached CSV file, or None if unavailable
        """
        # Check cache first
        cached_path = self._get_cached_file(year)
        if cached_path:
            logger.debug(f"Using cached NOAA data for {year}: {cached_path}")
            return cached_path
        
        # Download the file
        return self._download_year_data(year)
    
    def _get_cached_file(self, year: int) -> Optional[Path]:
        """
        Check if we have a cached file for the given year.
        
        Returns the path if found, None otherwise.
        """
        # Look for any matching file (the suffix varies by NOAA update date)
        pattern = f"StormEvents_details-ftp_v1.0_d{year}_*.csv"
        matches = list(self.cache_dir.glob(pattern))
        
        if matches:
            # Return the most recent one
            return sorted(matches)[-1]
        
        # Also check for decompressed files
        pattern = f"StormEvents_details-ftp_v1.0_d{year}_*.csv"
        matches = list(self.cache_dir.glob(pattern))
        if matches:
            return sorted(matches)[-1]
        
        return None
    
    def _download_year_data(self, year: int) -> Optional[Path]:
        """
        Download NOAA Storm Events data for a given year.
        
        Args:
            year: Year to download
            
        Returns:
            Path to downloaded/extracted CSV file, or None if failed
        """
        # First, we need to find the actual filename (it includes a creation date)
        filename = self._find_remote_filename(year)
        if not filename:
            logger.warning(f"Could not find NOAA data file for year {year}")
            return None
        
        # Download the gzipped file
        url = f"{NOAA_STORM_EVENTS_BASE_URL}{filename}"
        logger.info(f"Downloading NOAA storm data: {url}")
        
        try:
            response = self.session.get(url, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            raise NOAADownloadError(f"Failed to download {url}: {e}")
        
        # Save and decompress
        gz_path = self.cache_dir / filename
        gz_path.write_bytes(response.content)
        logger.debug(f"Downloaded {len(response.content)} bytes to {gz_path}")
        
        # Decompress to CSV
        csv_filename = filename.replace(".csv.gz", ".csv")
        csv_path = self.cache_dir / csv_filename
        
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as gz_file:
                csv_path.write_text(gz_file.read())
            logger.info(f"Extracted NOAA data to {csv_path}")
        except Exception as e:
            raise NOAAParseError(f"Failed to decompress {gz_path}: {e}")
        
        # Optionally remove the .gz file to save space
        # gz_path.unlink()
        
        return csv_path
    
    def _find_remote_filename(self, year: int) -> Optional[str]:
        """
        Find the actual NOAA filename for a given year.
        
        NOAA files have format: StormEvents_details-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz
        where the 'c' suffix is the creation/update date.
        
        Args:
            year: Year to find
            
        Returns:
            Filename string or None
        """
        try:
            # Fetch the directory listing
            response = self.session.get(NOAA_STORM_EVENTS_BASE_URL, timeout=30)
            response.raise_for_status()
            
            # Parse HTML to find matching files
            # Looking for pattern: StormEvents_details-ftp_v1.0_d{year}_c*.csv.gz
            pattern = rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz'
            matches = re.findall(pattern, response.text)
            
            if matches:
                # Return the most recent version (highest cYYYYMMDD)
                return sorted(matches)[-1]
            
            return None
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch NOAA directory listing: {e}")
            return None
    
    def _parse_csv_file(
        self,
        csv_path: Path,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: date,
        end_date: date,
        event_types: Optional[List[str]] = None,
        include_tornado_paths: bool = True,
    ) -> List[StormReport]:
        """
        Parse a NOAA Storm Events CSV file and filter by parameters.
        
        Args:
            csv_path: Path to the CSV file
            lat_min, lat_max, lon_min, lon_max: Bounding box filter
            start_date, end_date: Date range filter
            event_types: Event type filter (None = all)
            include_tornado_paths: Whether to parse tornado path data
            
        Returns:
            List of StormReport objects matching the filters
        """
        reports = []
        
        # Normalize event types for comparison
        event_types_lower = None
        if event_types:
            event_types_lower = [t.lower().replace("_", " ") for t in event_types]
        
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        report = self._parse_csv_row(
                            row=row,
                            lat_min=lat_min,
                            lat_max=lat_max,
                            lon_min=lon_min,
                            lon_max=lon_max,
                            start_date=start_date,
                            end_date=end_date,
                            event_types_lower=event_types_lower,
                            include_tornado_paths=include_tornado_paths,
                        )
                        if report:
                            reports.append(report)
                    except Exception as e:
                        # Log but continue parsing
                        logger.debug(f"Failed to parse row: {e}")
                        continue
                        
        except Exception as e:
            raise NOAAParseError(f"Failed to parse CSV file {csv_path}: {e}")
        
        return reports
    
    def _parse_csv_row(
        self,
        row: dict,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: date,
        end_date: date,
        event_types_lower: Optional[List[str]],
        include_tornado_paths: bool,
    ) -> Optional[StormReport]:
        """
        Parse a single CSV row into a StormReport.
        
        NOAA Storm Events CSV columns include:
        - BEGIN_YEARMONTH, BEGIN_DAY, BEGIN_TIME: Event start
        - END_YEARMONTH, END_DAY, END_TIME: Event end
        - STATE, STATE_FIPS, CZ_NAME: Location info
        - EVENT_TYPE: Type of event
        - BEGIN_LAT, BEGIN_LON, END_LAT, END_LON: Coordinates
        - TOR_F_SCALE: Tornado rating (EF0-EF5 or F0-F5)
        - TOR_LENGTH, TOR_WIDTH: Tornado dimensions
        - MAGNITUDE, MAGNITUDE_TYPE: Wind speed or hail size
        - INJURIES_DIRECT, INJURIES_INDIRECT: Injury counts
        - DEATHS_DIRECT, DEATHS_INDIRECT: Death counts
        - EVENT_NARRATIVE: Description
        
        Returns:
            StormReport if row matches filters, None otherwise
        """
        # Get event type and check filter
        event_type_raw = row.get("EVENT_TYPE", "").strip().lower()
        
        if event_types_lower and event_type_raw not in event_types_lower:
            return None
        
        # Map to our internal event type
        event_type = NOAA_EVENT_TYPE_MAP.get(event_type_raw, event_type_raw)
        
        # Parse coordinates
        # Use BEGIN_LAT/LON for the primary location
        try:
            lat = float(row.get("BEGIN_LAT") or row.get("LATITUDE") or 0)
            lon = float(row.get("BEGIN_LON") or row.get("LONGITUDE") or 0)
        except (ValueError, TypeError):
            return None
        
        # Check bounding box
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            return None
        
        # Parse date
        try:
            begin_yearmonth = row.get("BEGIN_YEARMONTH", "")
            begin_day = row.get("BEGIN_DAY", "1")
            
            if not begin_yearmonth:
                return None
            
            year = int(begin_yearmonth[:4])
            month = int(begin_yearmonth[4:6])
            day = int(begin_day) if begin_day else 1
            
            event_date = date(year, month, day)
        except (ValueError, TypeError):
            return None
        
        # Check date range
        if not (start_date <= event_date <= end_date):
            return None
        
        # Parse timestamp (combine date with time if available)
        begin_time = row.get("BEGIN_TIME", "0000")
        try:
            hour = int(begin_time[:2]) if len(begin_time) >= 2 else 0
            minute = int(begin_time[2:4]) if len(begin_time) >= 4 else 0
            timestamp = datetime(year, month, day, hour, minute)
        except (ValueError, TypeError):
            timestamp = datetime(year, month, day, 0, 0)
        
        # Parse magnitude
        magnitude = None
        magnitude_unit = None
        
        # For tornadoes, get wind speed from EF/F scale
        if event_type == StormEventType.TORNADO:
            tor_scale = row.get("TOR_F_SCALE", "").strip().upper()
            if tor_scale in EF_WIND_SPEEDS:
                magnitude = EF_WIND_SPEEDS[tor_scale]
                magnitude_unit = "mph"
            elif tor_scale in F_WIND_SPEEDS:
                magnitude = F_WIND_SPEEDS[tor_scale]
                magnitude_unit = "mph"
        else:
            # For other events, use MAGNITUDE field
            mag_str = row.get("MAGNITUDE", "")
            mag_type = row.get("MAGNITUDE_TYPE", "").upper()
            
            if mag_str:
                try:
                    magnitude = float(mag_str)
                    # Determine unit based on type
                    if mag_type in ("EG", "MG", "ES", "MS"):
                        # EG=Estimated Gust, MG=Measured Gust, ES=Estimated Speed, MS=Measured Speed
                        magnitude_unit = "mph"
                    elif mag_type == "":
                        # For thunderstorm wind, assume mph
                        if event_type in (StormEventType.THUNDERSTORM_WIND, StormEventType.HIGH_WIND):
                            magnitude_unit = "mph"
                except (ValueError, TypeError):
                    pass
        
        # Parse tornado path if applicable
        tornado_path = None
        if include_tornado_paths and event_type == StormEventType.TORNADO:
            tornado_path = self._parse_tornado_path_from_row(row)
        
        # Build location name
        location_name = row.get("CZ_NAME", "").strip().title()
        
        # Get state
        state = row.get("STATE", "").strip().upper()
        if len(state) > 2:
            # Some rows have full state name, try to get abbreviation
            state = row.get("STATE_FIPS", state)[:2] if row.get("STATE_FIPS") else state[:2]
        
        return StormReport(
            event_type=event_type,
            latitude=lat,
            longitude=lon,
            timestamp=timestamp,
            magnitude=magnitude,
            magnitude_unit=magnitude_unit,
            description=row.get("EVENT_NARRATIVE", "").strip()[:500] or None,  # Limit length
            source="NOAA Storm Events",
            location_name=location_name or None,
            state=state or None,
            county=row.get("CZ_NAME", "").strip() or None,
            tornado_path=tornado_path,
        )
    
    def _parse_tornado_path_from_row(self, row: dict) -> Optional[TornadoPath]:
        """
        Parse tornado path data from a CSV row.
        
        Args:
            row: CSV row dict
            
        Returns:
            TornadoPath if data available, None otherwise
        """
        try:
            # Get start/end coordinates
            begin_lat = float(row.get("BEGIN_LAT") or 0)
            begin_lon = float(row.get("BEGIN_LON") or 0)
            end_lat = float(row.get("END_LAT") or begin_lat)
            end_lon = float(row.get("END_LON") or begin_lon)
            
            if begin_lat == 0 and begin_lon == 0:
                return None
            
            # Get EF/F rating
            ef_rating = row.get("TOR_F_SCALE", "").strip().upper() or None
            
            # Get dimensions
            try:
                length_miles = float(row.get("TOR_LENGTH") or 0) or None
            except (ValueError, TypeError):
                length_miles = None
            
            try:
                # TOR_WIDTH is in yards
                max_width_yards = float(row.get("TOR_WIDTH") or 0) or None
            except (ValueError, TypeError):
                max_width_yards = None
            
            # Get casualties
            try:
                fatalities = int(row.get("DEATHS_DIRECT") or 0) + int(row.get("DEATHS_INDIRECT") or 0)
            except (ValueError, TypeError):
                fatalities = 0
            
            try:
                injuries = int(row.get("INJURIES_DIRECT") or 0) + int(row.get("INJURIES_INDIRECT") or 0)
            except (ValueError, TypeError):
                injuries = 0
            
            # Create path points
            path_points = [
                TornadoPathPoint(latitude=begin_lat, longitude=begin_lon),
            ]
            
            # Add end point if different from start
            if (end_lat != begin_lat or end_lon != begin_lon) and end_lat != 0:
                path_points.append(TornadoPathPoint(latitude=end_lat, longitude=end_lon))
            
            return TornadoPath(
                path_points=path_points,
                ef_rating=ef_rating,
                max_width_yards=max_width_yards,
                length_miles=length_miles,
                fatalities=fatalities,
                injuries=injuries,
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse tornado path: {e}")
            return None
    
    def clear_cache(self, year: Optional[int] = None) -> int:
        """
        Clear cached NOAA data files.
        
        Args:
            year: Specific year to clear (None = clear all)
            
        Returns:
            Number of files deleted
        """
        if year:
            pattern = f"StormEvents_details*_d{year}_*"
        else:
            pattern = "StormEvents_details*"
        
        count = 0
        for f in self.cache_dir.glob(pattern):
            f.unlink()
            count += 1
            logger.debug(f"Deleted cached file: {f}")
        
        logger.info(f"Cleared {count} cached NOAA files")
        return count
