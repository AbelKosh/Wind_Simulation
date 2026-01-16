"""
Cache manager for API responses.

This module provides file-based caching for Weather Company API responses
to minimize API calls and enable offline operation.

Cache Strategy:
- Cache key: {lat}_{lon}_{start_date}_{end_date}.json
- Expiry: 24 hours for recent data (< 7 days old), never for historical
- Location: data/cache/ directory
"""

import json
import hashlib
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    File-based cache manager for API responses.
    
    Handles caching of wind data responses to reduce API calls
    and enable offline operation after initial data fetch.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files. Uses settings default if None.
        """
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")
    
    def get_cache_key(
        self, 
        lat: float, 
        lon: float, 
        start_date: date, 
        end_date: date
    ) -> str:
        """
        Generate a unique cache key for the given parameters.
        
        Args:
            lat: Latitude
            lon: Longitude  
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string (used as filename)
        """
        # Round coordinates to 4 decimal places for consistent keys
        lat_str = f"{lat:.4f}".replace("-", "n")
        lon_str = f"{lon:.4f}".replace("-", "n")
        
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        key = f"{lat_str}_{lon_str}_{start_str}_{end_str}"
        
        # Use hash for very long keys (shouldn't happen with normal usage)
        if len(key) > 200:
            key = hashlib.md5(key.encode()).hexdigest()
        
        return key
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get full file path for cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get_cached_data(self, cache_key: str) -> Optional[dict]:
        """
        Retrieve cached data if it exists and is valid.
        
        Args:
            cache_key: Cache key from get_cache_key()
            
        Returns:
            Cached data dictionary, or None if not cached/invalid
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.debug(f"Cache hit: {cache_key}")
            return data
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid cache file {cache_key}: {e}")
            # Remove corrupted cache file
            cache_path.unlink(missing_ok=True)
            return None
        except Exception as e:
            logger.error(f"Error reading cache {cache_key}: {e}")
            return None
    
    def save_to_cache(self, cache_key: str, data: dict) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Cache key from get_cache_key()
            data: Data dictionary to cache
        """
        cache_path = self.get_cache_path(cache_key)
        
        # Add cache metadata
        cache_data = {
            "cached_at": datetime.utcnow().isoformat(),
            "cache_key": cache_key,
            "data": data,
        }
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.debug(f"Cached: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error saving cache {cache_key}: {e}")
    
    def is_cache_valid(
        self, 
        cache_key: str, 
        data_end_date: date,
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if cached data is still valid.
        
        Cache validity rules:
        - Historical data (> 7 days old): Never expires
        - Recent data (< 7 days old): Expires after max_age_hours
        
        Args:
            cache_key: Cache key to check
            data_end_date: End date of the cached data
            max_age_hours: Maximum age in hours for recent data
            
        Returns:
            True if cache is valid, False if expired or doesn't exist
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            cached_at = datetime.fromisoformat(cache_data.get("cached_at", ""))
            
            # Historical data (end date > 7 days ago) never expires
            days_old = (date.today() - data_end_date).days
            if days_old > 7:
                logger.debug(f"Cache valid (historical data): {cache_key}")
                return True
            
            # Recent data expires after max_age_hours
            age = datetime.utcnow() - cached_at
            is_valid = age < timedelta(hours=max_age_hours)
            
            if is_valid:
                logger.debug(f"Cache valid (age={age}): {cache_key}")
            else:
                logger.debug(f"Cache expired (age={age}): {cache_key}")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def get_or_fetch(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
        fetch_func,
    ) -> dict:
        """
        Get data from cache or fetch using provided function.
        
        This is a convenience method that handles the full cache workflow.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            fetch_func: Function to call if cache miss (should return dict)
            
        Returns:
            Data dictionary (from cache or fresh fetch)
        """
        cache_key = self.get_cache_key(lat, lon, start_date, end_date)
        
        # Check cache validity
        if self.is_cache_valid(cache_key, end_date):
            cached = self.get_cached_data(cache_key)
            if cached and "data" in cached:
                return cached["data"]
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {cache_key}")
        data = fetch_func()
        
        # Save to cache
        self.save_to_cache(cache_key, data)
        
        return data
    
    def clear_cache(self) -> int:
        """
        Clear all cached data.
        
        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Could not delete {cache_file}: {e}")
        
        logger.info(f"Cleared {count} cache files")
        return count
    
    def clear_expired(self, max_age_hours: int = 24) -> int:
        """
        Clear only expired cache entries.
        
        Args:
            max_age_hours: Maximum age for recent data
            
        Returns:
            Number of cache files deleted
        """
        count = 0
        now = datetime.utcnow()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                cached_at = datetime.fromisoformat(cache_data.get("cached_at", ""))
                age = now - cached_at
                
                # Only clear if older than max_age_hours
                # (we can't easily determine if it's historical without parsing the data)
                if age > timedelta(hours=max_age_hours * 7):  # 7x safety margin
                    cache_file.unlink()
                    count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {cache_file}: {e}")
        
        logger.info(f"Cleared {count} expired cache files")
        return count
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


# Module-level convenience functions
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create singleton cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_cache_key(lat: float, lon: float, start_date: date, end_date: date) -> str:
    """Generate cache key (convenience function)."""
    return get_cache_manager().get_cache_key(lat, lon, start_date, end_date)


def get_cached_data(cache_key: str) -> Optional[dict]:
    """Get cached data (convenience function)."""
    return get_cache_manager().get_cached_data(cache_key)


def save_to_cache(cache_key: str, data: dict) -> None:
    """Save to cache (convenience function)."""
    get_cache_manager().save_to_cache(cache_key, data)


def is_cache_valid(cache_key: str, data_end_date: date) -> bool:
    """Check cache validity (convenience function)."""
    return get_cache_manager().is_cache_valid(cache_key, data_end_date)


def clear_cache() -> int:
    """Clear all cache (convenience function)."""
    return get_cache_manager().clear_cache()
