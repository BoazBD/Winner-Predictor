import threading
import time
import pickle
import os
import json
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GameDataCache:
    """Local cache for game predictions that refreshes every 3 hours at scheduled times (00:03, 03:03, 06:03, etc.)."""
    
    # List of datetime fields that need special handling for JSON serialization
    DATETIME_FIELDS = ['match_time', 'prediction_timestamp', 'result_updated_at', 'timestamp', 'last_updated']
    
    # Scheduled refresh times (every 3 hours at 3 minutes past the hour)
    REFRESH_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]  # Hours when refresh should occur
    REFRESH_MINUTE = 3  # Minute past the hour when refresh should occur
    
    def __init__(self, refresh_interval_hours: int = 3):
        self.refresh_interval_hours = refresh_interval_hours  # Keep for backwards compatibility
        self.cache_lock = threading.RLock()
        self.background_thread = None
        self.stop_background_refresh = threading.Event()
        self._startup_time = datetime.now()  # Track when cache was initialized
        
        # Cache data
        self._profitable_games = []
        self._all_predictions = []
        self._metadata = {'leagues': [], 'models': []}
        self._last_refresh = None
        self._is_refreshing = False
        self._initial_refresh_done = False  # Track if we've done the initial refresh
        
        # File paths for persistence
        self.cache_dir = '/tmp/game_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.profitable_games_file = os.path.join(self.cache_dir, 'profitable_games.json')
        self.all_predictions_file = os.path.join(self.cache_dir, 'all_predictions.json')
        self.metadata_file = os.path.join(self.cache_dir, 'metadata.json')
        self.cache_info_file = os.path.join(self.cache_dir, 'cache_info.json')
        
        # Load existing cache if available
        cache_loaded = self._load_cache_from_disk()
        
        # If no valid cache was loaded, we need an initial refresh
        if not cache_loaded or (len(self._profitable_games) == 0 and len(self._all_predictions) == 0):
            logger.info("No valid cached data found - will perform initial refresh")
            self._needs_initial_refresh = True
        else:
            logger.info(f"Loaded valid cache with {len(self._profitable_games)} profitable games and {len(self._all_predictions)} predictions")
            self._needs_initial_refresh = False
            self._initial_refresh_done = True
        
        # Start background refresh
        self.start_background_refresh()
    
    def _convert_datetimes_to_strings(self, games_list):
        """Convert datetime objects to ISO format strings for JSON serialization."""
        serializable_games = []
        for game in games_list:
            game_copy = game.copy()
            for field in self.DATETIME_FIELDS:
                if field in game_copy and isinstance(game_copy[field], datetime):
                    game_copy[field] = game_copy[field].isoformat()
            serializable_games.append(game_copy)
        return serializable_games
    
    def _convert_strings_to_datetimes(self, games_list):
        """Convert ISO format strings back to datetime objects after loading from JSON."""
        for game in games_list:
            for field in self.DATETIME_FIELDS:
                if field in game and isinstance(game[field], str):
                    try:
                        game[field] = datetime.fromisoformat(game[field])
                    except ValueError:
                        # If parsing fails, set to None
                        logger.warning(f"Failed to parse datetime field '{field}': {game[field]}")
                        game[field] = None
        return games_list
    
    def _get_next_refresh_time(self, from_time: datetime = None) -> datetime:
        """Calculate the next scheduled refresh time."""
        if from_time is None:
            from_time = datetime.now()
        
        # Find the next refresh hour
        current_hour = from_time.hour
        current_minute = from_time.minute
        
        # Find the next scheduled hour
        next_hour = None
        for hour in self.REFRESH_HOURS:
            if hour > current_hour or (hour == current_hour and current_minute < self.REFRESH_MINUTE):
                next_hour = hour
                break
        
        # If no hour found today, use first hour of next day
        if next_hour is None:
            next_hour = self.REFRESH_HOURS[0]
            # Move to next day
            next_refresh = from_time.replace(hour=next_hour, minute=self.REFRESH_MINUTE, second=0, microsecond=0) + timedelta(days=1)
        else:
            # Use the found hour today
            next_refresh = from_time.replace(hour=next_hour, minute=self.REFRESH_MINUTE, second=0, microsecond=0)
        
        return next_refresh
    
    def _get_previous_refresh_time(self, from_time: datetime = None) -> datetime:
        """Calculate the previous scheduled refresh time."""
        if from_time is None:
            from_time = datetime.now()
        
        # Find the previous refresh hour
        current_hour = from_time.hour
        current_minute = from_time.minute
        
        # Find the previous scheduled hour
        previous_hour = None
        for hour in reversed(self.REFRESH_HOURS):
            if hour < current_hour or (hour == current_hour and current_minute >= self.REFRESH_MINUTE):
                previous_hour = hour
                break
        
        # If no hour found today, use last hour of previous day
        if previous_hour is None:
            previous_hour = self.REFRESH_HOURS[-1]
            # Move to previous day
            previous_refresh = from_time.replace(hour=previous_hour, minute=self.REFRESH_MINUTE, second=0, microsecond=0) - timedelta(days=1)
        else:
            # Use the found hour today
            previous_refresh = from_time.replace(hour=previous_hour, minute=self.REFRESH_MINUTE, second=0, microsecond=0)
        
        return previous_refresh
    
    def _load_cache_from_disk(self):
        """Load cached data from disk if available and not too old."""
        try:
            if os.path.exists(self.cache_info_file):
                with open(self.cache_info_file, 'r') as f:
                    cache_info = json.load(f)
                    last_refresh_str = cache_info.get('last_refresh')
                    if last_refresh_str:
                        last_refresh = datetime.fromisoformat(last_refresh_str)
                        # During startup, be more lenient about loading cached data
                        # Load cache if it's not too old (within 6 hours) to avoid immediate refresh
                        now = datetime.now()
                        cache_age = now - last_refresh
                        max_cache_age = timedelta(hours=6)  # More lenient than the 3-hour refresh schedule
                        
                        if cache_age <= max_cache_age:
                            self._last_refresh = last_refresh
                            
                            # Load the actual data
                            if os.path.exists(self.profitable_games_file):
                                with open(self.profitable_games_file, 'r') as f:
                                    self._profitable_games = json.load(f)
                                    # Convert datetime strings back to datetime objects
                                    self._convert_strings_to_datetimes(self._profitable_games)
                            
                            if os.path.exists(self.all_predictions_file):
                                with open(self.all_predictions_file, 'r') as f:
                                    self._all_predictions = json.load(f)
                                    # Convert datetime strings back to datetime objects
                                    self._convert_strings_to_datetimes(self._all_predictions)
                            
                            if os.path.exists(self.metadata_file):
                                with open(self.metadata_file, 'r') as f:
                                    self._metadata = json.load(f)
                            
                            logger.info("Cache loaded from disk successfully")
                            return True # Indicate successful load
                        else:
                            logger.info("Cache on disk is too old, will refresh from database")
        except Exception as e:
            logger.error(f"Error loading cache from disk: {e}")
        return False # Indicate failed load
    
    def _save_cache_to_disk(self):
        """Save cached data to disk for persistence across restarts."""
        try:
            # Convert datetime objects to strings for JSON serialization
            profitable_games_serializable = self._convert_datetimes_to_strings(self._profitable_games)
            all_predictions_serializable = self._convert_datetimes_to_strings(self._all_predictions)
            
            # Save data files
            with open(self.profitable_games_file, 'w') as f:
                json.dump(profitable_games_serializable, f)
            
            with open(self.all_predictions_file, 'w') as f:
                json.dump(all_predictions_serializable, f)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f)
            
            # Save cache info
            cache_info = {
                'last_refresh': self._last_refresh.isoformat() if self._last_refresh else None,
                'profitable_games_count': len(self._profitable_games),
                'all_predictions_count': len(self._all_predictions)
            }
            with open(self.cache_info_file, 'w') as f:
                json.dump(cache_info, f)
            
            logger.info("Cache saved to disk successfully")
        except Exception as e:
            logger.error(f"Error saving cache to disk: {e}")
    
    def start_background_refresh(self):
        """Start the background thread that refreshes cache at scheduled times (00:03, 03:03, 06:03, etc.)."""
        if self.background_thread and self.background_thread.is_alive():
            return
        
        self.stop_background_refresh.clear()
        self.background_thread = threading.Thread(target=self._background_refresh_loop, daemon=True)
        self.background_thread.start()
        logger.info("Started background cache refresh thread")
    
    def stop_background_refresh_thread(self):
        """Stop the background refresh thread."""
        self.stop_background_refresh.set()
        if self.background_thread:
            self.background_thread.join(timeout=5)
    
    def _background_refresh_loop(self):
        """Background loop that refreshes cache at scheduled times."""
        logger.info(f"Background refresh thread started at {datetime.now().strftime('%H:%M:%S')}")
        
        while not self.stop_background_refresh.is_set():
            try:
                now = datetime.now()
                
                # Check if cache needs refresh
                needs_refresh = self._needs_refresh()
                if needs_refresh:
                    # Check if this is an initial refresh
                    if hasattr(self, '_needs_initial_refresh') and self._needs_initial_refresh:
                        logger.info(f"Initial refresh triggered at {now.strftime('%H:%M:%S')}")
                    else:
                        logger.info(f"Scheduled refresh triggered at {now.strftime('%H:%M:%S')}")
                    self.refresh_cache()
                else:
                    # Log why we're not refreshing (for debugging startup issues)
                    startup_age = now - self._startup_time
                    if hasattr(self, '_needs_initial_refresh') and self._needs_initial_refresh:
                        logger.debug(f"Waiting for initial refresh - {startup_age.total_seconds():.0f}s since startup (need 5s)")
                    elif startup_age < timedelta(minutes=2) and self._initial_refresh_done:
                        logger.debug(f"Skipping refresh - in startup grace period ({startup_age.total_seconds():.0f}s since startup)")
                    elif self._last_refresh is None:
                        previous_scheduled = self._get_previous_refresh_time(now)
                        time_since_scheduled = now - previous_scheduled
                        logger.debug(f"Skipping refresh - never refreshed, only {time_since_scheduled.total_seconds():.0f}s since scheduled time")
                
                # Calculate how long to wait until the next check/refresh
                if hasattr(self, '_needs_initial_refresh') and self._needs_initial_refresh:
                    # For initial refresh, check every few seconds
                    wait_seconds = 10
                    logger.info(f"Waiting for initial refresh, checking again in {wait_seconds} seconds")
                else:
                    # Calculate how long to wait until the next scheduled refresh
                    next_refresh_time = self._get_next_refresh_time(now)
                    wait_seconds = (next_refresh_time - now).total_seconds()
                    
                    # Ensure we don't wait too long (max 1 hour) and check at least every minute
                    wait_seconds = min(wait_seconds, 3600)  # Max 1 hour wait
                    wait_seconds = max(wait_seconds, 60)    # Min 1 minute wait
                    
                    logger.info(f"Next scheduled refresh at {next_refresh_time.strftime('%Y-%m-%d %H:%M:%S')}, waiting {wait_seconds:.0f} seconds")
                
                # Wait for next refresh or stop signal
                if self.stop_background_refresh.wait(timeout=wait_seconds):
                    break  # Stop signal received
                    
            except Exception as e:
                logger.error(f"Error in background refresh loop: {e}")
                # Wait a bit before trying again
                self.stop_background_refresh.wait(timeout=300)
        
        logger.info("Background refresh thread stopped")
    
    def _needs_refresh(self) -> bool:
        """Check if cache needs to be refreshed based on scheduled times."""
        now = datetime.now()
        
        # If we need an initial refresh and haven't done it yet, do it now
        # (but give the system a few seconds to stabilize first)
        if hasattr(self, '_needs_initial_refresh') and self._needs_initial_refresh:
            startup_age = now - self._startup_time
            if startup_age > timedelta(seconds=5):  # Short grace period for initial refresh
                return True
            else:
                return False
        
        # Don't refresh immediately after startup for regular scheduled refreshes
        startup_grace_period = timedelta(minutes=2)
        if now - self._startup_time < startup_grace_period and self._initial_refresh_done:
            return False
        
        # If we've never refreshed, only refresh if we've been running for a while
        # and we're past a scheduled refresh time
        if self._last_refresh is None:
            # Only refresh if we're at least 5 minutes past a scheduled refresh time
            previous_scheduled_refresh = self._get_previous_refresh_time(now)
            time_since_scheduled = now - previous_scheduled_refresh
            return time_since_scheduled > timedelta(minutes=5)
        
        # Get the most recent scheduled refresh time
        previous_scheduled_refresh = self._get_previous_refresh_time(now)
        
        # If our last refresh was before the most recent scheduled time, we need to refresh
        return self._last_refresh < previous_scheduled_refresh
    
    def refresh_cache(self, force: bool = False):
        """Refresh cache data from the database."""
        if self._is_refreshing and not force:
            logger.info("Cache refresh already in progress, skipping")
            return
        
        with self.cache_lock:
            if self._is_refreshing and not force:
                return
            
            self._is_refreshing = True
            
            try:
                logger.info("Starting cache refresh from database")
                start_time = datetime.now()
                
                # Import here to avoid circular imports
                from db import (
                    get_profitable_games_from_firestore,
                    get_all_predictions_from_firestore,
                    get_prediction_metadata_from_firestore,
                    DATA_SOURCE
                )
                
                # Refresh profitable games
                if DATA_SOURCE == 'firestore':
                    self._profitable_games = get_profitable_games_from_firestore()
                    self._all_predictions = get_all_predictions_from_firestore()
                    leagues, models = get_prediction_metadata_from_firestore()
                    self._metadata = {'leagues': leagues, 'models': models}
                else:
                    logger.warning(f"Cache refresh not implemented for DATA_SOURCE: {DATA_SOURCE}")
                    return
                
                self._last_refresh = datetime.now()
                
                # Mark initial refresh as done if this was the initial refresh
                if hasattr(self, '_needs_initial_refresh') and self._needs_initial_refresh:
                    self._needs_initial_refresh = False
                    self._initial_refresh_done = True
                    logger.info("Initial cache refresh completed - future refreshes will follow schedule")
                
                # Save to disk
                self._save_cache_to_disk()
                
                # Force garbage collection to free memory
                gc.collect()
                
                refresh_duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Cache refresh completed in {refresh_duration:.2f} seconds")
                logger.info(f"Cached {len(self._profitable_games)} profitable games, {len(self._all_predictions)} total predictions")
                
            except Exception as e:
                logger.error(f"Error refreshing cache: {e}")
            finally:
                self._is_refreshing = False
    
    def wait_for_initial_data(self, timeout_seconds: int = 60) -> bool:
        """Wait for initial data to be loaded, with timeout."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            with self.cache_lock:
                # Check if we have data
                has_profitable_games = len(self._profitable_games) > 0
                has_all_predictions = len(self._all_predictions) > 0
                
                # If we have data, we're good
                if has_profitable_games or has_all_predictions:
                    logger.info(f"Initial data loaded: {len(self._profitable_games)} profitable games, {len(self._all_predictions)} predictions")
                    return True
                
                # Check if we're still waiting for initial refresh
                if hasattr(self, '_needs_initial_refresh') and self._needs_initial_refresh:
                    startup_age = datetime.now() - self._startup_time
                    if startup_age < timedelta(seconds=5):
                        logger.debug(f"Waiting for initial refresh grace period: {startup_age.total_seconds():.1f}s")
                    else:
                        logger.info("Waiting for initial refresh to complete...")
                else:
                    # If we don't need initial refresh but have no data, something went wrong
                    # Return False to avoid infinite waiting
                    logger.warning("No initial refresh needed but no data available - returning False")
                    return False
            
            # Wait a bit before checking again
            time.sleep(1)
        
        logger.warning(f"Timeout waiting for initial data after {timeout_seconds} seconds")
        return False
    
    def has_data(self) -> bool:
        """Check if the cache has any data available."""
        with self.cache_lock:
            return len(self._profitable_games) > 0 or len(self._all_predictions) > 0
    
    def get_profitable_games_with_wait(self, timeout_seconds: int = 60) -> List[Dict]:
        """Get profitable games, waiting for initial data if needed."""
        # If we already have data, return immediately
        if self.has_data():
            return self.get_profitable_games()
        
        # Wait for initial data if we don't have any
        data_loaded = self.wait_for_initial_data(timeout_seconds)
        
        # If we still don't have data after waiting, return empty list
        # The calling code can handle this case
        if not data_loaded:
            logger.warning("No data available after waiting - returning empty list")
            return []
        
        return self.get_profitable_games()
    
    def get_all_predictions_with_wait(self, timeout_seconds: int = 60) -> List[Dict]:
        """Get all predictions, waiting for initial data if needed."""
        # If we already have data, return immediately
        if self.has_data():
            return self.get_all_predictions()
        
        # Wait for initial data if we don't have any
        data_loaded = self.wait_for_initial_data(timeout_seconds)
        
        # If we still don't have data after waiting, return empty list
        # The calling code can handle this case
        if not data_loaded:
            logger.warning("No data available after waiting - returning empty list")
            return []
        
        return self.get_all_predictions()
    
    def get_profitable_games(self) -> List[Dict]:
        """Get profitable games from cache."""
        with self.cache_lock:
            if self._needs_refresh() and not self._is_refreshing:
                logger.info("Cache is stale, triggering refresh")
                self.refresh_cache()
            
            return self._profitable_games.copy()
    
    def get_all_predictions(self) -> List[Dict]:
        """Get all predictions from cache."""
        with self.cache_lock:
            if self._needs_refresh() and not self._is_refreshing:
                logger.info("Cache is stale, triggering refresh")
                self.refresh_cache()
            
            return self._all_predictions.copy()
    
    def get_paginated_predictions(self, page: int, per_page: int, league: str = '', 
                                model: str = '', game_id: str = '') -> Tuple[List[Dict], int]:
        """Get paginated predictions from cache with filtering."""
        with self.cache_lock:
            if self._needs_refresh() and not self._is_refreshing:
                self.refresh_cache()
            
            # Apply filters
            filtered_predictions = self._all_predictions.copy()
            
            if game_id:
                filtered_predictions = [p for p in filtered_predictions if p.get('id') == game_id]
            
            if league:
                filtered_predictions = [p for p in filtered_predictions if p.get('league') == league]
            
            if model:
                if '_' in model:
                    # Full model name match
                    filtered_predictions = [p for p in filtered_predictions if p.get('model_name') == model]
                else:
                    # Model type match
                    filtered_predictions = [p for p in filtered_predictions if p.get('model_type') == model]
            
            # Sort by prediction_timestamp descending
            filtered_predictions.sort(
                key=lambda x: x.get('prediction_timestamp', datetime.min), 
                reverse=True
            )
            
            # Apply pagination
            total_count = len(filtered_predictions)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_predictions = filtered_predictions[start_idx:end_idx]
            
            return paginated_predictions, total_count
    
    def get_metadata(self) -> Dict:
        """Get metadata (leagues, models) from cache."""
        with self.cache_lock:
            if self._needs_refresh() and not self._is_refreshing:
                self.refresh_cache()
            
            return self._metadata.copy()
    
    def get_cache_info(self) -> Dict:
        """Get information about the cache state."""
        with self.cache_lock:
            now = datetime.now()
            next_refresh = self._get_next_refresh_time(now)
            previous_refresh = self._get_previous_refresh_time(now)
            
            return {
                'profitable_games_count': len(self._profitable_games),
                'all_predictions_count': len(self._all_predictions),
                'last_refresh': self._last_refresh.isoformat() if self._last_refresh else None,
                'next_scheduled_refresh': next_refresh.isoformat(),
                'previous_scheduled_refresh': previous_refresh.isoformat(),
                'refresh_schedule': f"Every 3 hours at {self.REFRESH_MINUTE:02d} minutes past: {', '.join(f'{h:02d}:03' for h in self.REFRESH_HOURS)}",
                'needs_refresh': self._needs_refresh(),
                'is_refreshing': self._is_refreshing,
                'refresh_interval_hours': self.refresh_interval_hours
            }

# Global cache instance
_cache_instance = None

def get_cache() -> GameDataCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = GameDataCache()
    return _cache_instance

def initialize_cache():
    """Initialize the cache (call this on app startup)."""
    cache = get_cache()
    
    # Perform a synchronous initial refresh so data is available immediately
    try:
        logger.info("Performing initial cache refresh...")
        cache.refresh_cache(force=True)
        logger.info("Initial cache refresh completed")
    except Exception as e:
        logger.error(f"Initial cache refresh failed: {e}")
        # Continue startup – the Flask app will display a loading page if data is still missing
    
    # Background thread will handle scheduled refreshes
    return cache 