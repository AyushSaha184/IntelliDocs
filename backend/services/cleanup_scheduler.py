import threading
import time
from pathlib import Path
from backend.database.models import get_session_local
from backend.services.session_service import get_session_manager, DATA_DIR
from src.modules.QueryCache import get_retrieval_cache, get_llm_cache
from src.utils.Logger import get_logger

logger = get_logger(__name__)


class CleanupScheduler:
    """Periodically cleans up expired sessions (15 min duration OR 30 min idle)."""
    
    def __init__(self, interval_minutes: int = 10):
        self.interval_seconds = interval_minutes * 60
        self._stop_event = threading.Event()
        self._thread = None
    
    def start(self):
        """Start the cleanup scheduler in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Cleanup scheduler already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Cleanup scheduler started (interval: {self.interval_seconds}s)")
    
    def stop(self):
        """Stop the cleanup scheduler."""
        if self._thread is None:
            return
        
        self._stop_event.set()
        self._thread.join(timeout=5)
        logger.info("Cleanup scheduler stopped")
    
    def _run(self):
        """Main cleanup loop."""
        while not self._stop_event.is_set():
            try:
                self._cleanup()
                self._check_disk_space()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            
            # Sleep with interrupt check
            self._stop_event.wait(self.interval_seconds)
    
    def _check_disk_space(self):
        """Log disk space usage warnings."""
        try:
            import shutil
            stat = shutil.disk_usage(str(DATA_DIR))
            available_gb = stat.free / (1024**3)
            used_gb = stat.used / (1024**3)
            total_gb = stat.total / (1024**3)
            
            usage_percent = (used_gb / total_gb) * 100
            
            if available_gb < 2.0:
                logger.warning(
                    f"⚠️ LOW DISK SPACE: {available_gb:.2f}GB available ({usage_percent:.1f}% used). "
                    f"Consider cleaning up old documents."
                )
            elif available_gb < 5.0:
                logger.info(f"Disk space: {available_gb:.2f}GB available ({usage_percent:.1f}% used)")
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
    
    def _cleanup(self):
        """Perform cleanup of inactive sessions and expired cache entries."""
        db = get_session_local()()
        try:
            session_manager = get_session_manager()
            session_manager.cleanup_inactive_sessions(db)
            
            # Delete cache entries older than 30 minutes
            get_retrieval_cache().delete_older_than(30)
            get_llm_cache().delete_older_than(30)
        finally:
            db.close()


# Global scheduler instance
_cleanup_scheduler: CleanupScheduler = None


def start_cleanup_scheduler(interval_minutes: int = 10):
    """Start the global cleanup scheduler."""
    global _cleanup_scheduler
    if _cleanup_scheduler is None:
        _cleanup_scheduler = CleanupScheduler(interval_minutes)
    _cleanup_scheduler.start()


def stop_cleanup_scheduler():
    """Stop the global cleanup scheduler."""
    global _cleanup_scheduler
    if _cleanup_scheduler is not None:
        _cleanup_scheduler.stop()
