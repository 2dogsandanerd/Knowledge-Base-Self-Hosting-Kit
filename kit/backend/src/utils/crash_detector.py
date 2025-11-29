"""
ENHANCED Crash Detection System
Comprehensive logging with crash analysis and async debugging.
"""

import sys
import traceback
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
from functools import wraps
import json
import os

class CrashDetector:
    """Enhanced crash detection with full logging and async debugging."""

    def __init__(self, log_dir: str = None):
        # Use project root /logs/ directory
        if log_dir is None:
            # Get project root (4 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent
            self.log_dir = project_root / "logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(exist_ok=True)

        # Last request state (for crash reports)
        self.last_request: Optional[Dict[str, Any]] = None
        self.active_connections = {"chroma": 0, "ollama": 0}
        self.request_history = []  # Track last 10 requests
        self.max_history = 10

        # Setup enhanced logging and archive old logs
        self._setup_logging()
        self.archive_logs()

    def _setup_logging(self):
        """Setup enhanced logging - WARNING, ERROR, CRITICAL with detailed formatting."""

        # Remove default logger
        logger.remove()

        # Console: WARNING and above with color coding
        logger.add(
            sys.stderr,
            level="WARNING",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            colorize=True
        )

        # Main log file (WARNING and above)
        logger.add(
            self.log_dir / "backend.log",
            level="WARNING",
            rotation="50 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | PID:{process} | {message}",
            enqueue=True  # Thread-safe async logging
        )

        # Error log file (only ERRORS)
        logger.add(
            self.log_dir / "backend.error.log",
            level="ERROR",
            rotation="10 MB",
            retention="14 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            backtrace=True,  # Include full traceback
            diagnose=True    # Include variable values
        )

        # Warning log file
        logger.add(
            self.log_dir / "warnings.log",
            level="WARNING",
            rotation="10 MB",
            retention="3 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )

        # Crash log file (critical errors with full trace)
        logger.add(
            self.log_dir / "crashes.log",
            level="CRITICAL",
            rotation="50 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | CRASH | {name}:{function}:{line} | {message}\n{exception}",
            backtrace=True,
            diagnose=True
        )

    def archive_logs(self):
        """Archives old .log and .json files by renaming them with a timestamp."""
        logger.info(f"Archiving old logs from {self.log_dir}...")
        archived_count = 0
        try:
            # Create a timestamp for the archive
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Find all log and json files
            log_files = list(self.log_dir.glob("*.log"))
            json_files = list(self.log_dir.glob("*.json"))

            if not log_files and not json_files:
                logger.info("No old log files to archive.")
                return

            # Create a subdirectory for the archive
            archive_dir = self.log_dir / f"archive_{timestamp}"
            archive_dir.mkdir(exist_ok=True)

            for log_file in log_files + json_files:
                try:
                    new_name = archive_dir / log_file.name
                    log_file.rename(new_name)
                    archived_count += 1
                except OSError as e:
                    logger.warning(f"Could not archive {log_file.name}: {e}")

            if archived_count > 0:
                logger.info(f"Successfully archived {archived_count} old log file(s) to {archive_dir}")

        except OSError as e:
            logger.error(f"Error during log archiving: {e}")

    def track_request(self, method: str, path: str, params: Optional[Dict] = None):
        """Track current request (for crash reports)."""
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "path": path,
            "params": params or {}
        }
        self.last_request = request_info

        # Add to history (keep last 10)
        self.request_history.append(request_info)
        if len(self.request_history) > self.max_history:
            self.request_history.pop(0)

    def track_connection(self, service: str, delta: int):
        """Track active connections (warn if too many)."""
        self.active_connections[service] = max(0, self.active_connections.get(service, 0) + delta)

        # Warn if too many connections
        if self.active_connections[service] > 5:
            logger.warning(
                f"HIGH CONNECTION COUNT: {service} has {self.active_connections[service]} active connections!"
            )

    def log_crash(self, exception: Exception, context: Optional[Dict] = None):
        """Log crash with full context including request history."""
        crash_info = {
            "timestamp": datetime.now().isoformat(),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "last_request": self.last_request,
            "request_history": self.request_history[-10:],  # Last 10 requests
            "active_connections": self.active_connections,
            "context": context or {},
            "traceback": traceback.format_exc(),
            "process_id": os.getpid(),
            "python_version": sys.version
        }

        # Write crash report
        crash_file = self.log_dir / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(crash_file, 'w') as f:
            json.dump(crash_info, f, indent=2)

        # Log as CRITICAL with full details
        logger.critical(
            f"🔥 CRASH DETECTED: {type(exception).__name__}: {exception}\n"
            f"Last request: {self.last_request}\n"
            f"Request history (last {len(self.request_history)}): {self.request_history}\n"
            f"Active connections: {self.active_connections}\n"
            f"Process ID: {os.getpid()}\n"
            f"Crash report saved: {crash_file}"
        )

    def monitor_async_blocking(self, threshold_seconds: float = 3.0):
        """Decorator to detect blocking calls in async code."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                import time
                start = time.time()

                result = await func(*args, **kwargs)

                duration = time.time() - start
                if duration > threshold_seconds:
                    logger.warning(
                        f"SLOW ASYNC CALL: {func.__name__} took {duration:.2f}s (threshold: {threshold_seconds}s)"
                    )

                return result

            return wrapper
        return decorator

    def monitor_sync(self, threshold_seconds: float = 5.0):
        """Decorator to detect slow sync calls."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                import time
                start = time.time()

                result = func(*args, **kwargs)

                duration = time.time() - start
                if duration > threshold_seconds:
                    logger.warning(
                        f"SLOW SYNC CALL: {func.__name__} took {duration:.2f}s (threshold: {threshold_seconds}s)"
                    )

                return result

            return wrapper
        return decorator


# Global instance
crash_detector = CrashDetector()


def catch_crashes(func):
    """Decorator to catch and log all crashes."""

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                crash_detector.log_crash(e, {"function": func.__name__, "args": str(args)[:200]})
                raise

        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                crash_detector.log_crash(e, {"function": func.__name__, "args": str(args)[:200]})
                raise

        return sync_wrapper
