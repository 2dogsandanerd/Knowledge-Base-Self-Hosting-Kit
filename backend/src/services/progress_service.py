import collections
from datetime import datetime
from typing import List, Dict, Any

class ProgressManager:
    """Singleton to track system progress and logs for UI feedback."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressManager, cls).__new__(cls)
            cls._instance.logs = collections.deque(maxlen=50)
            cls._instance.current_status = "Idle"
            cls._instance.last_update = datetime.now()
        return cls._instance

    def add_log(self, message: str, level: str = "INFO"):
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })
        self.last_update = datetime.now()

    def set_status(self, status: str):
        """Set the current high-level status."""
        self.current_status = status
        self.add_log(f"System status changed: {status}")

    def get_progress(self) -> Dict[str, Any]:
        """Get the current progress and logs."""
        return {
            "status": self.current_status,
            "logs": list(self.logs),
            "last_update": self.last_update.isoformat()
        }

progress_manager = ProgressManager()
