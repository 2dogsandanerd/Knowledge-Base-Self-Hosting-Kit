import logging
import json
from datetime import datetime
import os

# Ensure log directory exists
LOG_DIR = os.getenv('LOG_DIR', '/var/log/app')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'backend.log')

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for Loki ingestion."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

# Configure root logger
handler = logging.FileHandler(LOG_FILE)
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Export logger for import elsewhere
logger = logging.getLogger(__name__)
