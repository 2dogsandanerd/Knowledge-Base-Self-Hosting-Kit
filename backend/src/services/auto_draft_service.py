import redis
import json
import os
from datetime import datetime
from loguru import logger

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

class AutoDraftService:
    """
    Auto-draft service with Redis-based distributed locking.
    Adapted from Streamlit (global variable) to FastAPI (Redis state).
    """

    def __init__(self, config_override: dict = None):
        self.config = config_override or {}
        self.check_interval = 300  # 5 minutes default

        # Redis client
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None

        # Redis keys
        self.service_key = "autodraft:service:status"
        self.lock_key = "autodraft:service:lock"

    def start_monitoring(self, check_interval: int = 300) -> bool:
        """
        Start auto-draft monitoring with distributed lock.
        Returns True if started, False if already running in another worker.
        """
        if not self.redis_client:
            logger.error("Redis not available, cannot start auto-draft service")
            return False

        # Try to acquire distributed lock
        lock_acquired = self.redis_client.setnx(
            self.lock_key,
            datetime.utcnow().isoformat()
        )

        if not lock_acquired:
            logger.warning("Auto-draft service already running in another worker")
            return False

        # Set lock expiration (10 minutes)
        self.redis_client.expire(self.lock_key, 600)

        # Store service status
        status_data = {
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "check_interval": check_interval,
            "worker_id": os.getpid()
        }
        self.redis_client.setex(
            self.service_key,
            3600,  # 1 hour TTL
            json.dumps(status_data)
        )

        self.check_interval = check_interval
        logger.success(f"Auto-draft service started (PID: {os.getpid()}, interval: {check_interval}s)")
        return True

    def stop_monitoring(self) -> bool:
        """Stop auto-draft monitoring"""
        if not self.redis_client:
            return False

        # Release lock
        self.redis_client.delete(self.lock_key)

        # Update status
        status_data = {
            "status": "stopped",
            "stopped_at": datetime.utcnow().isoformat()
        }
        self.redis_client.setex(
            self.service_key,
            3600,
            json.dumps(status_data)
        )

        logger.info("Auto-draft service stopped")
        return True

    def get_status(self) -> dict:
        """Get service status from Redis"""
        if not self.redis_client:
            return {"status": "unavailable", "error": "Redis not connected"}

        data = self.redis_client.get(self.service_key)
        if data:
            return json.loads(data)

        return {"status": "stopped"}

    @property
    def is_running(self) -> bool:
        """Check if service is running (across all workers)"""
        status = self.get_status()
        return status.get("status") == "running"

    def monitor_emails(self):
        """
        Background task to monitor emails and generate drafts.
        This would be called by FastAPI BackgroundTasks.
        """
        # TODO: Implement actual email monitoring logic
        # This is a placeholder for the background task
        logger.info("Email monitoring task started")
        pass
