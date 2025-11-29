from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
from src.database.database import get_db
from src.database.models import Setting
import json
import os
from typing import List
import logging

logger = logging.getLogger(__name__)

# Default security settings
DEFAULT_SECURITY_SETTINGS = {
    "allowed_roots": [
        "/mnt/dev/eingang",
        "/home",
    ]
}

class SettingsService:
    """Simple async key‑value store backed by SQLite.
    Uses the ``settings`` table defined in ``src.database.models``.
    """

    async def get(self, key: str) -> str | None:
        async with get_db() as session:  # get_db returns AsyncSession
            stmt = select(Setting.value).where(Setting.key == key)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return row

    async def set(self, key: str, value: str) -> None:
        async with get_db() as session:
            stmt = insert(Setting).values(key=key, value=value).on_conflict_do_update(
                index_elements=[Setting.key], set_={"value": value}
            )
            await session.execute(stmt)
            await session.commit()

    async def get_allowed_roots(self) -> List[str]:
        """
        Get the list of allowed root directories for path validation.

        Returns:
            List of absolute paths that are allowed for file operations.
            Falls back to DEFAULT_SECURITY_SETTINGS if not configured.
        """
        try:
            setting = await self.get("allowed_roots")
            if not setting:
                logger.info("No allowed_roots setting found, using defaults")
                return DEFAULT_SECURITY_SETTINGS["allowed_roots"]

            # Parse JSON string
            paths = json.loads(setting)
            if not isinstance(paths, list):
                logger.warning("allowed_roots is not a list, using defaults")
                return DEFAULT_SECURITY_SETTINGS["allowed_roots"]

            return paths
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to load allowed_roots: {e}, using defaults")
            return DEFAULT_SECURITY_SETTINGS["allowed_roots"]

    async def update_allowed_roots(self, paths: List[str]) -> None:
        """
        Update the list of allowed root directories.

        Args:
            paths: List of absolute directory paths to allow

        Raises:
            ValueError: If any path does not exist or is not a directory
        """
        if not paths:
            raise ValueError("At least one allowed root directory must be specified")

        # Validate all paths exist and are directories
        validated_paths = []
        for path in paths:
            abs_path = os.path.abspath(path)
            if not os.path.isdir(abs_path):
                raise ValueError(f"Path does not exist or is not a directory: {path}")
            validated_paths.append(abs_path)

        # Store as JSON
        await self.set("allowed_roots", json.dumps(validated_paths))
        logger.info(f"Updated allowed_roots to {len(validated_paths)} paths")
