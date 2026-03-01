from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from src.database.models import CollectionIndexConfig
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime


class CollectionRegistry:
    """
    Zentrale Registry für Collection-Konfigurationen.

    Verwaltet:
    - Index-Strategien pro Collection
    - Embedding-Konfigurationen
    - Structured Data Settings
    - Query-Routing Hints
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def register_collection(
        self,
        collection_name: str,
        index_strategy: str,
        data_type: str,
        **kwargs
    ) -> CollectionIndexConfig:
        """Registriert neue Collection oder updated bestehende"""

        # Prüfe ob schon existiert
        stmt = select(CollectionIndexConfig).where(
            CollectionIndexConfig.collection_name == collection_name
        )
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update
            existing.index_strategy = index_strategy
            existing.data_type = data_type

            for key, value in kwargs.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)

            await self.db.commit()
            await self.db.refresh(existing)
            logger.info(f"Updated collection config: {collection_name}")
            return existing

        else:
            # Create new
            config = CollectionIndexConfig(
                collection_name=collection_name,
                index_strategy=index_strategy,
                data_type=data_type,
                **kwargs
            )
            self.db.add(config)
            await self.db.commit()
            await self.db.refresh(config)

            logger.info(f"Registered new collection: {collection_name} ({index_strategy})")
            return config

    async def get_config(self, collection_name: str) -> Optional[CollectionIndexConfig]:
        """Holt Config für eine Collection"""
        stmt = select(CollectionIndexConfig).where(
            CollectionIndexConfig.collection_name == collection_name
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_configs(self) -> List[CollectionIndexConfig]:
        """Gibt alle Collection-Konfigurationen zurück"""
        stmt = select(CollectionIndexConfig)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_all_enabled_for_drafts(self) -> List[CollectionIndexConfig]:
        """Gibt alle für Draft-Generation aktivierten Collections zurück"""
        stmt = select(CollectionIndexConfig).where(
            CollectionIndexConfig.enabled_for_drafts == True
        ).order_by(
            # Sortiere nach Priority (high > medium > low)
            CollectionIndexConfig.priority.desc()
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_priority(
        self,
        collection_name: str,
        priority: str,
        weight: float = None
    ):
        """Ändert Priority einer Collection"""
        config = await self.get_config(collection_name)

        if config:
            config.priority = priority
            if weight is not None:
                config.weight = weight

            await self.db.commit()
            logger.info(f"Updated priority for {collection_name}: {priority}")

    async def track_usage(
        self,
        collection_name: str,
        relevance_score: float = None
    ):
        """Trackt Nutzung einer Collection"""
        config = await self.get_config(collection_name)

        if config:
            config.usage_count += 1
            config.last_used = datetime.now()

            # Update durchschnittliche Relevanz
            if relevance_score is not None:
                if config.avg_relevance == 0:
                    config.avg_relevance = relevance_score
                else:
                    # Rolling average
                    config.avg_relevance = (
                        config.avg_relevance * 0.9 + relevance_score * 0.1
                    )

            await self.db.commit()

    async def delete_collection(self, collection_name: str):
        """Löscht Collection-Config"""
        config = await self.get_config(collection_name)
        if config:
            await self.db.delete(config)
            await self.db.commit()
            logger.info(f"Deleted collection config: {collection_name}")

    async def toggle_enabled(self, collection_name: str, enabled: bool):
        """Aktiviert/Deaktiviert eine Collection für Draft-Generation"""
        config = await self.get_config(collection_name)
        if config:
            config.enabled_for_drafts = enabled
            await self.db.commit()
            logger.info(f"Set {collection_name} enabled_for_drafts to {enabled}")


# Dependency für FastAPI
def get_collection_registry(db: AsyncSession) -> CollectionRegistry:
    """Dependency für FastAPI"""
    return CollectionRegistry(db)
