from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ExtractionMetadata
from src.models.deduplication_models import DeduplicationPolicy


class VersionInfo:
    """Information about a specific version of a file."""
    def __init__(self, id: int, version: int, is_active: bool, created_at: datetime, 
                 updated_at: datetime, file_path: str):
        self.id = id
        self.version = version
        self.is_active = is_active
        self.created_at = created_at
        self.updated_at = updated_at
        self.file_path = file_path


class VersionManager:
    """Manages file versioning operations."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def should_update_version(
        self, 
        existing_metadata: ExtractionMetadata, 
        new_file_path: str,
        strategy: str = "HASH_COMPARE"
    ) -> bool:
        """
        Determine if a new version should be created based on comparison strategy.
        
        Args:
            existing_metadata: The current active version metadata
            new_file_path: Path to the new file being compared
            strategy: Strategy to use for comparison (HASH_COMPARE, TIMESTAMP_COMPARE, etc.)
            
        Returns:
            True if a new version should be created, False otherwise
        """
        if strategy == "HASH_COMPARE":
            # If we're receiving a file with the same hash, it's not actually different content
            # This should normally not happen since we detect duplicates by hash first
            # So this is more of a safety check
            return False  # Same hash means same content, no need for new version
            
        elif strategy == "TIMESTAMP_COMPARE":
            # Compare modification times to see if new file is more recent
            import os
            new_mod_time = datetime.fromtimestamp(os.path.getmtime(new_file_path))
            
            if existing_metadata.modified_date and new_mod_time > existing_metadata.modified_date:
                return True
            else:
                return False
                
        else:
            # Default to creating new version
            return True

    async def create_new_version(
        self,
        old_metadata_id: int,
        new_metadata: Dict[str, Any],
        new_file_path: str,
        new_file_hash: str
    ) -> int:
        """
        Create a new version of a file, deactivating the old version.
        
        Args:
            old_metadata_id: ID of the metadata record to replace
            new_metadata: Metadata for the new version
            new_file_path: Path to the new file
            new_file_hash: Hash of the new file (should be the same as the old one, but we accept it here)
            
        Returns:
            ID of the new metadata record
        """
        # Get the old metadata to access current values
        old_metadata = await self.db.get(ExtractionMetadata, old_metadata_id)
        if not old_metadata:
            raise ValueError(f"Old metadata with ID {old_metadata_id} not found")

        # Create the new metadata record with incremented version
        new_version = old_metadata.version + 1
        
        # Deactivate the old version
        old_metadata.is_active = False
        old_metadata.superseded_by_id = None  # Will be set to the new version's ID

        # Create new metadata record for the new version
        new_metadata_record = ExtractionMetadata(
            file_hash=new_file_hash,  # Keep the same hash for version history
            file_path=new_file_path,
            file_name=new_metadata.get('file_name', old_metadata.file_name),
            file_size=new_metadata.get('file_size', new_metadata.get('file_size')),
            mime_type=new_metadata.get('mime_type', old_metadata.mime_type),
            created_date=new_metadata.get('created_date', new_metadata.get('created_date')),
            modified_date=new_metadata.get('modified_date', new_metadata.get('modified_date')),
            language=new_metadata.get('language', old_metadata.language),
            ocr_confidence=new_metadata.get('ocr_confidence', old_metadata.ocr_confidence),
            structure_score=new_metadata.get('structure_score', old_metadata.structure_score),
            extra=new_metadata.get('extra', old_metadata.extra),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            
            # Version-specific fields
            version=new_version,
            is_active=True,
            superseded_by_id=None,  # New version doesn't supersede anything (yet)
            first_seen=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )

        # Update the old record to point to the new one
        old_metadata.superseded_by_id = new_metadata_record.id

        # Add the new record and update the old one
        self.db.add(new_metadata_record)
        self.db.add(old_metadata)
        
        await self.db.commit()
        await self.db.refresh(new_metadata_record)
        
        logger.info(f"Created new version {new_version} for file {new_file_hash[:8]}..., superseding version {old_metadata.version}")

        return new_metadata_record.id

    async def get_version_history(self, file_hash: str) -> List[VersionInfo]:
        """
        Get the version history for a file identified by hash.
        
        Args:
            file_hash: SHA256 hash of the file
            
        Returns:
            List of VersionInfo objects, chronologically ordered
        """
        query = select(ExtractionMetadata).where(
            ExtractionMetadata.file_hash == file_hash
        ).order_by(ExtractionMetadata.version.asc())
        
        result = await self.db.execute(query)
        metadata_records = result.scalars().all()
        
        versions = []
        for record in metadata_records:
            version_info = VersionInfo(
                id=record.id,
                version=record.version,
                is_active=record.is_active,
                created_at=record.created_at,
                updated_at=record.updated_at,
                file_path=record.file_path
            )
            versions.append(version_info)
        
        return versions

    async def rollback_to_version(self, metadata_id: int, target_version: int) -> bool:
        """
        Rollback to a specific version by activating it and deactivating others.
        
        Args:
            metadata_id: ID of any metadata record in the version chain
            target_version: Version number to rollback to
            
        Returns:
            True if rollback was successful, False otherwise
        """
        # First, we need to find the file_hash for this metadata chain
        initial_metadata = await self.db.get(ExtractionMetadata, metadata_id)
        if not initial_metadata:
            logger.error(f"Metadata with ID {metadata_id} not found")
            return False

        file_hash = initial_metadata.file_hash

        # Get all versions for this file
        all_versions = await self.db.execute(
            select(ExtractionMetadata).where(
                ExtractionMetadata.file_hash == file_hash
            )
        )
        all_records = all_versions.scalars().all()
        
        target_record = None
        for record in all_records:
            if record.version == target_version:
                target_record = record
                break
        
        if not target_record:
            logger.error(f"Target version {target_version} not found for file {file_hash[:8]}...")
            return False

        # Deactivate all versions
        for record in all_records:
            record.is_active = False
            record.superseded_by_id = target_record.id  # Point to the target as the current version

        # Activate the target version
        target_record.is_active = True
        target_record.superseded_by_id = None

        # Commit all changes
        for record in all_records:
            self.db.add(record)
        
        await self.db.commit()
        
        logger.info(f"Successfully rolled back to version {target_version} for file {file_hash[:8]}...")
        return True

    async def get_version_number(self, metadata_id: int) -> Optional[int]:
        """
        Get the version number for a specific metadata record.
        
        Args:
            metadata_id: ID of the metadata record
            
        Returns:
            Version number or None if not found
        """
        metadata = await self.db.get(ExtractionMetadata, metadata_id)
        if metadata:
            return metadata.version
        return None