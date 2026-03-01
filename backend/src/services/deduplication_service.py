import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    ExtractionMetadata, 
    DuplicationAuditLog, 
    DuplicationAuditLogAction
)
from src.models.deduplication_models import (
    DeduplicationPolicy, 
    DuplicationResult, 
    DuplicateReport, 
    MetadataDiff
)
from src.services.version_manager import VersionManager


class DeduplicationService:
    """Policy-based service for handling duplicate file detection and processing."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.version_manager = VersionManager(db_session)

    async def check_and_handle_duplicate(
        self,
        file_hash: str,
        file_path: str,
        metadata: Dict[str, Any],
        policy: DeduplicationPolicy = DeduplicationPolicy.VERSION
    ) -> DuplicationResult:
        """
        Check if file is a duplicate and handle according to policy.
        
        Args:
            file_hash: SHA256 hash of the file content
            file_path: Path to the file being processed
            metadata: Metadata associated with the file
            policy: How to handle duplicates (SKIP, VERSION, REPLACE_IF_NEWER, NOTIFY_ONLY)
            
        Returns:
            DuplicationResult indicating the action taken
        """
        logger.info(f"Checking for duplicate file with hash {file_hash[:8]}... using policy {policy}")

        # Find existing active metadata records with the same hash
        active_records = await self._get_active_metadata_by_hash(file_hash)

        if not active_records:
            # This is not a duplicate, return success result
            logger.debug(f"No existing record found for hash {file_hash[:8]}, not a duplicate")
            return DuplicationResult(
                is_duplicate=False,
                action_taken="NEW_FILE",
                message="File is not a duplicate, proceeding with processing"
            )

        # We have an active record with the same hash - this is a duplicate
        original_metadata = active_records[0]  # Take the first active one
        logger.info(f"Duplicate detected for hash {file_hash[:8]}, original record ID: {original_metadata.id}")

        # Calculate metadata differences
        metadata_diff = await self._calculate_metadata_diff(original_metadata, metadata)

        if policy == DeduplicationPolicy.SKIP:
            # Create audit log and return
            await self.create_audit_entry(
                original_metadata.id,
                file_path,
                file_hash,
                DuplicationAuditLogAction.SKIPPED,
                metadata_diff
            )
            return DuplicationResult(
                is_duplicate=True,
                action_taken="SKIPPED",
                original_metadata_id=original_metadata.id,
                message="Duplicate file skipped based on policy"
            )

        elif policy == DeduplicationPolicy.VERSION:
            # Create a new version of the file
            new_metadata_id = await self.version_manager.create_new_version(
                original_metadata.id, metadata, file_path, file_hash
            )
            await self.create_audit_entry(
                original_metadata.id,
                file_path,
                file_hash,
                DuplicationAuditLogAction.VERSIONED,
                metadata_diff
            )
            version_number = await self.version_manager.get_version_number(new_metadata_id)
            return DuplicationResult(
                is_duplicate=True,
                action_taken="VERSIONED",
                original_metadata_id=original_metadata.id,
                new_metadata_id=new_metadata_id,
                version=version_number,
                message=f"New version created for duplicate file (version {version_number})"
            )

        elif policy == DeduplicationPolicy.REPLACE_IF_NEWER:
            # Compare timestamps to see if the new file is newer than the original
            new_file_mod_time = metadata.get("modified_date")
            original_mod_time = original_metadata.modified_date

            should_replace = False
            if new_file_mod_time and original_mod_time:
                if isinstance(new_file_mod_time, str):
                    new_file_mod_time = datetime.fromisoformat(new_file_mod_time.replace('Z', '+00:00'))
                if isinstance(original_mod_time, str):
                    original_mod_time = datetime.fromisoformat(original_mod_time.replace('Z', '+00:00'))
                
                if new_file_mod_time > original_mod_time:
                    should_replace = True
            elif not original_mod_time and new_file_mod_time:
                # New file has modification time, original doesn't - treat as newer
                should_replace = True
            else:
                # Fallback: check file size if timestamps aren't available
                new_size = metadata.get("file_size")
                original_size = original_metadata.file_size
                if new_size and original_size and new_size != original_size:
                    should_replace = True

            if should_replace:
                # Create a new version (since we want to keep history)
                new_metadata_id = await self.version_manager.create_new_version(
                    original_metadata.id, metadata, file_path, file_hash
                )
                await self.create_audit_entry(
                    original_metadata.id,
                    file_path,
                    file_hash,
                    DuplicationAuditLogAction.REPLACED,
                    metadata_diff
                )
                version_number = await self.version_manager.get_version_number(new_metadata_id)
                return DuplicationResult(
                    is_duplicate=True,
                    action_taken="REPLACED",
                    original_metadata_id=original_metadata.id,
                    new_metadata_id=new_metadata_id,
                    version=version_number,
                    message=f"File replaced with newer version (version {version_number})"
                )
            else:
                # Skip since the new file isn't newer
                await self.create_audit_entry(
                    original_metadata.id,
                    file_path,
                    file_hash,
                    DuplicationAuditLogAction.SKIPPED,
                    metadata_diff
                )
                return DuplicationResult(
                    is_duplicate=True,
                    action_taken="SKIPPED",
                    original_metadata_id=original_metadata.id,
                    message="Duplicate file skipped because it's not newer than original"
                )

        elif policy == DeduplicationPolicy.NOTIFY_ONLY:
            # Only log the duplicate, don't change anything
            await self.create_audit_entry(
                original_metadata.id,
                file_path,
                file_hash,
                DuplicationAuditLogAction.NOTIFIED,
                metadata_diff
            )
            return DuplicationResult(
                is_duplicate=True,
                action_taken="NOTIFIED",
                original_metadata_id=original_metadata.id,
                message="Duplicate file detected and logged, no changes made"
            )
        
        else:
            # Default to SKIP if policy is unknown
            logger.warning(f"Unknown policy {policy}, defaulting to SKIP")
            await self.create_audit_entry(
                original_metadata.id,
                file_path,
                file_hash,
                DuplicationAuditLogAction.SKIPPED,
                metadata_diff
            )
            return DuplicationResult(
                is_duplicate=True,
                action_taken="SKIPPED",
                original_metadata_id=original_metadata.id,
                message="Unknown policy, duplicate skipped"
            )

    async def create_audit_entry(
        self,
        original_metadata_id: int,
        duplicate_file_path: str,
        duplicate_file_hash: str,
        action_taken: DuplicationAuditLogAction,
        metadata_diff: Optional[MetadataDiff] = None
    ) -> DuplicationAuditLog:
        """Create an audit log entry for duplicate detection."""
        audit_entry = DuplicationAuditLog(
            original_metadata_id=original_metadata_id,
            duplicate_file_path=duplicate_file_path,
            duplicate_file_hash=duplicate_file_hash,
            detection_timestamp=datetime.utcnow(),
            action_taken=action_taken,
            user_notified=False,  # This would be set to True when we implement user notifications
            metadata_diff=metadata_diff.dict() if metadata_diff else None
        )
        
        self.db.add(audit_entry)
        await self.db.commit()
        await self.db.refresh(audit_entry)
        
        logger.info(f"Audit entry created: {action_taken} for file {duplicate_file_hash[:8]}...")
        return audit_entry

    async def get_duplicate_report(
        self,
        date_range: Optional[tuple] = None,
        collection: Optional[str] = None
    ) -> List[DuplicateReport]:
        """Get a report of duplicate detection events."""
        query = select(DuplicationAuditLog).join(
            ExtractionMetadata,
            DuplicationAuditLog.original_metadata_id == ExtractionMetadata.id
        )

        # Apply date range filter if provided
        if date_range:
            start_date, end_date = date_range
            query = query.where(
                DuplicationAuditLog.detection_timestamp.between(start_date, end_date)
            )

        # Apply collection filter if provided
        # Note: This would require the collection to be stored in ExtractionMetadata
        # which might need to be added to the schema
        # For now, we'll skip collection filtering if not directly supported

        result = await self.db.execute(query)
        audit_logs = result.scalars().all()

        reports = []
        for log in audit_logs:
            report = DuplicateReport(
                original_file_path=log.original_metadata.file_path,
                duplicate_file_path=log.duplicate_file_path,
                file_hash=log.duplicate_file_hash,
                detection_timestamp=log.detection_timestamp,
                action_taken=log.action_taken.value,
                metadata_diff=log.metadata_diff
            )
            reports.append(report)

        return reports

    async def _get_active_metadata_by_hash(self, file_hash: str) -> List[ExtractionMetadata]:
        """Get all active metadata records with the given file hash."""
        query = select(ExtractionMetadata).where(
            and_(
                ExtractionMetadata.file_hash == file_hash,
                ExtractionMetadata.is_active == True
            )
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def _calculate_metadata_diff(
        self,
        original_metadata: ExtractionMetadata,
        new_metadata: Dict[str, Any]
    ) -> MetadataDiff:
        """Calculate the difference between original and new file metadata."""
        old_values = {
            'file_path': original_metadata.file_path,
            'file_size': original_metadata.file_size,
            'created_date': original_metadata.created_date.isoformat() if original_metadata.created_date else None,
            'modified_date': original_metadata.modified_date.isoformat() if original_metadata.modified_date else None,
            'mime_type': original_metadata.mime_type,
            'language': original_metadata.language,
            'file_hash': original_metadata.file_hash
        }
        
        # Update with values from the extra field if not already in old_values
        if original_metadata.extra:
            for key, value in original_metadata.extra.items():
                if key not in old_values:
                    old_values[key] = value

        new_values = new_metadata.copy()
        
        # Create list of changed fields
        changed_fields = []
        all_keys = set(old_values.keys()).union(set(new_values.keys()))
        
        for key in all_keys:
            old_val = old_values.get(key)
            new_val = new_values.get(key)
            
            if old_val != new_val:
                changed_fields.append(key)

        return MetadataDiff(
            old_values=old_values,
            new_values=new_values,
            changed_fields=changed_fields
        )