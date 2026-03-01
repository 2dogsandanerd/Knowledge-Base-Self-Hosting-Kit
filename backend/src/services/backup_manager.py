import asyncio
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ExtractionMetadata, ExtractionResultDB, ExtractionBackup, BackupType, BackupStatus
from src.models.deduplication_models import DeduplicationPolicy


class BackupResult:
    """Result of a backup operation."""
    def __init__(self, backup_id: Optional[int], success: bool, message: str, records_count: int = 0):
        self.backup_id = backup_id
        self.success = success
        self.message = message
        self.records_count = records_count


class RestoreResult:
    """Result of a restore operation."""
    def __init__(self, success: bool, message: str, records_restored: int = 0):
        self.success = success
        self.message = message
        self.records_restored = records_restored


class ScheduleInfo:
    """Information about backup scheduling."""
    def __init__(self, enabled: bool, schedule: str, retention_policy: Dict[str, Any]):
        self.enabled = enabled
        self.schedule = schedule
        self.retention_policy = retention_policy


class BackupManager:
    """Manages backup and restore operations for extraction data."""

    def __init__(self, db_session: AsyncSession, backup_path: str = "./data/backups"):
        self.db = db_session
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Default compression settings
        self.compression_enabled = True

    async def create_backup(self, backup_type: BackupType = BackupType.FULL) -> BackupResult:
        """
        Create a backup of extraction metadata and results.
        
        Args:
            backup_type: Type of backup to create (FULL or INCREMENTAL)
            
        Returns:
            BackupResult with information about the operation
        """
        backup_timestamp = datetime.utcnow()
        backup_filename = f"extraction_backup_{backup_timestamp.strftime('%Y%m%d_%H%M%S')}.{backup_type.value.lower()}.json"
        backup_filepath = self.backup_path / backup_filename
        
        try:
            # Create backup record with IN_PROGRESS status
            backup_record = ExtractionBackup(
                backup_timestamp=backup_timestamp,
                backup_type=backup_type,
                records_count=0,  # Will update after backup is complete
                backup_file_path=str(backup_filepath),
                checksum="",  # Will calculate after file is written
                compression="gzip" if self.compression_enabled else "none",
                status=BackupStatus.IN_PROGRESS
            )
            
            self.db.add(backup_record)
            await self.db.commit()
            await self.db.refresh(backup_record)
            
            # Get the data to backup based on backup type
            if backup_type == BackupType.FULL:
                metadata_records, result_records = await self._get_all_extraction_data()
            else:  # INCREMENTAL
                # Get records since the last successful backup
                metadata_records, result_records = await self._get_incremental_data()
            
            # Prepare backup data structure
            backup_data = {
                "backup_info": {
                    "type": backup_type.value,
                    "timestamp": backup_timestamp.isoformat(),
                    "total_metadata_records": len(metadata_records),
                    "total_result_records": len(result_records)
                },
                "extraction_metadata": [self._serialize_metadata(record) for record in metadata_records],
                "extraction_results": [self._serialize_result(record) for record in result_records]
            }
            
            # Write backup to file
            with open(backup_filepath, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, default=str, indent=2, ensure_ascii=False)
            
            # Calculate checksum
            checksum = self._calculate_file_checksum(backup_filepath)
            
            # Update backup record
            backup_record.checksum = checksum
            backup_record.records_count = len(metadata_records)
            backup_record.status = BackupStatus.COMPLETED
            
            self.db.add(backup_record)
            await self.db.commit()
            
            logger.info(f"Backup created successfully: {backup_filename}, {len(metadata_records)} records")
            
            return BackupResult(
                backup_id=backup_record.id,
                success=True,
                message=f"Backup created successfully: {backup_filename}",
                records_count=len(metadata_records)
            )
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            
            # Update status to FAILED
            if 'backup_record' in locals():
                backup_record.status = BackupStatus.FAILED
                self.db.add(backup_record)
                await self.db.commit()
            
            # Remove the incomplete backup file if it was created
            if backup_filepath.exists():
                backup_filepath.unlink()
                
            return BackupResult(
                backup_id=backup_record.id if 'backup_record' in locals() else None,
                success=False,
                message=f"Backup failed: {str(e)}"
            )

    async def restore_from_backup(self, backup_id: int) -> RestoreResult:
        """
        Restore extraction data from a backup.
        
        Args:
            backup_id: ID of the backup to restore from
            
        Returns:
            RestoreResult with information about the operation
        """
        # Get backup record
        backup_record = await self.db.get(ExtractionBackup, backup_id)
        if not backup_record:
            return RestoreResult(
                success=False,
                message=f"Backup with ID {backup_id} not found"
            )
        
        backup_filepath = Path(backup_record.backup_file_path)
        if not backup_filepath.exists():
            return RestoreResult(
                success=False,
                message=f"Backup file does not exist: {backup_filepath}"
            )
        
        # Verify integrity
        if not await self.verify_backup_integrity(backup_id):
            return RestoreResult(
                success=False,
                message="Backup integrity check failed"
            )
        
        try:
            # Load backup data
            with open(backup_filepath, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Deserialize and store metadata records
            metadata_records = backup_data.get("extraction_metadata", [])
            result_records = backup_data.get("extraction_results", [])
            
            restored_metadata_count = 0
            restored_result_count = 0
            
            # Restore metadata records
            for metadata_json in metadata_records:
                metadata_obj = self._deserialize_metadata(metadata_json)
                # Check if a record with this file_hash and version already exists
                existing = await self.db.execute(
                    select(ExtractionMetadata).where(
                        ExtractionMetadata.file_hash == metadata_obj.file_hash,
                        ExtractionMetadata.version == metadata_obj.version
                    )
                )
                existing_record = existing.scalar_one_or_none()
                
                if existing_record:
                    # Update existing record
                    for field, value in metadata_obj.__dict__.items():
                        if field not in ['id', '_sa_instance_state']:  # Skip SQLAlchemy internal fields
                            setattr(existing_record, field, value)
                else:
                    # Add new record
                    self.db.add(metadata_obj)
                
                restored_metadata_count += 1
            
            # Restore result records
            for result_json in result_records:
                result_obj = self._deserialize_result(result_json)
                # Check if a record with this metadata_id already exists
                existing = await self.db.execute(
                    select(ExtractionResultDB).where(
                        ExtractionResultDB.metadata_id == result_obj.metadata_id
                    )
                )
                existing_record = existing.scalar_one_or_none()
                
                if existing_record:
                    # Update existing record
                    for field, value in result_obj.__dict__.items():
                        if field not in ['id', '_sa_instance_state']:  # Skip SQLAlchemy internal fields
                            setattr(existing_record, field, value)
                else:
                    # Add new record
                    self.db.add(result_obj)
                
                restored_result_count += 1
            
            await self.db.commit()
            
            logger.info(f"Restore completed: {restored_metadata_count} metadata records, {restored_result_count} result records")
            
            return RestoreResult(
                success=True,
                message=f"Restore completed: {restored_metadata_count} metadata records, {restored_result_count} result records",
                records_restored=restored_metadata_count
            )
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return RestoreResult(
                success=False,
                message=f"Restore failed: {str(e)}"
            )

    async def verify_backup_integrity(self, backup_id: int) -> bool:
        """
        Verify the integrity of a backup by checking its checksum.
        
        Args:
            backup_id: ID of the backup to verify
            
        Returns:
            True if integrity check passes, False otherwise
        """
        backup_record = await self.db.get(ExtractionBackup, backup_id)
        if not backup_record:
            logger.error(f"Backup with ID {backup_id} not found for integrity check")
            return False
        
        backup_filepath = Path(backup_record.backup_file_path)
        if not backup_filepath.exists():
            logger.error(f"Backup file does not exist: {backup_filepath}")
            return False
        
        # Calculate current checksum
        current_checksum = self._calculate_file_checksum(backup_filepath)
        
        # Compare with stored checksum
        is_valid = current_checksum == backup_record.checksum
        if not is_valid:
            logger.warning(f"Checksum mismatch for backup {backup_id}: expected {backup_record.checksum}, got {current_checksum}")
        
        return is_valid

    def get_backup_schedule(self) -> ScheduleInfo:
        """
        Get the current backup scheduling configuration.
        
        Returns:
            ScheduleInfo with scheduling information
        """
        # For now, return default configuration
        # In a real implementation, this would come from config service
        return ScheduleInfo(
            enabled=True,
            schedule="0 2 * * *",  # Daily at 2 AM
            retention_policy={
                "daily": 7,
                "weekly": 4,
                "monthly": 12,
                "cleanup_job_enabled": True
            }
        )

    def _serialize_metadata(self, metadata: ExtractionMetadata) -> Dict[str, Any]:
        """Serialize metadata object to JSON-compatible dict."""
        return {
            'id': metadata.id,
            'file_hash': metadata.file_hash,
            'file_path': metadata.file_path,
            'file_name': metadata.file_name,
            'file_size': metadata.file_size,
            'mime_type': metadata.mime_type,
            'created_date': metadata.created_date.isoformat() if metadata.created_date else None,
            'modified_date': metadata.modified_date.isoformat() if metadata.modified_date else None,
            'language': metadata.language,
            'ocr_confidence': metadata.ocr_confidence,
            'structure_score': metadata.structure_score,
            'extra': metadata.extra,
            'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
            'updated_at': metadata.updated_at.isoformat() if metadata.updated_at else None,
            'version': metadata.version,
            'is_active': metadata.is_active,
            'superseded_by_id': metadata.superseded_by_id,
            'first_seen': metadata.first_seen.isoformat() if metadata.first_seen else None,
            'last_updated': metadata.last_updated.isoformat() if metadata.last_updated else None
        }

    def _serialize_result(self, result: ExtractionResultDB) -> Dict[str, Any]:
        """Serialize result object to JSON-compatible dict."""
        return {
            'id': result.id,
            'metadata_id': result.metadata_id,
            'extracted_text': result.extracted_text,
            'extraction_engine': result.extraction_engine,
            'text_length': result.text_length,
            'quality_score': result.quality_score,
            'error_code': result.error_code,
            'error_message': result.error_message,
            'created_at': result.created_at.isoformat() if result.created_at else None
        }

    def _deserialize_metadata(self, metadata_json: Dict[str, Any]) -> ExtractionMetadata:
        """Deserialize JSON dict to metadata object."""
        # Convert ISO format strings back to datetime objects
        created_date = datetime.fromisoformat(metadata_json['created_date']) if metadata_json['created_date'] else None
        modified_date = datetime.fromisoformat(metadata_json['modified_date']) if metadata_json['modified_date'] else None
        created_at = datetime.fromisoformat(metadata_json['created_at']) if metadata_json['created_at'] else None
        updated_at = datetime.fromisoformat(metadata_json['updated_at']) if metadata_json['updated_at'] else None
        first_seen = datetime.fromisoformat(metadata_json['first_seen']) if metadata_json['first_seen'] else None
        last_updated = datetime.fromisoformat(metadata_json['last_updated']) if metadata_json['last_updated'] else None
        
        metadata = ExtractionMetadata(
            id=metadata_json['id'],
            file_hash=metadata_json['file_hash'],
            file_path=metadata_json['file_path'],
            file_name=metadata_json['file_name'],
            file_size=metadata_json['file_size'],
            mime_type=metadata_json['mime_type'],
            created_date=created_date,
            modified_date=modified_date,
            language=metadata_json['language'],
            ocr_confidence=metadata_json['ocr_confidence'],
            structure_score=metadata_json['structure_score'],
            extra=metadata_json['extra'],
            created_at=created_at,
            updated_at=updated_at,
            version=metadata_json['version'],
            is_active=metadata_json['is_active'],
            superseded_by_id=metadata_json['superseded_by_id'],
            first_seen=first_seen,
            last_updated=last_updated
        )
        return metadata

    def _deserialize_result(self, result_json: Dict[str, Any]) -> ExtractionResultDB:
        """Deserialize JSON dict to result object."""
        created_at = datetime.fromisoformat(result_json['created_at']) if result_json['created_at'] else None
        
        result = ExtractionResultDB(
            id=result_json['id'],
            metadata_id=result_json['metadata_id'],
            extracted_text=result_json['extracted_text'],
            extraction_engine=result_json['extraction_engine'],
            text_length=result_json['text_length'],
            quality_score=result_json['quality_score'],
            error_code=result_json['error_code'],
            error_message=result_json['error_message'],
            created_at=created_at
        )
        return result

    async def _get_all_extraction_data(self) -> tuple[List[ExtractionMetadata], List[ExtractionResultDB]]:
        """Get all extraction metadata and results."""
        # Get all metadata
        metadata_result = await self.db.execute(select(ExtractionMetadata))
        metadata_records = metadata_result.scalars().all()
        
        # Get all results
        result_result = await self.db.execute(select(ExtractionResultDB))
        result_records = result_result.scalars().all()
        
        return metadata_records, result_records

    async def _get_incremental_data(self) -> tuple[List[ExtractionMetadata], List[ExtractionResultDB]]:
        """Get extraction data since the last successful backup."""
        # Find the most recent successful backup
        last_backup_result = await self.db.execute(
            select(ExtractionBackup)
            .where(ExtractionBackup.status == BackupStatus.COMPLETED)
            .order_by(ExtractionBackup.backup_timestamp.desc())
            .limit(1)
        )
        last_backup = last_backup_result.scalar_one_or_none()
        
        if last_backup:
            # Get records modified since the last backup
            metadata_result = await self.db.execute(
                select(ExtractionMetadata)
                .where(ExtractionMetadata.last_updated > last_backup.backup_timestamp)
            )
            metadata_records = metadata_result.scalars().all()
            
            # For results, we need to get them based on their metadata's last update
            # Since result doesn't have a last_updated field, we get them via metadata
            result_result = await self.db.execute(
                select(ExtractionResultDB)
                .join(ExtractionMetadata, ExtractionResultDB.metadata_id == ExtractionMetadata.id)
                .where(ExtractionMetadata.last_updated > last_backup.backup_timestamp)
            )
            result_records = result_result.scalars().all()
        else:
            # If no previous backup, do a full backup instead
            logger.info("No previous backup found, returning all data for incremental backup")
            metadata_records, result_records = await self._get_all_extraction_data()
        
        return metadata_records, result_records

    def _calculate_file_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def cleanup_old_backups(self):
        """Remove old backups according to retention policy."""
        retention_policy = self.get_backup_schedule().retention_policy
        
        # Get all backups
        all_backups_result = await self.db.execute(
            select(ExtractionBackup)
            .order_by(ExtractionBackup.backup_timestamp.desc())
        )
        all_backups = all_backups_result.scalars().all()
        
        # Group backups by type (daily, weekly, monthly)
        # This is simplified - a real implementation would handle more complex retention logic
        now = datetime.utcnow()
        
        for backup in all_backups:
            # Calculate age in days
            age_days = (now - backup.backup_timestamp).days
            
            # Apply basic retention (keep daily backups for 7 days)
            if backup.backup_type == BackupType.FULL and age_days > retention_policy.get('daily', 7):
                # Remove old backup file
                backup_path = Path(backup.backup_file_path)
                if backup_path.exists():
                    backup_path.unlink()
                
                # Remove backup record from database
                await self.db.delete(backup)
                await self.db.commit()
                logger.info(f"Removed old backup: {backup_path}")