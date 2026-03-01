"""
Document Store for Parent Documents.

This module implements a dedicated SQLite database for storing full parent documents
to support the parent-child relationship needed for solving the chunk-size dilemma.
It provides bulk operations for efficient storage and retrieval of document content.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from llama_index.core.schema import Document as LlamaDocument
from loguru import logger


class DocumentStore:
    """
    A dedicated SQLite store for parent documents.

    Implements the storage layer for parent documents that are referenced by
    child chunks during retrieval. Provides efficient bulk operations for
    inserting and retrieving documents.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the DocumentStore.

        Args:
            db_path: Path to the SQLite database file. 
                     Defaults to backend/data/document_store.db
        """
        if db_path is None:
            from pathlib import Path
            db_path = str(Path(__file__).parent.parent.parent / "data" / "document_store.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.logger = logger.bind(component="DocumentStore")
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create the parent_documents table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_documents (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        
        self.logger.info(f"Document store initialized at {self.db_path}")

    def mset(self, documents: List[Tuple[str, LlamaDocument]]) -> bool:
        """
        Bulk insert/update parent documents.

        Args:
            documents: List of (doc_id, LlamaDocument) tuples to store

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare data for insertion
                data = []
                for doc_id, llama_doc in documents:
                    metadata = json.dumps(llama_doc.metadata) if llama_doc.metadata else '{}'
                    data.append((doc_id, llama_doc.text, metadata))
                
                # Bulk insert/update
                conn.executemany("""
                    INSERT OR REPLACE INTO parent_documents (doc_id, content, metadata)
                    VALUES (?, ?, ?)
                """, data)
                
                conn.commit()
                
                self.logger.info(f"Successfully stored {len(documents)} parent documents")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing parent documents: {e}")
            return False

    def mget(self, doc_ids: List[str]) -> List[Optional[LlamaDocument]]:
        """
        Bulk retrieve parent documents by ID.

        Args:
            doc_ids: List of document IDs to retrieve

        Returns:
            List of LlamaDocument objects (None for missing IDs)
        """
        if not doc_ids:
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create placeholders for SQL query
                placeholders = ','.join('?' * len(doc_ids))
                query = f"""
                    SELECT doc_id, content, metadata 
                    FROM parent_documents 
                    WHERE doc_id IN ({placeholders})
                """
                
                cursor = conn.execute(query, doc_ids)
                results = cursor.fetchall()
                
                # Create a mapping of doc_id to document for fast lookup
                doc_map = {}
                for row in results:
                    doc_id, content, metadata_str = row
                    try:
                        metadata = json.loads(metadata_str) if metadata_str else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    # Create LlamaDocument instance
                    doc = LlamaDocument(text=content, metadata=metadata)
                    doc.id_ = doc_id  # Set the ID explicitly
                    doc_map[doc_id] = doc
                
                # Return documents in the same order as requested
                return [doc_map.get(doc_id) for doc_id in doc_ids]
                
        except Exception as e:
            self.logger.error(f"Error retrieving parent documents: {e}")
            return [None] * len(doc_ids)

    def delete(self, doc_ids: List[str]) -> bool:
        """
        Bulk delete parent documents by ID.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        if not doc_ids:
            return True
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ','.join('?' * len(doc_ids))
                query = f"DELETE FROM parent_documents WHERE doc_id IN ({placeholders})"
                
                conn.execute(query, doc_ids)
                conn.commit()
                
                self.logger.info(f"Successfully deleted {len(doc_ids)} parent documents")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting parent documents: {e}")
            return False

    def exists(self, doc_id: str) -> bool:
        """
        Check if a parent document exists.

        Args:
            doc_id: Document ID to check

        Returns:
            True if exists, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM parent_documents WHERE doc_id = ?", 
                    (doc_id,)
                )
                return cursor.fetchone() is not None
                
        except Exception as e:
            self.logger.error(f"Error checking existence of parent document {doc_id}: {e}")
            return False

    def clear_all(self) -> bool:
        """
        Remove all parent documents from the store.

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM parent_documents")
                conn.commit()
                
                self.logger.info("Cleared all parent documents from store")
                return True
                
        except Exception as e:
            self.logger.error(f"Error clearing document store: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document store.

        Returns:
            Dictionary with store statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count total documents
                cursor = conn.execute("SELECT COUNT(*) FROM parent_documents")
                total_docs = cursor.fetchone()[0]
                
                # Get size on disk
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    "total_documents": total_docs,
                    "size_bytes": db_size,
                    "db_path": str(self.db_path)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting document store stats: {e}")
            return {"total_documents": 0, "size_bytes": 0, "db_path": str(self.db_path)}