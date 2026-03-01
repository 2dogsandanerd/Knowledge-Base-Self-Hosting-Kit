"""
Folder Ingestion Endpoint.

Triggers asynchronous ingestion of a local folder into the RAG system.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.core.ingest_config import ChunkConfig
from src.services.docling_service import docling_service
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
import asyncio
from pathlib import Path
from src.services.ingestion_task_manager import ingestion_task_manager, TaskStatus, FileResult
from datetime import datetime
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)
router = APIRouter()


def flatten_metadata(metadata: dict) -> dict:
    """Flatten nested metadata for ChromaDB compatibility.

    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    This function converts nested dicts and lists to JSON strings.
    """
    flattened = {}
    for key, value in metadata.items():
        if value is None:
            flattened[key] = None
        elif isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, (dict, list)):
            # Convert nested structures to JSON string
            import json
            flattened[key] = json.dumps(value)
        else:
            # Convert other types to string
            flattened[key] = str(value)
    return flattened


class IngestFolderRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute path to the folder to ingest")
    collection_name: str = Field(..., description="Name of the target collection")
    profile: str = Field("default", description="Ingestion profile (codebase, documents, default)")
    recursive: bool = Field(True, description="Whether to scan recursively")
    allowed_extensions: Optional[List[str]] = Field(None, description="Specific extensions to include")

async def _process_files_background(
    task_id: str,
    files_to_ingest: List[str],
    request: IngestFolderRequest,
    rag_client
):
    """Background task to process files and update task status."""
    task = ingestion_task_manager.get_task(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return

    task.status = TaskStatus.PROCESSING
    task.started_at = datetime.now()

    successful = 0
    failed = 0

    for idx, file_path in enumerate(files_to_ingest, 1):
        try:
            logger.info(f"[{idx}/{len(files_to_ingest)}] Processing: {file_path}")
            filename = os.path.basename(file_path)
            ext = Path(file_path).suffix.lower()
            start_time = datetime.now()

            # Determine processing method based on file type
            if docling_service.is_supported_file(file_path):
                # Documents: Use Docling Service
                process_result = await docling_service.process_file(file_path)

                if not process_result["success"]:
                    failed += 1
                    task.file_results.append(FileResult(
                        file_path=file_path,
                        filename=filename,
                        success=False,
                        error=process_result.get("error", "Docling processing failed"),
                        processing_time=(datetime.now() - start_time).total_seconds()
                    ))
                    task.processed_files = idx
                    task.failed_files = failed
                    continue

                markdown_content = process_result["content"]
                doc_metadata = process_result["metadata"]
            else:
                # Code files: Read as plain text
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                    doc_metadata = {
                        "title": filename,
                        "file_type": ext,
                        "size": os.path.getsize(file_path)
                    }
                except UnicodeDecodeError:
                    # Try with different encoding for non-UTF8 files
                    with open(file_path, 'r', encoding='latin-1') as f:
                        markdown_content = f.read()
                    doc_metadata = {
                        "title": filename,
                        "file_type": ext,
                        "size": os.path.getsize(file_path),
                        "encoding": "latin-1"
                    }

            # Add ingestion metadata
            doc_metadata.update({
                'source': filename,
                'file_type': ext,
                'collection': request.collection_name,
                'processed_by': 'docling_service_v1'
            })

            # Use LlamaIndex SentenceSplitter
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )

            llama_doc = LlamaDocument(
                text=markdown_content,
                metadata=doc_metadata
            )

            # Split content
            nodes = text_splitter.get_nodes_from_documents([llama_doc])

            # Get or create collection
            try:
                collection = await asyncio.to_thread(
                    rag_client.chroma_manager.get_collection,
                    request.collection_name
                )
            except Exception:
                response = await rag_client.create_collection(name=request.collection_name)
                if not response.is_success:
                    raise Exception(f"Failed to create collection: {response.error}")
                collection = await asyncio.to_thread(
                    rag_client.chroma_manager.get_collection,
                    request.collection_name
                )

            # Prepare for ChromaDB
            texts = [node.text for node in nodes]
            metadatas = [flatten_metadata(node.metadata) for node in nodes]
            ids = [f"{filename}_{i}_{os.urandom(4).hex()}" for i in range(len(nodes))]

            # Generate Embeddings
            embedding_instance = rag_client.embedding_manager.get_embeddings()
            if not embedding_instance:
                raise Exception("Failed to initialize embeddings")

            if hasattr(embedding_instance, 'aget_text_embedding_batch'):
                embeddings = await embedding_instance.aget_text_embedding_batch(texts)
            elif hasattr(embedding_instance, 'get_text_embedding_batch'):
                embeddings = await asyncio.to_thread(embedding_instance.get_text_embedding_batch, texts)
            elif hasattr(embedding_instance, 'aembed_documents'):
                embeddings = await embedding_instance.aembed_documents(texts)
            elif hasattr(embedding_instance, 'embed_documents'):
                embeddings = await asyncio.to_thread(embedding_instance.embed_documents, texts)
            else:
                raise Exception("Embedding instance has no compatible embed method")

            # Add to ChromaDB
            await asyncio.to_thread(collection.add,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            chunk_count = len(nodes)
            successful += 1
            task.file_results.append(FileResult(
                file_path=file_path,
                filename=filename,
                success=True,
                chunks=chunk_count,
                processing_time=(datetime.now() - start_time).total_seconds()
            ))

            # Update task progress
            task.processed_files = idx
            task.successful_files = successful
            task.total_chunks += chunk_count

            logger.info(f"✓ [{idx}/{len(files_to_ingest)}] '{filename}': {chunk_count} chunks")

        except Exception as e:
            failed += 1
            logger.error(f"✗ [{idx}/{len(files_to_ingest)}] Failed to process {file_path}: {e}")
            task.file_results.append(FileResult(
                file_path=file_path,
                filename=os.path.basename(file_path),
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            ))
            task.processed_files = idx
            task.failed_files = failed

    # Mark task as complete
    task.completed_at = datetime.now()
    if failed == 0:
        task.status = TaskStatus.SUCCESS
    elif successful == 0:
        task.status = TaskStatus.FAILED
    else:
        task.status = TaskStatus.PARTIAL

    logger.info(f"Task {task_id} completed: {successful} successful, {failed} failed")


@router.post("/ingest-folder")
async def ingest_folder_endpoint(
    request: IngestFolderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    rag_client=Depends(get_rag_client)
):
    """
    Asynchronous ingestion of a local folder with real-time progress tracking.

    1. Scans the folder for matching files.
    2. Creates a background task to process them.
    3. Returns task_id for progress polling.
    """
    logger.info(f"Received folder ingestion request: {request.folder_path} -> {request.collection_name}")

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")

    # 1. Determine extensions based on profile
    extensions = request.allowed_extensions
    if not extensions:
        if request.profile == "codebase":
            extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json", ".yml", ".yaml", ".html", ".css", ".sql"]
        elif request.profile == "documents":
            extensions = [".pdf", ".docx", ".txt", ".md"]
        else:
            # Default: broad set
            extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".js"]

    extensions = [ext.lower() for ext in extensions]
    logger.info(f"Using extensions filter: {extensions}")

    # 2. Scan for files
    files_to_ingest = []
    for root, dirs, filenames in os.walk(request.folder_path):
        # Skip common ignore dirs
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build"}]

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            # Only filter by extension list (includes both documents and code files)
            if ext in extensions:
                full_path = os.path.join(root, filename)
                files_to_ingest.append(full_path)

        if not request.recursive:
            break

    if not files_to_ingest:
        raise HTTPException(status_code=400, detail="No matching files found in the specified folder.")

    logger.info(f"Found {len(files_to_ingest)} files to ingest.")

    # 3. Create background task
    task_id = ingestion_task_manager.create_task(
        file_count=len(files_to_ingest),
        collection_name=request.collection_name,
        user_id=current_user.id if hasattr(current_user, 'id') else None
    )

    # 4. Start background processing
    background_tasks.add_task(
        _process_files_background,
        task_id,
        files_to_ingest,
        request,
        rag_client
    )

    logger.info(f"Created task {task_id} for {len(files_to_ingest)} files")

    return {
        "task_id": task_id,
        "status": "processing",
        "files_found": len(files_to_ingest),
        "collection": request.collection_name,
        "message": f"Ingestion started. Poll /api/v1/rag/ingestion/ingest-status/{task_id} for progress."
    }
