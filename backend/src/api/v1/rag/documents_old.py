"""
RAG Document Management endpoints.

Handles document upload, retrieval, and deletion.
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import os
import shutil
from pathlib import Path

import asyncio

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.services.ingestion_task_manager import ingestion_task_manager
from src.core.docling_loader import DoclingLoaderFactory

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.xlsx', '.html', '.md', '.csv', '.txt', '.eml', '.mbox']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.get("/collections/{collection_name}/embedding-info")
async def get_collection_embedding_info(
    collection_name: str,
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Get embedding configuration for a collection and check compatibility."""
    logger.debug(f"Getting embedding info for collection: {collection_name}")

    try:
        metadata = await rag_client.get_collection_metadata(collection_name)

        if not metadata:
            logger.warning(f"No metadata found for collection '{collection_name}'")
            raise HTTPException(
                status_code=404,
                detail=f"No metadata found for collection '{collection_name}'. This might be an old collection."
            )

        from src.services.config_service import config_service
        config = config_service.load_configuration()

        current_embedding_model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        current_embedding_provider = config.get("EMBEDDING_PROVIDER", "ollama")

        try:
            current_dimensions = await rag_client.get_embedding_dimensions(current_embedding_model)
        except Exception:
            current_dimensions = 768

        current_embedding = {
            "model": current_embedding_model,
            "provider": current_embedding_provider,
            "dimensions": current_dimensions
        }

        collection_embedding = {
            "model": metadata.get("embedding_model"),
            "provider": metadata.get("embedding_provider"),
            "dimensions": metadata.get("embedding_dimensions")
        }

        compatible = (
            current_embedding["model"] == collection_embedding["model"] and
            current_embedding["dimensions"] == collection_embedding["dimensions"]
        )

        logger.debug(f"Embedding compatibility check for '{collection_name}': {compatible}")

        return {
            "collection_embedding": collection_embedding,
            "current_settings": current_embedding,
            "compatible": compatible,
            "created_at": metadata.get("created_at"),
            "description": metadata.get("description", "")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embedding info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-upload")
async def validate_upload(
    collection_name: str = Form(...),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Validate if upload is compatible with collection embedding configuration."""
    logger.debug(f"Validating upload for collection: {collection_name}")

    try:
        metadata = await rag_client.get_collection_metadata(collection_name)

        if not metadata:
            logger.warning(f"No metadata for '{collection_name}', allowing upload with warning")
            return {
                "valid": True,
                "warning": "No metadata found for this collection. Upload may fail if embedding models don't match.",
                "collection_model": "unknown",
                "current_model": "unknown"
            }

        from src.services.config_service import config_service
        config = config_service.load_configuration()

        current_model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        collection_model = metadata.get("embedding_model")
        collection_dims = metadata.get("embedding_dimensions")

        try:
            current_dims = await rag_client.get_embedding_dimensions(current_model)
        except Exception:
            current_dims = 768

        if current_model != collection_model or current_dims != collection_dims:
            logger.warning(f"Embedding mismatch: {current_model} vs {collection_model}")
            return JSONResponse(
                status_code=400,
                content={
                    "valid": False,
                    "error": "Embedding mismatch",
                    "collection_model": collection_model,
                    "current_model": current_model,
                    "collection_dims": collection_dims,
                    "current_dims": current_dims,
                    "suggestion": "Please change EMBEDDING_MODEL in Settings to match the collection"
                }
            )

        logger.debug("Upload validation passed")
        return {
            "valid": True,
            "message": "Upload is compatible",
            "collection_model": collection_model,
            "current_model": current_model
        }

    except Exception as e:
        logger.error(f"Upload validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Upload and index documents to ChromaDB with embedding validation."""
    logger.debug(f"Uploading {len(files)} files to collection '{collection_name}'")

    try:
        # STEP 1: Validate embedding compatibility
        metadata = await rag_client.get_collection_metadata(collection_name)

        if metadata:
            from src.services.config_service import config_service
            config = config_service.load_configuration()
            current_model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
            collection_model = metadata.get("embedding_model")
            collection_dims = metadata.get("embedding_dimensions")

            try:
                current_dims = await rag_client.get_embedding_dimensions(current_model)
            except Exception:
                current_dims = 768

            if current_model != collection_model or current_dims != collection_dims:
                logger.error(f"Embedding mismatch: {current_model} vs {collection_model}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding mismatch! Collection requires '{collection_model}' ({collection_dims} dims), but current settings use '{current_model}' ({current_dims} dims)."
                )

        results = []
        total_chunks = 0

        temp_dir = "/tmp/rag_upload"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            for file in files:
                try:
                    file_path = os.path.join(temp_dir, file.filename)

                    with open(file_path, "wb") as buffer:
                        content = await file.read()
                        buffer.write(content)

                    path_obj = Path(file_path)
                    file_extension = path_obj.suffix.lower()

                    # Use DoclingLoaderFactory instead of langchain loaders
                    from src.core.docling_loader import DoclingLoaderFactory
                    loader = DoclingLoaderFactory.create_loader(str(file_path))
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata.update({
                            'source': file.filename,
                            'file_type': file_extension,
                            'collection': collection_name
                        })

                    # Use LlamaIndex SentenceSplitter instead of langchain RecursiveCharacterTextSplitter
                    from llama_index.core.node_parser import SentenceSplitter
                    text_splitter = SentenceSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Convert documents to LlamaIndex format for splitting
                    from llama_index.core.schema import Document as LlamaDocument
                    llama_docs = []
                    for doc in docs:
                        llama_doc = LlamaDocument(
                            text=doc.text if hasattr(doc, 'text') else doc.page_content,
                            metadata=doc.metadata
                        )
                        llama_docs.append(llama_doc)
                    
                    # Split using LlamaIndex
                    nodes = text_splitter.get_nodes_from_documents(llama_docs)
                    
                    # Convert back to our format
                    splits = []
                    for node in nodes:
                        # Create a document-like object with page_content and metadata
                        split_doc = type('Document', (), {
                            'page_content': node.text,
                            'metadata': node.metadata
                        })()
                        splits.append(split_doc)

                    try:
                        collection = await asyncio.to_thread(rag_client.chroma_manager.get_collection, collection_name)
                    except Exception:
                        response = await rag_client.create_collection(name=collection_name)
                        if not response.is_success:
                            raise HTTPException(status_code=500, detail=f"Failed to create collection: {response.error}")
                        collection = await asyncio.to_thread(rag_client.chroma_manager.get_collection, collection_name)

                    texts = [doc.page_content for doc in splits]

                    # Get embedding instance from embedding_manager
                    embedding_instance = rag_client.embedding_manager.get_embeddings()
                    if not embedding_instance:
                        raise Exception("Failed to initialize embeddings")

                    # Generate embeddings (use async if available)
                    if hasattr(embedding_instance, 'aembed_documents'):
                        embeddings = await embedding_instance.aembed_documents(texts)
                    else:
                        embeddings = await asyncio.to_thread(embedding_instance.embed_documents, texts)

                    for i, doc in enumerate(splits):
                        await asyncio.to_thread(collection.add,
                            documents=[doc.page_content],
                            embeddings=[embeddings[i]],
                            metadatas=[doc.metadata],
                            ids=[f"{file.filename}_{i}_{os.urandom(4).hex()}"]
                        )

                    chunk_count = len(splits)
                    total_chunks += chunk_count

                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "chunks": chunk_count
                    })

                    logger.info(f"Successfully uploaded '{file.filename}': {chunk_count} chunks")

                except Exception as file_error:
                    logger.error(f"Failed to upload '{file.filename}': {file_error}")
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "chunks": 0,
                        "error": str(file_error)
                    })

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        successful_files = sum(1 for r in results if r["success"])
        logger.info(f"Upload complete: {successful_files}/{len(files)} files, {total_chunks} chunks")

        return {
            "success": successful_files > 0,
            "files_processed": successful_files,
            "total_files": len(files),
            "total_chunks": total_chunks,
            "results": results,
            "collection": collection_name,
            "chunk_config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        }

    except HTTPException:
        raise  # Re-raise HTTPException unchanged (e.g. 400 for embedding mismatch)
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest-documents")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Ingest documents using Docling with background processing and task tracking."""
    logger.debug(f"Ingesting {len(files)} documents to collection '{collection_name}' using Docling")
    
    try:
        # Validate embedding compatibility
        metadata = await rag_client.get_collection_metadata(collection_name)
        
        if metadata:
            from src.services.config_service import config_service
            config = config_service.load_configuration()
            current_model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
            collection_model = metadata.get("embedding_model")
            collection_dims = metadata.get("embedding_dimensions")

            try:
                current_dims = await rag_client.get_embedding_dimensions(current_model)
            except Exception:
                current_dims = 768

            if current_model != collection_model or current_dims != collection_dims:
                logger.error(f"Embedding mismatch: {current_model} vs {collection_model}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding mismatch! Collection requires '{collection_model}' ({collection_dims} dims), but current settings use '{current_model}' ({current_dims} dims)."
                )

        # Save uploaded files temporarily
        temp_dir = "/tmp/rag_docling_upload"
        os.makedirs(temp_dir, exist_ok=True)
        
        validated_assignments = []
        
        try:
            # Process uploaded files
            for file in files:
                # Validate file extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in SUPPORTED_EXTENSIONS:
                    logger.warning(f"Skipping unsupported file {file.filename}: {file_ext}")
                    continue

                file_path = os.path.join(temp_dir, file.filename)
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    # Validate file size
                    if len(content) > MAX_FILE_SIZE:
                        logger.warning(f"Skipping file {file.filename} due to size: {len(content)} bytes")
                        continue
                    buffer.write(content)
                
                # Validate file with DoclingLoaderFactory
                try:
                    loader = DoclingLoaderFactory.create_loader(file_path)
                except ValueError as e:
                    logger.warning(f"Skipping unsupported file {file.filename}: {e}")
                    continue  # Skip unsupported files
                
                validated_assignments.append({
                    "file_path": file_path,
                    "filename": file.filename,
                    "collection": collection_name,
                    "content": content,
                    "size": len(content)
                })
            
            if not validated_assignments:
                raise HTTPException(status_code=400, detail="No supported files were uploaded")
            
            # Create background task
            task_id = ingestion_task_manager.create_task(
                file_count=len(validated_assignments),
                collection_name=collection_name,
                assignments=validated_assignments,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Start async processing using FastAPI background tasks
            # Phase 8f: Use new async process_task_async() method via processor
            processor = ingestion_task_manager.get_processor()
            background_tasks.add_task(
                processor.process_task_async,
                task_id=task_id,
                rag_client=rag_client
            )
            
            logger.info(f"Started ingestion task: {task_id} for {len(validated_assignments)} files")
            
            return {
                "success": True,
                "task_id": task_id,
                "files_count": len(validated_assignments),
                "collection": collection_name
            }

        finally:
            # Note: Temp files will be cleaned up by the processing task after completion
            # Or need to be cleaned up separately
            pass

    except Exception as e:
        logger.error(f"Ingest documents failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/documents")
async def get_documents(
    collection_name: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Get documents from collection with pagination."""
    logger.debug(f"Getting documents from '{collection_name}': limit={limit}, offset={offset}")

    try:
        docs_data = await rag_client.get_documents(
            collection_name=collection_name,
            limit=limit,
            offset=offset
        )
        logger.debug(f"Retrieved {len(docs_data.get('documents', []))} documents")
        return docs_data

    except Exception as e:
        logger.error(f"Failed to get documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection_name: str = Query(...),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Delete document from collection."""
    logger.debug(f"Deleting document '{document_id}' from collection '{collection_name}'")

    try:
        response = await rag_client.delete_document(
            doc_id=document_id,
            collection_name=collection_name
        )

        if not response.is_success:
            logger.error(f"Failed to delete document '{document_id}': {response.error}")
            raise HTTPException(status_code=500, detail=response.error)

        logger.info(f"Document '{document_id}' deleted successfully")
        return {
            "success": True,
            "deleted_id": document_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/test")
async def test_query(
    query: str = Form(...),
    collection_name: str = Form("default"),
    n_results: int = Form(5),
    generate_answer: bool = Form(False),
    advanced_mode: bool = Form(False),  # NEW: Enable Multi-Collection like DraftService
    domain: Optional[str] = Form(None),  # NEW: Domain-based routing
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """
    Test RAG query with custom parameters and optional LLM answer generation.

    NEW ADVANCED MODE:
    - advanced_mode=True: Uses SAME Multi-Collection logic as DraftService
    - Queries ALL collections (or domain-filtered collections)
    - Cross-collection ranking (Top N most relevant across ALL collections)
    - Returns collection stats showing source breakdown
    """
    logger.debug(f"Testing query: {query[:50]}... (advanced={advanced_mode}, domain={domain}, generate_answer={generate_answer})")

    try:
        formatted_results = []
        collection_stats = {}

        if advanced_mode:
            # ✅ ADVANCED MODE: Use SAME logic as DraftService.generate_draft()
            logger.info("🚀 Advanced Mode: Using Multi-Collection RAG (like DraftService)")

            # Step 1: Get all available collections
            response = await rag_client.list_collections()
            if not response.is_success:
                logger.error(f"Failed to list collections: {response.error}")
                all_available_collections = []
            else:
                all_available_collections = response.data

            # Step 2: Domain-based collection selection (SAME as DraftService)
            from src.services.config_service import config_service
            config = config_service.load_configuration()

            # Load rag_domains.json
            import json
            from pathlib import Path
            try:
                domains_path = Path(__file__).parent.parent.parent / "core" / "rag_domains.json"
                with open(domains_path, 'r', encoding='utf-8') as f:
                    rag_domains = json.load(f).get("domains", {})
            except Exception:
                rag_domains = {}

            collections_to_query = []

            if domain and rag_domains:
                logger.info(f"Domain-based routing for domain: '{domain}'")
                domain_info = rag_domains.get(domain)
                if domain_info:
                    domain_collections = domain_info.get("collections", [])
                    collections_to_query = [c for c in domain_collections if c in all_available_collections]
                    if not collections_to_query:
                        logger.warning(f"Domain '{domain}' collections not found. Falling back to all.")

            # Fallback: Use all collections or config-filtered
            if not collections_to_query:
                config_collections_str = config.get("DRAFT_RAG_COLLECTIONS", "").strip()
                if config_collections_str:
                    configured_list = [c.strip() for c in config_collections_str.split(",")]
                    collections_to_query = [c for c in configured_list if c in all_available_collections]
                    if not collections_to_query:
                        collections_to_query = all_available_collections
                else:
                    collections_to_query = all_available_collections

            logger.info(f"Querying {len(collections_to_query)} collections: {collections_to_query}")

            # Step 3: Query each collection with adaptive k (SAME as DraftService)
            all_results = []
            k_per_collection = max(10, 100 // len(collections_to_query)) if collections_to_query else n_results

            for collection_name_iter in collections_to_query:
                try:
                    response = await rag_client.query(
                        query_text=query,
                        collection_names=[collection_name_iter],
                        n_results=k_per_collection
                    )

                    if response.is_success and response.data:
                        for res in response.data:
                            res['source_collection'] = collection_name_iter
                        all_results.extend(response.data)
                        collection_stats[collection_name_iter] = len(response.data)
                except Exception as col_error:
                    logger.warning(f"Failed to query collection '{collection_name_iter}': {col_error}")
                    continue

            # Step 4: Cross-collection ranking (SAME as DraftService)
            all_results.sort(key=lambda x: x.get('relevance_score', 1 - x.get('distance', 0)), reverse=True)
            top_results = all_results[:n_results]

            # Step 5: Format results with collection tags
            for i, result in enumerate(top_results):
                formatted_results.append({
                    "rank": i + 1,
                    "content": result.get('content', result.get('document', '')),
                    "metadata": result.get('metadata', {}),
                    "distance": result.get('distance', 0),
                    "relevance_score": result.get('relevance_score', 1 - result.get('distance', 0)),
                    "source_collection": result.get('source_collection', 'unknown')  # NEW!
                })

            # Collection breakdown for top results
            top_collection_breakdown = {}
            for res in top_results:
                coll = res.get('source_collection', 'unknown')
                top_collection_breakdown[coll] = top_collection_breakdown.get(coll, 0) + 1

            logger.info(
                f"Advanced Mode: {len(all_results)} total results from {len(collection_stats)} collections. "
                f"Top {n_results} from: {top_collection_breakdown}"
            )

        else:
            # ❌ SIMPLE MODE: Single-collection query (old behavior)
            logger.info(f"Simple Mode: Single-collection query on '{collection_name}'")

            response = await rag_client.query(
                query_text=query,
                collection_names=[collection_name],
                n_results=n_results
            )

            if not response.is_success:
                logger.error(f"Query failed: {response.error}")
                raise HTTPException(status_code=500, detail=response.error)

            for i, result in enumerate(response.data):
                formatted_results.append({
                    "rank": i + 1,
                    "content": result.get('content', result.get('document', '')),
                    "metadata": result.get('metadata', {}),
                    "distance": result.get('distance', 0),
                    "relevance_score": 1 - result.get('distance', 0),
                    "source_collection": collection_name  # Single collection
                })

            collection_stats[collection_name] = len(formatted_results)

        logger.info(f"Query returned {len(formatted_results)} results")

        result_data = {
            "query": query,
            "collection": collection_name if not advanced_mode else "multi-collection",
            "results": formatted_results,
            "source_chunks": formatted_results,
            "total_results": len(formatted_results),
            "advanced_mode": advanced_mode,
            "collection_stats": collection_stats,  # NEW: Shows which collections were queried
            "domain": domain
        }

        # Generate LLM answer if requested
        if generate_answer and formatted_results:
            try:
                from src.core.llm_singleton import get_llm
                llm = get_llm()

                if llm:
                    # For counting queries, use ALL available results (not just top 5)
                    # to avoid misleading the LLM about total counts
                    chunks_to_use = formatted_results  # Use ALL results for accuracy

                    # Build context from results with collection tags (NEW for Advanced Mode)
                    if advanced_mode:
                        context_chunks = "\n\n---\n\n".join([
                            f"[Collection: {r['source_collection'].upper()} | Chunk {i+1} | Relevance: {r['relevance_score']:.2f}]\n{r['content']}"
                            for i, r in enumerate(chunks_to_use)
                        ])
                        collection_info = f"Daten aus {len(collection_stats)} Collections: {', '.join(collection_stats.keys())}"
                    else:
                        context_chunks = "\n\n---\n\n".join([
                            f"[Chunk {i+1}, Relevance: {r['relevance_score']:.2f}]\n{r['content']}"
                            for i, r in enumerate(chunks_to_use)
                        ])
                        collection_info = f"Daten aus Collection: {collection_name}"

                    # Build prompt - WICHTIG: Erkläre das Limit-Problem!
                    num_chunks = len(chunks_to_use)
                    num_requested = n_results

                    prompt = f"""WICHTIG: ANTWORTE AUSSCHLIEẞLICH AUF DEUTSCH! NIEMALS AUF ENGLISCH!

Du bist ein hilfreicher Firmen-Assistent mit Zugriff auf eine Multi-Domain Wissensdatenbank.

═══════════════════════════════════════════════════
RAG-SYSTEM INFO:
{"🚀 ADVANCED MODE: Multi-Collection Query über ALLE Domains" if advanced_mode else "📋 SIMPLE MODE: Single-Collection Query"}
{collection_info}
═══════════════════════════════════════════════════

═══════════════════════════════════════════════════
EINSCHRÄNKUNG DER SUCHERGEBNISSE:
Du siehst die TOP {num_chunks} relevantesten Chunks (Limit: {num_requested}).
{"Diese wurden aus ALLEN verfügbaren Collections ausgewählt und nach Relevanz sortiert." if advanced_mode else ""}
Es können WEITERE Informationen in der Datenbank existieren, die hier NICHT gezeigt werden!
═══════════════════════════════════════════════════

VERFÜGBARE DATEN:
{context_chunks}

KUNDENFRAGE: {query}

═══════════════════════════════════════════════════
DEINE ANWEISUNGEN (STRIKT BEFOLGEN):
═══════════════════════════════════════════════════

1. SPRACHE: Antworte AUSSCHLIEẞLICH auf DEUTSCH!

2. DATENQUELLE NUTZEN:
   {"- Die Chunks kommen aus verschiedenen Collections (siehe Collection-Tag)" if advanced_mode else "- Die Chunks kommen aus einer Collection"}
   - Nutze die Relevanz-Scores - höhere Scores = verlässlichere Daten
   - Bei widersprüchlichen Informationen: Bevorzuge höhere Relevanz

3. ZÄHLEN: Wenn nach "wie viele" gefragt wird:
   - Zähle die UNTERSCHIEDLICHEN Entities in den Chunks
   - Sage: "In den angezeigten Ergebnissen sind mindestens X [Items] sichtbar"
   - Erwähne: "Es könnten weitere [Items] in der Datenbank existieren"

4. EHRLICHKEIT: Wenn das Limit zu niedrig ist, weise darauf hin!

5. DETAILS: Nenne konkrete Details aus den Chunks (Namen, Zahlen, Spezifikationen, etc.)

6. TRANSPARENZ:
   {"- Wenn Daten aus verschiedenen Collections kommen, erkenne das an" if advanced_mode else ""}
   - Sei ehrlich wenn Informationen fehlen oder unklar sind

═══════════════════════════════════════════════════
ANTWORT (AUF DEUTSCH):"""

                    # Generate answer using LlamaIndex LLM interface
                    try:
                        # Use LlamaIndex LLM interface instead of LangChain invoke
                        if hasattr(llm, 'complete'):
                            llm_response = await asyncio.to_thread(llm.complete, prompt)
                        elif hasattr(llm, 'predict'):
                            llm_response = await asyncio.to_thread(llm.predict, prompt)
                        elif hasattr(llm, 'chat'):
                            llm_response = await asyncio.to_thread(llm.chat, prompt)
                        else:
                            llm_response = await asyncio.to_thread(llm, prompt)  # Direct call as fallback
                        answer = str(llm_response)
                    except Exception as invoke_error:
                        logger.error(f"LLM invocation failed: {invoke_error}")
                        answer = f"LLM invocation error: {str(invoke_error)}"

                    result_data["answer"] = answer
                    logger.info(f"Generated LLM answer ({len(answer)} chars)")
                else:
                    logger.warning("LLM not available for answer generation")
                    result_data["answer"] = "LLM is not available. Please check your LLM configuration."

            except Exception as llm_error:
                logger.error(f"LLM answer generation failed: {llm_error}", exc_info=True)
                result_data["answer"] = f"Failed to generate answer: {str(llm_error)}"

        return result_data

    except Exception as e:
        logger.error(f"Query test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
