from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.api.v1.dependencies import get_rag_client, get_current_user
from src.core.rag_client import RAGClient
from src.database.models import User

# Legacy models removed - /query endpoint is now in api/v1/rag/query.py

class AddTextRequest(BaseModel):
    text: str
    metadata: Dict[str, Any]
    collection_name: str = "default"

class AddTextResponse(BaseModel):
    success: bool
    message: str
    ids: List[str]

class CollectionRequest(BaseModel):
    collection_name: str

class CollectionResponse(BaseModel):
    success: bool
    message: str

class CollectionsResponse(BaseModel):
    collections: List[str]

router = APIRouter()

# Legacy /query endpoint removed - now handled by api/v1/rag/query.py


@router.post("/add-text", response_model=AddTextResponse)
async def add_text_to_rag(
    request: AddTextRequest,
    rag_client: RAGClient = Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Async add text to the RAG collection."""
    try:
        ids = await rag_client.add_text(
            request.text,
            request.metadata,
            collection_name=request.collection_name
        )
        return AddTextResponse(
            success=True, 
            message="Text added successfully", 
            ids=ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-collection", response_model=CollectionResponse)
async def create_collection(
    request: CollectionRequest,
    rag_client: RAGClient = Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Async create a new RAG collection."""
    try:
        await rag_client.create_collection(request.collection_name)
        return CollectionResponse(success=True, message=f"Collection '{request.collection_name}' created.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-collection", response_model=CollectionResponse)
async def delete_collection(
    request: CollectionRequest,
    rag_client: RAGClient = Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Async delete a RAG collection."""
    try:
        await rag_client.delete_collection(request.collection_name)
        return CollectionResponse(success=True, message=f"Collection '{request.collection_name}' deleted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-collections", response_model=CollectionsResponse)
async def list_collections(
    rag_client: RAGClient = Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Async list all RAG collections."""
    try:
        collections = await rag_client.list_collections()
        return CollectionsResponse(collections=collections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
