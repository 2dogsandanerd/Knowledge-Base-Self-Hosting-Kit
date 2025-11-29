"""
LLM-powered Ingestion Tasks API.

Provides REST endpoints for LLM-based document ingestion operations.
"""

from fastapi import APIRouter, Depends, Form
from src.api.v1.dependencies import get_llm_task_router, get_current_user
from src.services.llm_task_router import LLMTaskRouter, TaskType
from src.database.models import User
from src.core.exceptions import ValidationError, IngestionError
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate-ragignore")
async def generate_ragignore_endpoint(
    folder_path: str = Form(..., description="Path to folder to analyze"),
    include_examples: bool = Form(True, description="Include comment examples"),
    aggressive: bool = Form(False, description="More aggressive exclusions"),
    task_router: LLMTaskRouter = Depends(get_llm_task_router),
    current_user: User = Depends(get_current_user)
):
    """
    Generate .ragignore file from intelligent folder analysis.

    **Platform 1: Ragignore Generation Station** 🚂

    Analyzes the folder structure, classifies file types, and uses LLM to generate
    a smart .ragignore file that optimizes document ingestion for RAG systems.

    **Process:**
    1. Scans folder recursively (up to depth 10)
    2. Classifies files into categories (code, docs, build artifacts, etc.)
    3. LLM generates .ragignore with smart exclusion patterns
    4. Returns editable suggestion with statistics

    **Example Response:**
    ```json
    {
        "ragignore": "# RAG Ignore File\\nnode_modules/\\nvenv/\\n...",
        "analysis_summary": {
            "total_files": 15234,
            "files_to_ignore": 12000,
            "files_to_keep": 3234,
            "estimated_size_reduction_mb": 450.5
        },
        "reasoning": "Node.js project with Python backend...",
        "detected_categories": {
            "source_code": 3000,
            "dependencies": 12000,
            "documentation": 234
        }
    }
    ```

    **Args:**
        folder_path: Absolute or relative path to folder
        include_examples: Include helpful comments with examples (default: True)
        aggressive: More strict exclusions (default: False)

    **Returns:**
        Dictionary with .ragignore content, statistics, and reasoning

    **Raises:**
        ValidationError: If folder path is invalid
        IngestionError: If generation fails
    """
    logger.info(f"Generating .ragignore for: {folder_path} (user: {current_user.id})")

    try:
        result = await task_router.execute(
            TaskType.GENERATE_RAGIGNORE,
            {
                "folder_path": folder_path,
                "include_examples": include_examples,
                "aggressive": aggressive
            }
        )

        logger.info(f".ragignore generated successfully for {folder_path}")
        return result

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise ValidationError(str(e))

    except Exception as e:
        logger.error(f"Ragignore generation failed: {e}", exc_info=True)
        raise IngestionError(f"Failed to generate .ragignore: {str(e)}")


@router.post("/save-ragignore")
async def save_ragignore_endpoint(
    folder_path: str = Form(...),
    content: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """
    Save .ragignore file to disk.

    Args:
        folder_path: Folder where to save .ragignore
        content: .ragignore file content

    Returns:
        {"success": True, "file_path": "..."}

    Raises:
        ValidationError: If path is invalid
        IngestionError: If save fails
    """
    logger.info(f"Saving .ragignore to: {folder_path}")

    try:
        # Validate folder path
        abs_path = os.path.abspath(folder_path)
        if not os.path.exists(abs_path):
            raise ValidationError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(abs_path):
            raise ValidationError(f"Path is not a directory: {folder_path}")

        # Write .ragignore file
        ragignore_path = os.path.join(abs_path, ".ragignore")

        # Check write permissions
        if not os.access(abs_path, os.W_OK):
            raise ValidationError(f"No write permission for: {folder_path}")

        with open(ragignore_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f".ragignore saved successfully: {ragignore_path}")

        return {
            "success": True,
            "file_path": ragignore_path,
            "message": ".ragignore saved successfully"
        }

    except ValidationError:
        raise

    except Exception as e:
        logger.error(f"Failed to save .ragignore: {e}", exc_info=True)
        raise IngestionError(f"Failed to save .ragignore: {str(e)}")


@router.post("/generate-summary")
async def generate_summary_endpoint(
    folder_path: str = Form(..., description="Path to folder to analyze"),
    format: str = Form("markdown", description="Output format (markdown | json)"),
    include_file_tree: bool = Form(True, description="Include directory tree"),
    max_depth: int = Form(3, description="Max tree depth"),
    task_router: LLMTaskRouter = Depends(get_llm_task_router),
    current_user: User = Depends(get_current_user)
):
    """
    Generate comprehensive markdown summary of folder.

    **Platform 2: Summary Generation Station** 🚂

    Args:
        folder_path: Path to analyze
        format: Output format (markdown | json)
        include_file_tree: Include directory tree
        max_depth: Max tree depth

    Returns:
        {
            "summary": "# Project Overview\\n...",
            "metadata": {...}
        }
    """
    logger.info(f"Generating summary for: {folder_path} (user: {current_user.id})")

    try:
        result = await task_router.execute(
            TaskType.GENERATE_SUMMARY,
            {
                "folder_path": folder_path,
                "format": format,
                "include_file_tree": include_file_tree,
                "max_depth": max_depth
            }
        )

        return result

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        raise IngestionError(f"Failed to generate summary: {str(e)}")


@router.post("/generate-collection-config")
async def generate_collection_config_endpoint(
    folder_path: str = Form(..., description="Path to folder to analyze"),
    use_case: str = Form("search", description="Intended use case (search, qa, chatbot)"),
    task_router: LLMTaskRouter = Depends(get_llm_task_router),
    current_user: User = Depends(get_current_user)
):
    """
    Generate optimal RAG collection configuration.

    **Platform 3: Collection Config Station** 🚂

    Analyzes folder content and suggests:
    - Collection structure (how to split files)
    - Chunking strategy (size, overlap)
    - Embedding model recommendation

    Args:
        folder_path: Path to analyze
        use_case: Intended use case

    Returns:
        JSON with configuration and reasoning
    """
    logger.info(f"Generating collection config for: {folder_path} (user: {current_user.id})")

    try:
        result = await task_router.execute(
            TaskType.GENERATE_COLLECTION_CONFIG,
            {
                "folder_path": folder_path,
                "use_case": use_case
            }
        )

        return result

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Config generation failed: {e}", exc_info=True)
        raise IngestionError(f"Failed to generate config: {str(e)}")
