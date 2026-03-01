import os
import markdown
from src.core.exceptions import DocumentNotFoundError
from fastapi import APIRouter, HTTPException
from starlette.responses import HTMLResponse
from pathlib import Path
import asyncio

router = APIRouter()

DOCS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "docs"

async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

def get_docs_tree_sync(path: Path) -> list:
    tree = []
    for item in sorted(path.iterdir()):
        if item.name.startswith('.') or item.name in ["doku_rag", "module"]:
            continue
        
        entry = {"name": item.name}
        if item.is_dir():
            children = get_docs_tree_sync(item)
            if children:
                entry["children"] = children
                tree.append(entry)
        elif item.name.endswith(".md"):
            entry["path"] = str(item.relative_to(DOCS_DIR))
            tree.append(entry)
    return tree

@router.get("/", summary="Get documentation file tree")
async def get_documentation_tree():
    if not await run_in_threadpool(DOCS_DIR.is_dir):
        raise DocumentNotFoundError("docs_directory", collection="internal_docs")
    
    tree = await run_in_threadpool(get_docs_tree_sync, DOCS_DIR)
    return tree

@router.get("/{filepath:path}", summary="Get raw markdown file content")
async def get_documentation_file(filepath: str):
    try:
        full_path = await run_in_threadpool(lambda: DOCS_DIR.joinpath(filepath).resolve())
        if not str(full_path).startswith(str(DOCS_DIR.resolve())):
            raise ValidationError("File path is outside of the allowed directory", details={"path": file_path})

        is_file = await run_in_threadpool(full_path.is_file)
        if not is_file or not full_path.name.endswith(".md"):
            raise DocumentNotFoundError(str(full_path), collection="internal_docs")

        def read_file_sync():
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()

        md_content = await run_in_threadpool(read_file_sync)

        return {"content": md_content}

    except FileNotFoundError:
        raise DocumentNotFoundError("file", collection="internal_docs")
    except Exception as e:
        raise DocumentNotFoundError(str(e), collection="docs")