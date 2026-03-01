from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import List
from .models import WizardStep, WizardState, SystemCheckResponse, SystemCheckResult
import httpx
import os
from src.core.chroma_manager import get_chroma_manager

router = APIRouter()

# Static definition of steps for now
DEFAULT_STEPS = [
    WizardStep(id="system_check", title="System Check", description="Prüfe Verbindungen zu ChromaDB, Redis und Ollama", component="SystemCheck"),
    WizardStep(id="model_selection", title="Modell Auswahl", description="Wähle das LLM für die Antwort-Generierung", component="ModelSelector"),
    WizardStep(id="data_ingestion", title="Daten Import", description="Lade E-Mails und Dokumente in den RAG-Index", component="DataIngestion"),
    WizardStep(id="completion", title="Abschluss", description="Zusammenfassung und Start", component="Completion")
]

@router.get("/steps", response_model=List[WizardStep])
async def get_steps():
    """Returns the list of available onboarding/maintenance steps."""
    return DEFAULT_STEPS

@router.post("/run/system_check", response_model=SystemCheckResponse)
async def run_system_check():
    """Executes a system health check for critical components."""
    checks = []
    overall_status = "ok"

    # 1. Check ChromaDB
    try:
        manager = get_chroma_manager()
        # Use the proper method to get the client
        client = manager.get_client()
        if client:
            client.heartbeat()
            checks.append(SystemCheckResult(component="ChromaDB", status="ok", message="Connected successfully"))
        else:
            overall_status = "error"
            checks.append(SystemCheckResult(component="ChromaDB", status="error", message="Connection failed", details="Client not available"))
    except Exception as e:
        overall_status = "error"
        checks.append(SystemCheckResult(component="ChromaDB", status="error", message="Connection failed", details=str(e)))

    # 2. Check Ollama
    # Try Docker host first, then localhost
    ollama_hosts = [
        os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
        "http://localhost:11434"
    ]
    
    ollama_ok = False
    last_error = None
    
    for host in ollama_hosts:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{host}/api/tags")
                if resp.status_code == 200:
                     checks.append(SystemCheckResult(component="Ollama", status="ok", message=f"Connected to {host}"))
                     ollama_ok = True
                     break
                else:
                     last_error = f"Status {resp.status_code}: {resp.text}"
        except Exception as e:
            last_error = str(e)
    
    if not ollama_ok:
        overall_status = "error"
        checks.append(SystemCheckResult(component="Ollama", status="error", message="Connection failed", details=last_error))

    return SystemCheckResponse(overall_status=overall_status, checks=checks)


