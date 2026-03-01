from pydantic import BaseModel
from typing import List, Dict, Any


class EvaluationRequest(BaseModel):
    """Input for RAG‑evaluation.

    * ``generated`` – list of texts produced by the RAG system.
    * ``reference`` – list of ground‑truth/reference texts.
    """
    generated: List[str]
    reference: List[str]


class EvaluationResult(BaseModel):
    """Simplified RAGAS result.

    In a real implementation the ``details`` field would contain the full
    ragas output (multiple scores, per‑sample breakdown, etc.).
    """
    scores: Dict[str, float]
    details: Dict[str, Any] | None = None
