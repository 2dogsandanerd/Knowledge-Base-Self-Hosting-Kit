from fastapi import APIRouter, HTTPException
from loguru import logger
from .service import evaluate_ragas
from .models import EvaluationRequest, EvaluationResult

router = APIRouter()

@router.post("/", response_model=EvaluationResult, tags=["Evaluation"])
async def evaluate(request: EvaluationRequest) -> EvaluationResult:
    """Endpoint that runs RAGAS evaluation on the provided texts.
    It delegates to ``evaluate_ragas`` which runs the synchronous ``ragas.evaluate``
    in a thread pool to keep the FastAPI event loop responsive.
    """
    try:
        result = await evaluate_ragas(request)
        logger.info("RAGAS evaluation completed successfully")
        return result
    except ValueError as ve:
        logger.warning(f"RAGAS evaluation input error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal evaluation error")
