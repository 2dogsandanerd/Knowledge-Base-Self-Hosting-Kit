from pydantic import BaseModel

class ModelConfig(BaseModel):
    """Persisted LLM model selection."""
    model_name: str
