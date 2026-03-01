from pydantic import BaseModel
from typing import List, Optional

class WizardStep(BaseModel):
    id: str
    title: str
    description: str
    status: str = "pending" # pending, active, completed, error
    component: str # Name der Frontend-Komponente, die angezeigt werden soll

class WizardState(BaseModel):
    current_step_index: int
    steps: List[WizardStep]
    is_running: bool

class SystemCheckResult(BaseModel):
    component: str
    status: str # "ok", "error"
    message: str
    details: Optional[str] = None

class SystemCheckResponse(BaseModel):
    overall_status: str # "ok", "warning", "error"
    checks: List[SystemCheckResult]
