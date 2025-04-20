from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import numpy as np

class PredictionRequest(BaseModel):
    job_name: str
    job_id: str
    model: str = Field("esmfold")
    sequence: str
    user_id: str

class PredictionResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    job_name: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    pdb_content: Optional[str] = None
    distogram: Optional[list] = None
    plddt_score: Optional[float] = None
    user_id: str

# Add this new model before the endpoints
class SuccessfulJobResponse(BaseModel):
    job_id: str
    job_name: str
    created_at: datetime
    completed_at: datetime | None
    result_path: str | None
    user_id: str

    class Config:
        from_attributes = True