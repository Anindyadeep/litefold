from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import numpy as np

class PredictionRequest(BaseModel):
    job_name: str
    job_id: str
    model: str = Field(default="alphafold2", pattern="^(alphafold2|esm3)$")
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
    distogram: Optional[dict] = None
    plddt_score: Optional[float] = None
    user_id: str
    model: str

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

class DeleteJobRequest(BaseModel):
    job_id: str
    user_id: str

class DeleteJobResponse(BaseModel):
    job_id: str
    user_id: str
    success: bool
    error_message: Optional[str] = None


class TMScoreComparisionRequest(BaseModel):
    compare_to_job_id: str # The job ID of the structure to compare with
    compare_with_job_id: Optional[str]=None # The job ID of the structure to compare to
    compare_with_file_name: Optional[str]=None 
    compare_with_file_content: Optional[str]=None 
    user_id: str
    model: str = Field(default="alphafold2", pattern="^(alphafold2|esm3)$")

    
class TMScoreComparisonResponse(BaseModel):
    user_id: str
    experiment_id: str  
    aligned_pdb_content: str
    aligned_job_id: str  # Job ID which got aligned with the compare_to_job_id
    tm_score: float
    rmsd: float
    success: bool 
    error_message: Optional[str] = None

class ExperimentRecord(BaseModel):
    experiment_id: str
    user_id: str
    experiment_type: str = "comparison"
    compare_to_job_id: str
    compare_with_job_id: Optional[str] = None
    compare_with_file_name: Optional[str] = None
    model: str = Field(default="alphafold2", pattern="^(alphafold2|esm3)$")
    compare_to_sequence: str
    compare_with_sequence: str
    tm_score: float
    rmsd: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class ExperimentResponse(BaseModel):
    experiment_id: str
    user_id: str

class ExperimentNoteRequest(BaseModel):
    experiment_id: str
    user_id: str
    note: str

class ExperimentNoteResponse(BaseModel):
    experiment_id: str
    user_id: str
    note: str
    success: bool
    error_message: Optional[str] = None