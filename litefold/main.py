from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import torch
import logging
from models import Job, SessionLocal
from schemas import PredictionRequest, PredictionResponse, JobStatus, SuccessfulJobResponse
from pathlib import Path
import queue 
import threading
from fold_models import ESMFold
from constants import CUDA_DEVICE
from contextlib import asynccontextmanager
from Bio.PDB import PDBParser
import numpy as np
import biotite.structure.io as bsio
from typing import List
from pydantic import BaseModel

# Add cors headers
from fastapi.middleware.cors import CORSMiddleware


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables
job_queue = queue.Queue()
model = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_gpu():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Please check your GPU setup.")
        return False
    
    try:
        # Test CUDA device
        test_tensor = torch.tensor([1.0], device=CUDA_DEVICE)
        logger.info(f"Successfully created test tensor on {CUDA_DEVICE}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(CUDA_DEVICE)}")
        return True
    except Exception as e:
        logger.error(f"Error testing CUDA device: {str(e)}")
        return False

def load_model():
    global model
    if model is None:
        try:
            logger.info("Starting model loading process...")
            model_name = "esmfold_3B_v1"
            
            # Check GPU first
            if not check_gpu():
                raise RuntimeError("GPU check failed")

            if model_name.endswith(".pt"):
                model_path = Path(model_name)
                logger.info(f"Loading model from local path: {model_path}")
                model_data = torch.load(str(model_path), map_location="cpu")
            else:
                url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
                logger.info(f"Downloading model from: {url}")
                model_data = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")

            cfg = model_data["cfg"]["model"]
            model_state = model_data["model"]
            
            logger.info("Initializing ESMFold model...")
            model = ESMFold(esmfold_config=cfg)
            model.load_state_dict(model_state, strict=False)
            
            logger.info(f"Moving model to device: {CUDA_DEVICE}")
            model = model.eval().to(CUDA_DEVICE)
            
            # Verify model is on GPU
            if next(model.parameters()).is_cuda:
                logger.info("Model successfully loaded on GPU")
            else:
                raise RuntimeError("Model failed to move to GPU")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def process_prediction(job_id: str, sequence: str, db: Session):
    try:
        logger.info(f"Processing prediction for job {job_id}")
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return
            
        job.status = "processing"
        db.commit()
        
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Running inference for sequence of length {len(sequence)}")
        with torch.no_grad():
            output = model.infer_pdb(sequence)
        
        # Save result
        output_path = output_dir / f"{job_id}.pdb"
        with open(output_path, "w") as f:
            f.write(output)
        
        logger.info(f"Successfully saved results to {output_path}")
        job.status = "successful"
        job.completed_at = datetime.now()
        job.result_path = str(output_path)
        db.commit()

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        job.status = "crashed"
        job.error_message = str(e)
        job.completed_at = datetime.now()
        db.commit()

def background_worker():
    logger.info("Starting background worker")
    db = SessionLocal()
    while True:
        try:
            job_id = job_queue.get()
            logger.info(f"Processing job {job_id} from queue")
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                process_prediction(job.job_id, job.sequence, db)
            else:
                logger.error(f"Job {job_id} not found in database")
        except Exception as e:
            logger.error(f"Error in background worker: {str(e)}", exc_info=True)
        finally:
            job_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and start worker thread
    logger.info("Starting up FastAPI application")
    try:
        load_model()
        logger.info("Model loaded successfully")
        
        # Start background worker thread
        worker_thread = threading.Thread(target=background_worker, daemon=True)
        worker_thread.start()
        logger.info("Background worker started")
        
    except Exception as e:
        logger.error(f"Failed during startup: {str(e)}")
        # We don't raise here to allow the API to start, but model-dependent endpoints will fail
    
    yield  # Server is running and handling requests here
    
    # Shutdown: Cleanup resources
    logger.info("Shutting down application")
    try:
        # Clear model from GPU
        global model
        if model is not None:
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model cleared from GPU")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Create FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)    

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    gpu_status = "available" if torch.cuda.is_available() else "not available"
    model_status = "loaded" if model is not None else "not loaded"
    model_device = next(model.parameters()).device if model is not None else "N/A"
    
    status = {
        "status": "healthy" if model is not None and torch.cuda.is_available() else "degraded",
        "gpu_status": gpu_status,
        "model_status": model_status,
        "model_device": str(model_device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        status["cuda_device_name"] = torch.cuda.get_device_name(CUDA_DEVICE)
        status["cuda_memory_allocated"] = torch.cuda.memory_allocated(CUDA_DEVICE)
        status["cuda_memory_reserved"] = torch.cuda.memory_reserved(CUDA_DEVICE)
    
    return status

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    logger.info(f"Received prediction request for job {request.job_id} from user {request.user_id}")
    
    if model is None:
        logger.error("Model not loaded, cannot process prediction request")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        job = Job(
            job_id=request.job_id,
            job_name=request.job_name,
            model=request.model,
            sequence=request.sequence,
            status="pending",
            user_id=request.user_id
        )

        db.add(job)
        db.commit()
        logger.info(f"Created job record for {request.job_id} for user {request.user_id}")

        # Add job to queue
        job_queue.put(request.job_id)
        logger.info(f"Added job {request.job_id} to processing queue")
        
        return PredictionResponse(
            job_id=request.job_id,
            status="pending",
            message="Job submitted successfully"
        )
    except Exception as e:
        logger.error(f"Error submitting job {request.job_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str, db: Session = Depends(get_db)):
    logger.info(f"Checking status for job {job_id}")
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.warning(f"Job {job_id} not found")
            raise HTTPException(status_code=404, detail="Job not found")
        
        response = JobStatus(
            job_id=job.job_id,
            job_name=job.job_name,
            status=job.status,
            created_at=job.created_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            user_id=job.user_id
        )

        # If job is successful, add the additional information
        if job.status == "successful" and job.result_path:
            # Read PDB content
            with open(job.result_path, "r") as f:
                pdb_content = f.read()
            response.pdb_content = pdb_content

            # Calculate distogram and pLDDT
            def get_ca_coordinates(structure):
                coords = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if 'CA' in residue:
                                ca = residue['CA'].get_vector().get_array()
                                coords.append(ca)
                return np.array(coords)

            def calculate_distogram(coords):
                dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                return dist_matrix.tolist()  # Convert to list for JSON serialization

            # Parse structure and get coordinates
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", job.result_path)
            coords = get_ca_coordinates(structure)
            response.distogram = calculate_distogram(coords)

            # Calculate pLDDT score
            struct = bsio.load_structure(job.result_path, extra_fields=["b_factor"])
            response.plddt_score = float(struct.b_factor.mean())  # Convert to float for JSON serialization
        
        return response
    except Exception as e:
        logger.error(f"Error checking status for job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/successful-jobs/{user_id}", response_model=List[SuccessfulJobResponse])
async def list_successful_jobs(user_id: str, db: Session = Depends(get_db)):
    logger.info(f"Fetching list of successful jobs for user {user_id}")
    try:
        successful_jobs = db.query(Job).filter(
            Job.status == "successful",
            Job.user_id == user_id
        ).all()
        # Return empty list if no jobs found
        return [SuccessfulJobResponse(
            job_id=job.job_id,
            job_name=job.job_name,
            created_at=job.created_at,
            completed_at=job.completed_at,
            result_path=job.result_path,
            user_id=job.user_id
        ) for job in successful_jobs]
    except Exception as e:
        logger.error(f"Error fetching successful jobs for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)