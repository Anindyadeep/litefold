from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import logging
from models import Job, SessionLocal
from schemas import PredictionRequest, PredictionResponse, JobStatus, SuccessfulJobResponse
from pathlib import Path
import queue 
import threading
import modal
from typing import List
import numpy as np
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from Bio.PDB import PDBParser
import biotite.structure.io as bsio

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
modal_client = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_modal_client():
    global modal_client
    try:
        logger.info("Initializing Modal client...")
        cls = modal.Cls.from_name("litefold-serverless", "LiteFoldServer")
        modal_client = cls()
        logger.info("Modal client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Modal client: {str(e)}")
        return False

def process_prediction(job_id: str, sequence: str, user_id: str, db: Session):
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

        # Call the Modal function
        logger.info(f"Calling Modal function for sequence of length {len(sequence)}")
        result = modal_client.predict.remote(job_id=job_id, sequence=sequence, user_id=user_id)
        
        # Check if the prediction was successful
        if result.get("success", False):
            output_path = output_dir / f"{job_id}.pdb"
            
            # Save PDB content if available
            if "pdb_content" in result and result["pdb_content"]:
                with open(output_path, "w") as f:
                    f.write(result["pdb_content"])
                logger.info(f"Saved PDB content to {output_path}")
                job.result_path = str(output_path)
                
                try:
                    # The below code mimics the calculation done in native_gpu_backend's status endpoint
                    # Import necessary libraries here to avoid potential import errors
                    
                    # Calculate distogram
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
                        return dist_matrix

                    # Parse structure and get coordinates
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure("protein", output_path)
                    coords = get_ca_coordinates(structure)
                    distogram = calculate_distogram(coords)
                    
                    # Save distogram to file
                    distogram_path = output_dir / f"{job_id}_distogram.npy"
                    np.save(str(distogram_path), distogram)
                    logger.info(f"Calculated and saved distogram to {distogram_path}")
                    
                    # Calculate pLDDT score
                    struct = bsio.load_structure(str(output_path), extra_fields=["b_factor"])
                    plddt_score = float(struct.b_factor.mean())
                    
                    # Save pLDDT to file
                    plddt_path = output_dir / f"{job_id}_plddt.txt"
                    with open(plddt_path, "w") as f:
                        f.write(str(plddt_score))
                    logger.info(f"Calculated and saved pLDDT score to {plddt_path}")
                    
                except Exception as e:
                    logger.error(f"Error calculating/saving distogram or pLDDT: {str(e)}")
            # If pdb_content is not available, fallback to coordinates
            elif "coords" in result and result["coords"] is not None:
                try:
                    # Convert coordinates to PDB format
                    coords = np.array(result["coords"])
                    pdb_lines = []
                    
                    # Write PDB header
                    pdb_lines.append("HEADER    PROTEIN")
                    pdb_lines.append("TITLE     PREDICTED STRUCTURE")
                    
                    # Create ATOM records for CA atoms
                    atom_idx = 1
                    for i, coord in enumerate(coords):
                        residue_idx = i + 1
                        x, y, z = coord
                        atom_line = f"ATOM  {atom_idx:5d}  CA  GLY {' '}{residue_idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {result.get('plddt', 50):6.2f}      C  "
                        pdb_lines.append(atom_line)
                        atom_idx += 1
                    
                    pdb_lines.append("END")
                    
                    pdb_content = "\n".join(pdb_lines)
                    with open(output_path, "w") as f:
                        f.write(pdb_content)
                    
                    job.result_path = str(output_path)
                    logger.info(f"Created PDB file from coordinates at {output_path}")
                    
                    # Save distogram if available, otherwise calculate it
                    if "distogram" in result:
                        distogram = np.array(result["distogram"])
                        distogram_path = output_dir / f"{job_id}_distogram.npy"
                        np.save(str(distogram_path), distogram)
                        logger.info(f"Saved distogram from result to {distogram_path}")
                    
                    # Save pLDDT if available
                    if "plddt" in result:
                        plddt_path = output_dir / f"{job_id}_plddt.txt"
                        with open(plddt_path, "w") as f:
                            f.write(str(result["plddt"]))
                        logger.info(f"Saved pLDDT score to {plddt_path}")
                except Exception as e:
                    logger.error(f"Error creating PDB from coordinates: {str(e)}")
                    job.result_path = None
            else:
                job.result_path = None
                logger.warning("No PDB content or coordinates found in Modal response")
            
            # Update job status
            job.status = "successful"
            job.completed_at = datetime.now()
            db.commit()
            logger.info(f"Job {job_id} completed successfully")
        else:
            # Handle errors
            error_msg = result.get("error", "Unknown error during Modal prediction")
            logger.error(f"Modal prediction failed: {error_msg}")
            job.status = "crashed"
            job.error_message = error_msg
            job.completed_at = datetime.now()
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
            job_data = job_queue.get()
            
            # Support both dictionary and job_id string formats
            if isinstance(job_data, dict):
                job_id = job_data["job_id"]
                user_id = job_data.get("user_id")
                sequence = job_data.get("sequence")
                
                logger.info(f"Processing job {job_id} from queue with user {user_id}")
                
                # If sequence is not in job_data, get it from database
                if sequence is None:
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        sequence = job.sequence
                        user_id = job.user_id
                    else:
                        logger.error(f"Job {job_id} not found in database")
                        continue
                
                process_prediction(job_id, sequence, user_id, db)
            else:
                # Handle the native_gpu_backend style where only job_id is in queue
                job_id = job_data
                logger.info(f"Processing job {job_id} from queue")
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    process_prediction(job.job_id, job.sequence, job.user_id, db)
                else:
                    logger.error(f"Job {job_id} not found in database")
        except Exception as e:
            logger.error(f"Error in background worker: {str(e)}", exc_info=True)
        finally:
            job_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Modal client and start worker thread
    logger.info("Starting up FastAPI application")
    try:
        success = initialize_modal_client()
        if success:
            # Start background worker thread
            worker_thread = threading.Thread(target=background_worker, daemon=True)
            worker_thread.start()
            logger.info("Background worker started")
        else:
            logger.error("Failed to initialize Modal client")
    except Exception as e:
        logger.error(f"Failed during startup: {str(e)}")
    
    yield  # Server is running and handling requests here
    
    # Shutdown
    logger.info("Shutting down application")

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
    modal_status = "connected" if modal_client is not None else "not connected"
    
    status = {
        "status": "healthy" if modal_client is not None else "degraded",
        "modal_status": modal_status,
        "job_queue_size": job_queue.qsize(),
        # Add extra fields for consistency with GPU backend
        "backend_type": "modal",
        "model_status": "remote" if modal_client is not None else "not available"
    }
    
    return status

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    logger.info(f"Received prediction request for job {request.job_id} from user {request.user_id}")
    
    if modal_client is None:
        logger.error("Modal client not initialized, cannot process prediction request")
        raise HTTPException(status_code=503, detail="Modal service not available")

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

        # Using same format as native_gpu_backend.py - just put job_id in queue
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

        # If job is successful and has a result path, read the PDB content and other data
        if job.status == "successful" and job.result_path:
            output_dir = Path("results")
            
            # Read PDB content if file exists
            try:
                if Path(job.result_path).exists():
                    with open(job.result_path, "r") as f:
                        response.pdb_content = f.read()
            except Exception as e:
                logger.error(f"Error reading PDB file for job {job_id}: {str(e)}")
            
            # First try to read distogram from file
            distogram_loaded = False
            try:
                distogram_path = output_dir / f"{job_id}_distogram.npy"
                if distogram_path.exists():
                    distogram = np.load(str(distogram_path))
                    response.distogram = distogram.tolist()  # Convert to list for JSON
                    distogram_loaded = True
            except Exception as e:
                logger.error(f"Error reading distogram file for job {job_id}: {str(e)}")
            
            # If distogram wasn't loaded from file, calculate it from PDB
            if not distogram_loaded and Path(job.result_path).exists():
                try:
                    from Bio.PDB import PDBParser
                    
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
                    logger.info(f"Calculated distogram on-the-fly for job {job_id}")
                except Exception as e:
                    logger.error(f"Error calculating distogram for job {job_id}: {str(e)}")
            
            # First try to read pLDDT from file
            plddt_loaded = False
            try:
                plddt_path = output_dir / f"{job_id}_plddt.txt"
                if plddt_path.exists():
                    with open(plddt_path, "r") as f:
                        response.plddt_score = float(f.read().strip())
                        plddt_loaded = True
            except Exception as e:
                logger.error(f"Error reading pLDDT file for job {job_id}: {str(e)}")
            
            # If pLDDT wasn't loaded from file, calculate it from PDB
            if not plddt_loaded and Path(job.result_path).exists():
                try:
                    import biotite.structure.io as bsio
                    struct = bsio.load_structure(job.result_path, extra_fields=["b_factor"])
                    response.plddt_score = float(struct.b_factor.mean())
                    logger.info(f"Calculated pLDDT on-the-fly for job {job_id}")
                except Exception as e:
                    logger.error(f"Error calculating pLDDT for job {job_id}: {str(e)}")
        
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
    uvicorn.run(app, host="0.0.0.0", port=8020)
