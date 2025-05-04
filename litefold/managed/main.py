import os
import queue
import requests
import logging
import psycopg2
import psycopg2.extras
import biotite.structure.io as bsio
from Bio.PDB import PDBParser
from contextlib import asynccontextmanager
import threading
import tempfile
import uuid

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, BackgroundTasks, HTTPException
from datetime import datetime
from typing import List

from schemas import (
    PredictionRequest,
    PredictionResponse,
    JobStatus,
    SuccessfulJobResponse,
    DeleteJobRequest,
    DeleteJobResponse,
    TMScoreComparisionRequest,
    TMScoreComparisonResponse,
    ExperimentRecord,
    ExperimentNoteRequest,
    ExperimentNoteResponse
)
from utils import get_ca_coordinates, calculate_distogram, align_structures_and_calculate_tmscore, extract_sequence_from_pdb_content
from supabase_config import init_storage, upload_file, download_file, delete_file


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

job_queue = queue.Queue()

# Database connection parameters
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DEPLOY_MODELS_SITE_MAP = {
    "alphafold2": os.getenv("DEPLOYED_AF2_URL"),
    "esm3": os.getenv("DEPLOYED_ESM3_URL")
}

# Initialize Supabase storage
if not init_storage():
    raise RuntimeError("Failed to initialize Supabase storage")

def get_db_connection():
    try:
        connection = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            sslmode='require',   
            gssencmode='disable'
        )
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def get_job(job_id):
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM managed_db WHERE job_id = %s", (job_id,))
        job = cursor.fetchone()
        cursor.close()
        return dict(job) if job else None
    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {e}")
        raise
    finally:
        conn.close()

def create_job(job_data):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO managed_db 
            (job_id, job_name, model, sequence, status, user_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_data["job_id"],
                job_data["job_name"],
                job_data["model"],
                job_data["sequence"],
                job_data["status"],
                job_data["user_id"],
                datetime.now()
            )
        )
        conn.commit()
        cursor.close()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating job: {e}")
        raise
    finally:
        conn.close()

def update_job_status(job_id, status, result_path=None):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        updates = ["status = %s", "completed_at = %s"]
        params = [status, datetime.now()]
            
        if result_path is not None:
            updates.append("result_path = %s")
            params.append(result_path)
            
        query = f"UPDATE managed_db SET {', '.join(updates)} WHERE job_id = %s"
        params.append(job_id)
        
        cursor.execute(query, params)
        
        # Check how many rows were affected by the update
        rows_affected = cursor.rowcount
        logger.info(f"Update affected {rows_affected} rows for job {job_id[:6]}...")
        
        conn.commit()
        cursor.close()
        
        # Verify the update by fetching the job again
        check_conn = get_db_connection()
        try:
            check_cursor = check_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            check_cursor.execute("SELECT status FROM managed_db WHERE job_id = %s", (job_id,))
            job = check_cursor.fetchone()
            check_cursor.close()
            if job:
                logger.info(f"Verified job {job_id[:6]} status is now: {job['status']}")
            else:
                logger.error(f"Job {job_id} not found after update!")
        finally:
            check_conn.close()
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating job status: {e}")
        raise
    finally:
        conn.close()
        
def get_successful_jobs(user_id):
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(
            "SELECT * FROM managed_db WHERE status = 'successful' AND user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        jobs = cursor.fetchall()
        cursor.close()
        return [dict(job) for job in jobs]
    except Exception as e:
        logger.error(f"Error fetching successful jobs: {e}")
        raise
    finally:
        conn.close()

def delete_job(job_id: str, user_id: str):
    conn = get_db_connection()
    try:
        # First get the job to verify ownership and get the result path if available
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(
            "SELECT * FROM managed_db WHERE job_id = %s AND user_id = %s",
            (job_id, user_id)
        )
        job = cursor.fetchone()
        
        if not job:
            logger.warning(f"Job {job_id} not found or does not belong to user {user_id}")
            return False, "Job not found or unauthorized"
            
        # If there's a result file in storage, delete it
        if job["result_path"]:
            logger.info(f"Deleting file for job {job_id} from storage")
            if not delete_file(user_id, job_id):
                logger.warning(f"Failed to delete file for job {job_id} from storage")
        
        # Now delete the job from the database
        cursor.execute(
            "DELETE FROM managed_db WHERE job_id = %s AND user_id = %s",
            (job_id, user_id)
        )
        conn.commit()
        
        rows_affected = cursor.rowcount
        cursor.close()
        
        if rows_affected > 0:
            logger.info(f"Successfully deleted job {job_id} for user {user_id}")
            return True, None
        else:
            logger.warning(f"No rows affected when deleting job {job_id}")
            return False, "Failed to delete job record"
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting job {job_id}: {e}")
        return False, str(e)
    finally:
        conn.close()

def process_prediction(job_id: str, sequence: str, user_id: str, model: str):
    try:
        logger.info(f"Processing prediction for job {job_id[:6]}")
        update_job_status(job_id, "processing")

        logger.info(f"Calling Modal predict.remote for job {job_id[:6]}")
        base_url = DEPLOY_MODELS_SITE_MAP[model]
        result = requests.post(
            f"{base_url}/predict",
            json={
                "user_id": user_id,
                "job_id": job_id, 
                "sequence": sequence
            }
        ).json()

        if result.get("success", False):
            logger.info(f"Modal job {job_id[:6]} successful, saving results")
            if "pdb_content" in result and result["pdb_content"]:
                try:
                    # Upload to Supabase storage using helper function
                    file_content = result["pdb_content"].encode('utf-8')
                    file_url = upload_file(user_id, job_id, file_content)
                    
                    if file_url:
                        logger.info(f"PDB content uploaded to Supabase storage: {file_url}")
                        update_job_status(
                            job_id=job_id,
                            status="successful",
                            result_path=file_url,
                        )
                        logger.info(f"Job {job_id[:6]} status updated to 'successful'")
                    else:
                        raise Exception("Failed to get file URL after upload")
                        
                except Exception as upload_err:
                    logger.error(f"Error uploading to Supabase: {str(upload_err)}")
                    update_job_status(
                        job_id=job_id, 
                        status="crashed",
                        result_path=None
                    )
            else:
                logger.error(f"No PDB content in Modal result for job {job_id}")
                update_job_status(
                    job_id=job_id,
                    status="crashed",
                    result_path=None
                )
        else:
            logger.error(f"Modal job {job_id} failed")
            update_job_status(
                job_id=job_id,
                status="crashed",
                result_path=None
            )

    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        update_job_status(
            job_id=job_id,
            status="crashed",
            result_path=None
        )

def background_worker():
    logger.info("Starting background worker")
    while True:
        try:
            job_id = job_queue.get()
            logger.info(f"Processing job {job_id[:6]} from queue")
            job = get_job(job_id=job_id)
            if job:
                process_prediction(
                    job["job_id"], job["sequence"], job["user_id"], job["model"]
                )
            else:
                logger.error(f"Job {job_id} not found in database")
        except Exception as e:
            logger.error(f"Error in background worker: {str(e)}", exc_info=True)
        finally:
            job_queue.task_done()
            
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application")
    try:
        worker_thread = threading.Thread(target=background_worker, daemon=True)
        worker_thread.start()
        logger.info("Background worker started")
        
    except Exception as e:
        logger.error(f"Failed during startup: {str(e)}")
    yield 
    logger.info("Shutting down application")


# Create FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)   
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    status = {
        "status": "healthy",
        "database_status": "connected" if DB_USER and DB_PASSWORD and DB_HOST and DB_PORT and DB_NAME else "not configured"
    }
    return status

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(f"Received prediction request for job {request.job_id} from user {request.user_id}")
    try:
        job_data = {
            "job_id": request.job_id,
            "job_name": request.job_name,
            "model": request.model,
            "sequence": request.sequence,
            "status": "pending",
            "user_id": request.user_id
        }

        create_job(job_data)
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
async def get_status(job_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(
            "SELECT * FROM managed_db WHERE job_id = %s",
            (job_id,)
        )
        job = cursor.fetchone()
        cursor.close()
        conn.close()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        response = JobStatus(
            job_id=job["job_id"],
            job_name=job["job_name"],
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
            error_message=job.get("error_message"),
            user_id=job["user_id"],
            model=job["model"]
        )

        if job["status"] == "successful" and job["result_path"]:
            try:
                file_data = download_file(job["user_id"], job_id)
                
                if file_data:
                    pdb_content = file_data.decode('utf-8')
                    response.pdb_content = pdb_content

                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_file:
                        temp_file.write(pdb_content)
                        temp_file.flush()
                        
                        try:
                            parser = PDBParser(QUIET=True)
                            structure = parser.get_structure("protein", temp_file.name)
                            coords = get_ca_coordinates(structure)
                            response.distogram = calculate_distogram(coords)
                            struct = bsio.load_structure(temp_file.name, extra_fields=["b_factor"])
                            response.plddt_score = float(struct.b_factor.mean())
                        finally:
                            os.unlink(temp_file.name)
                else:
                    logger.error(f"Failed to download file for job {job_id}")

            except Exception as e:
                logger.error(f"Error fetching PDB content from Supabase: {str(e)}")
                pass

        return response

    except Exception as e:
        logger.error(f"Error checking status for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/successful-jobs/{user_id}", response_model=List[SuccessfulJobResponse])
async def list_successful_jobs(user_id: str):
    logger.info(f"Fetching list of successful jobs for user {user_id}")
    try:
        successful_jobs = get_successful_jobs(user_id)
        return [SuccessfulJobResponse(
            job_id=job["job_id"],
            job_name=job["job_name"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
            result_path=job["result_path"],
            user_id=job["user_id"]
        ) for job in successful_jobs]
    except Exception as e:
        logger.error(f"Error fetching successful jobs for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-job", response_model=DeleteJobResponse)
async def delete_job_endpoint(request: DeleteJobRequest):
    logger.info(f"Received request to delete job {request.job_id} from user {request.user_id}")
    try:
        success, error_message = delete_job(request.job_id, request.user_id)
        
        if success:
            return DeleteJobResponse(
                job_id=request.job_id,
                user_id=request.user_id,
                success=True
            )
        else:
            return DeleteJobResponse(
                job_id=request.job_id,
                user_id=request.user_id,
                success=False,
                error_message=error_message
            )
    except Exception as e:
        logger.error(f"Error deleting job {request.job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def create_experiment_record(experiment_data):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments 
            (experiment_id, user_id, experiment_type, compare_to_job_id, compare_with_job_id, 
            compare_with_file_name, model, tm_score, rmsd, compare_to_sequence, compare_with_sequence, 
            success, error_message, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                experiment_data["experiment_id"],
                experiment_data["user_id"],
                experiment_data["experiment_type"],
                experiment_data["compare_to_job_id"],
                experiment_data["compare_with_job_id"],
                experiment_data["compare_with_file_name"],
                experiment_data["model"],
                experiment_data["tm_score"],
                experiment_data["rmsd"],
                experiment_data["compare_to_sequence"],
                experiment_data["compare_with_sequence"],
                experiment_data["success"],
                experiment_data["error_message"],
                datetime.now()
            )
        )
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating experiment record: {e}")
        raise
    finally:
        conn.close()

@app.post("/compare-structures", response_model=TMScoreComparisonResponse)
async def compare_structures(request: TMScoreComparisionRequest):
    logger.info(f"Received structure comparison request for job {request.compare_to_job_id} from user {request.user_id}")
    try:
        # Generate an experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Get the predicted structure (compare_to_job_id)
        logger.info(f"Looking up job {request.compare_to_job_id}")
        predict_job = get_job(request.compare_to_job_id)
        if not predict_job:
            logger.error(f"Job {request.compare_to_job_id} not found in database")
            raise HTTPException(status_code=404, detail=f"Job {request.compare_to_job_id} not found")
        
        logger.info(f"Job status: {predict_job['status']}, Result path: {predict_job.get('result_path')}")
        if predict_job["status"] != "successful" or not predict_job["result_path"]:
            logger.error(f"Job {request.compare_to_job_id} not successful or no results available")
            raise HTTPException(status_code=400, detail=f"Job {request.compare_to_job_id} not successful or no results available")
        
        # Download the predicted structure
        logger.info(f"Downloading PDB content for job {request.compare_to_job_id}")
        try:
            predicted_pdb_content = download_file(request.user_id, request.compare_to_job_id).decode('utf-8')
            logger.info(f"Successfully downloaded PDB content for {request.compare_to_job_id}, length: {len(predicted_pdb_content)}")
        except Exception as e:
            logger.error(f"Error downloading file for job {request.compare_to_job_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")
        
        # Get the ground truth structure
        ground_truth_pdb_content = None
        compare_with_sequence = None
        compare_with_file_name = None
        
        if request.compare_with_job_id:
            # Compare with another prediction
            logger.info(f"Looking up comparison job {request.compare_with_job_id}")
            compare_job = get_job(request.compare_with_job_id)
            if not compare_job:
                logger.error(f"Job {request.compare_with_job_id} not found in database")
                raise HTTPException(status_code=404, detail=f"Job {request.compare_with_job_id} not found")
                
            logger.info(f"Comparison job status: {compare_job['status']}, Result path: {compare_job.get('result_path')}")
            if compare_job["status"] != "successful" or not compare_job["result_path"]:
                logger.error(f"Job {request.compare_with_job_id} not successful or no results available")
                raise HTTPException(status_code=400, detail=f"Job {request.compare_with_job_id} not successful or no results available")
                
            try:
                logger.info(f"Downloading PDB content for comparison job {request.compare_with_job_id}")
                ground_truth_pdb_content = download_file(request.user_id, request.compare_with_job_id).decode('utf-8')
                logger.info(f"Successfully downloaded comparison PDB content, length: {len(ground_truth_pdb_content)}")
                compare_with_sequence = compare_job["sequence"]
                compare_with_file_name = None
            except Exception as e:
                logger.error(f"Error downloading file for comparison job {request.compare_with_job_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error downloading comparison file: {str(e)}")
        elif request.compare_with_file_content:
            # Use directly provided file content instead of downloading from bucket
            logger.info("Using provided file content for comparison")
            ground_truth_pdb_content = request.compare_with_file_content
            # Extract sequence from PDB file
            try:
                compare_with_sequence = extract_sequence_from_pdb_content(ground_truth_pdb_content)
                logger.info(f"Extracted sequence from provided PDB file: {compare_with_sequence[:20]}...")
            except Exception as e:
                logger.error(f"Failed to extract sequence from provided PDB file: {str(e)}")
                compare_with_sequence = "Unable to extract sequence from PDB file"
            compare_with_file_name = f"direct_content_{experiment_id}"  # Generate a unique identifier for direct content
        elif request.compare_with_file_name:
            # Compare with user-uploaded file
            logger.info(f"Looking up file {request.compare_with_file_name}")
            try:
                ground_truth_pdb_content = download_file(request.user_id, request.compare_with_file_name).decode('utf-8')
                logger.info(f"Successfully downloaded file {request.compare_with_file_name}, length: {len(ground_truth_pdb_content)}")
                # Extract sequence from PDB file
                try:
                    compare_with_sequence = extract_sequence_from_pdb_content(ground_truth_pdb_content)
                    logger.info(f"Extracted sequence from uploaded PDB file: {compare_with_sequence[:20]}...")
                except Exception as e:
                    logger.error(f"Failed to extract sequence from uploaded PDB file: {str(e)}")
                    compare_with_sequence = "Unable to extract sequence from PDB file"
                compare_with_file_name = request.compare_with_file_name
            except Exception as e:
                logger.error(f"Error downloading file {request.compare_with_file_name}: {str(e)}")
                raise HTTPException(status_code=404, detail=f"File {request.compare_with_file_name} not found: {str(e)}")
        else:
            logger.error("No comparison source provided")
            raise HTTPException(status_code=400, detail="Either compare_with_job_id, compare_with_file_content, or compare_with_file_name must be provided")
        
        # Make sure we have valid PDB content for both structures
        if not predicted_pdb_content or len(predicted_pdb_content) < 100:
            logger.error(f"Invalid or empty PDB content for job {request.compare_to_job_id}")
            raise HTTPException(status_code=400, detail=f"Invalid or empty PDB content for job {request.compare_to_job_id}")
            
        if not ground_truth_pdb_content or len(ground_truth_pdb_content) < 100:
            logger.error("Invalid or empty comparison PDB content")
            raise HTTPException(status_code=400, detail="Invalid or empty comparison PDB content")
        
        # Perform the alignment and TM-score calculation
        try:
            logger.info("Performing alignment and TM-score calculation")
            result = align_structures_and_calculate_tmscore(
                predicted_pdb_content=predicted_pdb_content,
                ground_truth_pdb_content=ground_truth_pdb_content
            )
            logger.info(f"Alignment successful, TM-score: {result['tm_score']}, RMSD: {result['rmsd']}")
            
            # Upload the aligned structure
            logger.info("Uploading aligned structure")
            aligned_file_content = result["aligned_pdb_content"].encode('utf-8')
            aligned_file_url = upload_file(request.user_id, f"aligned_{experiment_id}", aligned_file_content)
            logger.info(f"Aligned structure uploaded successfully: {aligned_file_url}")
            
            # Create experiment record
            logger.info("Creating experiment record")
            experiment_data = {
                "experiment_id": experiment_id,
                "user_id": request.user_id,
                "experiment_type": "comparison",
                "compare_to_job_id": request.compare_to_job_id,
                "compare_with_job_id": request.compare_with_job_id,
                "compare_with_file_name": compare_with_file_name if not request.compare_with_job_id else None,
                "model": request.model,
                "tm_score": result["tm_score"],
                "rmsd": result["rmsd"],
                "compare_to_sequence": predict_job["sequence"],
                "compare_with_sequence": compare_with_sequence,
                "success": True,
                "error_message": None
            }
            
            create_experiment_record(experiment_data)
            logger.info(f"Experiment record created with ID: {experiment_id}")
            
            # Create response
            aligned_job_id = request.compare_with_job_id if request.compare_with_job_id else (compare_with_file_name or "direct_content")
            return TMScoreComparisonResponse(
                user_id=request.user_id,
                experiment_id=experiment_id,
                aligned_pdb_content=result["aligned_pdb_content"],
                aligned_job_id=aligned_job_id,
                tm_score=result["tm_score"],
                rmsd=result["rmsd"],
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error performing alignment: {str(e)}", exc_info=True)
            # Create failed experiment record
            experiment_data = {
                "experiment_id": experiment_id,
                "user_id": request.user_id,
                "experiment_type": "comparison",
                "compare_to_job_id": request.compare_to_job_id,
                "compare_with_job_id": request.compare_with_job_id,
                "compare_with_file_name": compare_with_file_name if not request.compare_with_job_id else None,
                "model": request.model,
                "tm_score": 0,
                "rmsd": None,
                "compare_to_sequence": predict_job["sequence"],
                "compare_with_sequence": compare_with_sequence,
                "success": False,
                "error_message": str(e)
            }
            
            create_experiment_record(experiment_data)
            logger.info(f"Created failed experiment record with ID: {experiment_id}")
            
            raise HTTPException(status_code=500, detail=f"Error performing alignment: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing structures: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/experiments/{user_id}", response_model=List[ExperimentRecord])
async def get_experiments(user_id: str):
    logger.info(f"Fetching experiments for user {user_id}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(
            "SELECT * FROM experiments WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        experiments = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [ExperimentRecord(**dict(exp)) for exp in experiments]
    except Exception as e:
        logger.error(f"Error fetching experiments for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def update_experiment_note(experiment_id: str, user_id: str, note: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Check if the experiment exists and belongs to the user
        cursor.execute(
            "SELECT * FROM experiments WHERE experiment_id = %s AND user_id = %s",
            (experiment_id, user_id)
        )
        experiment = cursor.fetchone()
        
        if not experiment:
            logger.warning(f"Experiment {experiment_id} not found or does not belong to user {user_id}")
            return False, "Experiment not found or unauthorized"
        
        # Update the notes column
        cursor.execute(
            "UPDATE experiments SET notes = %s WHERE experiment_id = %s AND user_id = %s",
            (note, experiment_id, user_id)
        )
        conn.commit()
        
        rows_affected = cursor.rowcount
        cursor.close()
        
        if rows_affected > 0:
            logger.info(f"Successfully updated notes for experiment {experiment_id}")
            return True, None
        else:
            logger.warning(f"No rows affected when updating notes for experiment {experiment_id}")
            return False, "Failed to update experiment notes"
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating notes for experiment {experiment_id}: {str(e)}")
        return False, str(e)
    finally:
        conn.close()

@app.post("/update-experiment-note", response_model=ExperimentNoteResponse)
async def update_note_endpoint(request: ExperimentNoteRequest):
    logger.info(f"Received request to update notes for experiment {request.experiment_id} from user {request.user_id}")
    try:
        success, error_message = update_experiment_note(request.experiment_id, request.user_id, request.note)
        
        return ExperimentNoteResponse(
            experiment_id=request.experiment_id,
            user_id=request.user_id,
            note=request.note,
            success=success,
            error_message=error_message
        )
    except Exception as e:
        logger.error(f"Error updating notes for experiment {request.experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8178)