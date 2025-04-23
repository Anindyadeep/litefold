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

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, BackgroundTasks, HTTPException
from datetime import datetime
from typing import List

from schemas import (
    PredictionRequest,
    PredictionResponse,
    JobStatus,
    SuccessfulJobResponse
)
from utils import get_ca_coordinates, calculate_distogram
from supabase_config import init_storage, upload_file, download_file

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
DEPLOYED_MODAL_APP_URL = os.getenv("DEPLOYED_MODAL_APP_URL")

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


def process_prediction(job_id: str, sequence: str, user_id: str):
    try:
        logger.info(f"Processing prediction for job {job_id[:6]}")
        update_job_status(job_id, "processing")

        logger.info(f"Calling Modal predict.remote for job {job_id[:6]}")
        result = requests.post(
            f"{DEPLOYED_MODAL_APP_URL}/predict",
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
                process_prediction(job["job_id"], job["sequence"], job["user_id"])
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
    allow_origins=["https://litefold.vercel.app"],
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
            user_id=job["user_id"]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8178)