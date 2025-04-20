import modal
from pathlib import Path
import torch
import logging
from typing import List
import numpy as np
from Bio.PDB import PDBParser
import biotite.structure.io as bsio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Modal stub
stub = modal.Stub("esmfold-lite")

# Create volume for storing results
volume = modal.Volume.from_name("esmfold-results", create_if_missing=True)

# Create shared dictionary for job tracking
jobs_dict = modal.Dict.from_name("esmfold-jobs", create_if_missing=True)

# Define the image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "biotite",
        "biopython",
        "fastapi",
        "sqlalchemy",
        "pydantic",
        "uvicorn",
    )
    .pip_install("fair-esm", index_url="https://download.pytorch.org/whl/cu118")
)

@stub.cls(
    image=image,
    gpu="A10G",
    container_idle_timeout=300,  # 5 minutes
    volumes={"/results": volume}
)
class ESMFoldModel:
    def __enter__(self):
        from fold_models import ESMFold

        # Load model
        logger.info("Starting model loading process...")
        model_name = "esmfold_3B_v1"
        
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        logger.info(f"Downloading model from: {url}")
        model_data = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")

        cfg = model_data["cfg"]["model"]
        model_state = model_data["model"]
        
        logger.info("Initializing ESMFold model...")
        self.model = ESMFold(esmfold_config=cfg)
        self.model.load_state_dict(model_state, strict=False)
        
        logger.info("Moving model to CUDA")
        self.model = self.model.eval().cuda()
        
        if next(self.model.parameters()).is_cuda:
            logger.info("Model successfully loaded on GPU")
        else:
            raise RuntimeError("Model failed to move to GPU")

    @modal.method()
    def predict(self, job_id: str, sequence: str, user_id: str):
        try:
            logger.info(f"Processing prediction for job {job_id}")
            
            # Update job status
            jobs_dict[job_id] = {
                "status": "processing",
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "sequence": sequence
            }
            
            output_dir = Path("/results")
            output_dir.mkdir(exist_ok=True)

            logger.info(f"Running inference for sequence of length {len(sequence)}")
            with torch.no_grad():
                output = self.model.infer_pdb(sequence)
            
            # Save result
            output_path = output_dir / f"{job_id}.pdb"
            with open(output_path, "w") as f:
                f.write(output)
            
            logger.info(f"Successfully saved results to {output_path}")
            
            # Update job status
            jobs_dict[job_id].update({
                "status": "successful",
                "completed_at": datetime.now().isoformat(),
                "result_path": str(output_path)
            })

            return {"status": "successful", "job_id": job_id}

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
            jobs_dict[job_id].update({
                "status": "crashed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat()
            })
            return {"status": "crashed", "error": str(e)}

    @modal.method()
    def get_job_status(self, job_id: str):
        job = jobs_dict.get(job_id)
        if not job:
            return {"error": "Job not found"}
        
        response = {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "user_id": job["user_id"]
        }

        if job["status"] == "successful" and "result_path" in job:
            # Read PDB content
            with open(job["result_path"], "r") as f:
                pdb_content = f.read()
            response["pdb_content"] = pdb_content

            # Calculate distogram and pLDDT
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", job["result_path"])
            
            # Get CA coordinates
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca = residue['CA'].get_vector().get_array()
                            coords.append(ca)
            coords = np.array(coords)
            
            # Calculate distogram
            dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            response["distogram"] = dist_matrix.tolist()

            # Calculate pLDDT score
            struct = bsio.load_structure(job["result_path"], extra_fields=["b_factor"])
            response["plddt_score"] = float(struct.b_factor.mean())

        return response

    @modal.method()
    def list_successful_jobs(self, user_id: str) -> List[dict]:
        successful_jobs = []
        for job_id, job_data in jobs_dict.items():
            if job_data["status"] == "successful" and job_data["user_id"] == user_id:
                successful_jobs.append({
                    "job_id": job_id,
                    "created_at": job_data["created_at"],
                    "completed_at": job_data["completed_at"],
                    "result_path": job_data.get("result_path"),
                    "user_id": user_id
                })
        return successful_jobs

@stub.function(
    image=image,
    volumes={"/results": volume}
)
@modal.web_endpoint()
def app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    web_app = FastAPI()
    model = ESMFoldModel()

    class PredictionRequest(BaseModel):
        job_id: str
        sequence: str
        user_id: str
        job_name: str = None
        model: str = "esmfold_3B_v1"

    @web_app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "gpu_status": "available",
            "model_status": "loaded"
        }

    @web_app.post("/predict")
    async def predict(request: PredictionRequest):
        try:
            result = model.predict.remote(
                job_id=request.job_id,
                sequence=request.sequence,
                user_id=request.user_id
            )
            return {
                "job_id": request.job_id,
                "status": result["status"],
                "message": "Job submitted successfully"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/status/{job_id}")
    async def get_status(job_id: str):
        try:
            return model.get_job_status.remote(job_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/successful-jobs/{user_id}")
    async def list_successful_jobs(user_id: str):
        try:
            return model.list_successful_jobs.remote(user_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

if __name__ == "__main__":
    stub.serve()
