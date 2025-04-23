import modal
import logging
import requests
from pathlib import Path
import torch
from tqdm.auto import tqdm
from typing import Dict, Any

from fold_models import ESMFold
logger = logging.getLogger(__name__)

MODEL_DIR = "/models"
RESULTS_DIR = "/results"
DEST_PATH = Path(MODEL_DIR) / "litefold_3B"
MODEL_NAME = "esmfold_3B_v1"
MINUTES = 60

def load_esmfold_model():
    logger.info("Starting model loading process ...")
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{MODEL_NAME}.pt"
    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEST_PATH.exists():
        logger.info(f"Downloading {url} to {DEST_PATH}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(DEST_PATH, "wb") as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc='Downloading model') as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        logger.info(f"File already exists at {DEST_PATH}")


app = modal.App("litefold-serverless")
model_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "biotite",
        "biopython",
        "fastapi[standard]",
        "sqlalchemy",
        "omegaconf"
    )
    .pip_install("fair-esm[esmfold]")
)
model_volume = modal.Volume.from_name("model", create_if_missing=True)
results_volume = modal.Volume.from_name("results", create_if_missing=True)


@app.cls(
    image=model_image,
    gpu="any",
    volumes={MODEL_DIR: model_volume, RESULTS_DIR: results_volume},
    timeout= 10 * MINUTES,
    container_idle_timeout=600,
)
class ESMFoldServer:
    @modal.enter()
    def load_model(self):
        load_esmfold_model()
        model_data = torch.load(
            str(DEST_PATH), map_location="cpu", weights_only=False
        )
        cfg = model_data["cfg"]["model"]
        model_state = model_data["model"]
        model = ESMFold(esmfold_config=cfg)

        expected_keys = set(model.state_dict().keys())
        found_keys = set(model_state.keys())
        missing_essential_keys = [
            k for k in expected_keys - found_keys if not k.startswith("esm.")
        ]
        if missing_essential_keys:
            raise RuntimeError(
                f"Keys '{', '.join(missing_essential_keys)}' are missing."
            )
        model.load_state_dict(model_state, strict=False)
        logger.info("Push the model weights to GPU")
        self.model = model.cuda()

    
    @modal.method()
    def predict(self, user_id: str, job_id: str, sequence: str) -> Dict[str, Any]:
        try:
            with torch.no_grad():
                output = self.model.infer_pdb(sequence)

            return {
                "user_id": user_id,
                "success": True,
                "job_id": job_id,
                "pdb_content": output
            }
        except Exception as e:
            import traceback
            logger.error(f"Internal Server error: {e}\nTraceback: {''.join(traceback.format_tb(e.__traceback__))}")
            return {
                "user_id": user_id,
                "success": False,
                "job_id": job_id,
                "pdb_content": None
            }

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel

        class PredictionRequest(BaseModel):
            user_id: str
            job_id: str
            sequence: str

        class PredictionResponse(BaseModel):
            user_id: str
            success: bool
            job_id: str
            pdb_content: str = None

        api_app = FastAPI()
        
        api_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api_app.get("/status")
        async def status():
            return {"status": "online"}
            
        @api_app.post("/predict", response_model=PredictionResponse)
        async def run_prediction(request: PredictionRequest):
            try:
                result = self.predict.remote(
                    user_id=request.user_id,
                    job_id=request.job_id,
                    sequence=request.sequence
                )
                return result
            except Exception as e:
                logger.error(f"Error processing prediction: {e}")
                import traceback
                logger.error(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
                raise HTTPException(status_code=500, detail=str(e))
                
        return api_app