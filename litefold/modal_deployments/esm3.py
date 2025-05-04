import modal
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)
MODEL_DIR = "/models"
RESULTS_DIR = "/results"
DEST_PATH = Path(MODEL_DIR) / "litefold_3B"
MODEL_NAME = "esmfold_3B_v1"
MINUTES = 60

app = modal.App("litefold-serverless")
model_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("clang")
    .pip_install(
        "torch",
        "biotite",
        "biopython",
        "fastapi[standard]",
        "sqlalchemy",
        "omegaconf",
        "esm"
    )
)

model_volume = modal.Volume.from_name("model", create_if_missing=True)
results_volume = modal.Volume.from_name("results", create_if_missing=True)


@app.cls(
    image=model_image,
    gpu="A10G",
    volumes={MODEL_DIR: model_volume, RESULTS_DIR: results_volume},
    timeout= 10 * MINUTES,
    container_idle_timeout=600,
    secrets=[modal.Secret.from_name("esm3-hf")]
)
class ESMFoldServer:
    @modal.enter()
    def load_model(self):
        logger.info("Starting model loading process ...")
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESM3InferenceClient

        self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")
        logger.info("Model loaded successfully")

    
    @modal.method()
    def predict(self, user_id: str, job_id: str, sequence: str) -> Dict[str, Any]:
        try:
            from esm.sdk.api import ESMProtein, GenerationConfig
            protein = ESMProtein(sequence=sequence)
            protein = self.model.generate(
                protein, GenerationConfig(
                    track="structure", num_steps=10, temperature=0.1
                )
            )
            protein.to_pdb("./generation.pdb")

            # Read the pdb file
            with open("./generation.pdb", "r") as f:
                pdb_content = f.read()

            return {
                "user_id": user_id,
                "success": True,
                "job_id": job_id,
                "pdb_content": pdb_content
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