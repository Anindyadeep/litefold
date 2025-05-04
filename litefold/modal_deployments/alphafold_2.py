import modal
from modal_deployments.utils import (
    setup_alphafold2_runnable_env,
    run_alphafold2_prediction_job,
    setup_console_logger
)

MODEL_DIR = "/mnt/models"
RESULTS_DIR = "/mnt/results"
COLABFOLD_DIR = "/opt/localcolabfold"
CUDA_VERSION = "12.4.0"
CUDA_FLAVOUR = "devel"
OPERATING_SYSTEM = "ubuntu22.04"
IMAGE_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOUR}-{OPERATING_SYSTEM}"
CUDA_PATH = "/usr/local/cuda"
BIN_PATH = f"{CUDA_PATH}/bin"
LIB_PATH = f"{CUDA_PATH}/lib64"
MINUTES = 60

app = modal.App("alphafold-serverless")
logger = setup_console_logger("ALPHAFOLD-2")

# Running some set of commands
# Thanks to localcolabfold which helps to run colabfold locally: https://github.com/YoshitakaMo/localcolabfold

model_image = (
    modal.Image.from_registry(f"nvidia/cuda:{IMAGE_TAG}", add_python="3.11")
    .apt_install(["wget", "git"])
    .run_commands(f"mkdir -p {COLABFOLD_DIR}")
    # Download and install Miniforge
    .run_commands(
        f"cd {COLABFOLD_DIR}",
        f"wget -q -P {COLABFOLD_DIR} https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh",
        f"bash {COLABFOLD_DIR}/Miniforge3-Linux-x86_64.sh -b -p {COLABFOLD_DIR}/conda",
        f"rm {COLABFOLD_DIR}/Miniforge3-Linux-x86_64.sh"
    )
    # Setup conda environment
    .run_commands(
        "export PATH=/opt/localcolabfold/conda/bin:$PATH && "
        "conda update -y -n base conda && "
        "conda create -y -p /opt/localcolabfold/colabfold-conda -c conda-forge -c bioconda "
        "git python=3.10 openmm==8.2.0 pdbfixer kalign2=2.04 hhsuite=3.3.0 mmseqs2"
    )
    # Install all the essential packages
    .run_commands(
        "/opt/localcolabfold/colabfold-conda/bin/pip install --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'",
        "/opt/localcolabfold/colabfold-conda/bin/pip install 'colabfold[alphafold]'",
        "/opt/localcolabfold/colabfold-conda/bin/pip install --upgrade 'jax[cuda12]==0.5.3'",
        "/opt/localcolabfold/colabfold-conda/bin/pip install --upgrade tensorflow",
        "/opt/localcolabfold/colabfold-conda/bin/pip install silence_tensorflow",
        # Install missing dependency for Modal
        "/opt/localcolabfold/colabfold-conda/bin/pip install grpclib",
        "/opt/localcolabfold/colabfold-conda/bin/pip install aiohttp",
        "/opt/localcolabfold/colabfold-conda/bin/pip install fastapi",
        "/opt/localcolabfold/colabfold-conda/bin/pip install uvicorn",
        "/opt/localcolabfold/colabfold-conda/bin/pip install pydantic",
    )
    # Setup the updater
    .run_commands(
        f"wget -qnc -O {COLABFOLD_DIR}/update_linux.sh "
        f"https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/update_linux.sh",
        f"chmod +x {COLABFOLD_DIR}/update_linux.sh"
    )
    # Modify code for non-GUI environment
    .run_commands(
        "cd /opt/localcolabfold/colabfold-conda/lib/python3.10/site-packages/colabfold && "
        "sed -i -e \"s#from matplotlib import pyplot as plt#import matplotlib\\nmatplotlib.use('Agg')\\nimport matplotlib.pyplot as plt#g\" plot.py && "
        "sed -i -e \"s#appdirs.user_cache_dir(__package__ or \\\"colabfold\\\")#\\\"/opt/localcolabfold/colabfold\\\"#g\" download.py && "
        "sed -i -e \"s#from io import StringIO#from io import StringIO\\nfrom silence_tensorflow import silence_tensorflow\\nsilence_tensorflow()#g\" batch.py && "
        "rm -rf __pycache__"
    )
    # Create colabfold directory (will be linked to the mount later)
    .run_commands(f"mkdir -p {COLABFOLD_DIR}/colabfold")
    # Add to system PATH
    .run_commands(
        "echo 'export PATH=/opt/localcolabfold/colabfold-conda/bin:$PATH' > /etc/profile.d/colabfold.sh"
    )
    # Set environment variables
    .env({"PATH": f"{COLABFOLD_DIR}/colabfold-conda/bin:$PATH"})
)

# Time out is reserved to 2 hrs because AlphaFold inference takes time
# for medium and big size molecule
# Time out is increased to 6 hrs because AlphaFold inference can take significant time
# for medium and big size molecules
@app.cls(
    image=model_image,
    gpu="any",
    volumes={MODEL_DIR: modal.Volume.from_name("model", create_if_missing=True),
             RESULTS_DIR: modal.Volume.from_name("results", create_if_missing=True)},
    timeout=360 * MINUTES,
    scaledown_window=10 * MINUTES,
)
class AlphaFold2Server:
    @modal.enter()
    def load_model(self):
        setup_alphafold2_runnable_env(
            cuda_path=CUDA_PATH,
            bin_path=BIN_PATH,
            lib_path=LIB_PATH,
            colabfold_dir=COLABFOLD_DIR,
            model_dir=MODEL_DIR
        )
    
    @modal.method()
    def predict(
        self,
        job_id: str,
        fasta_content: str | None = None,
        use_templates: bool | None = False,
        use_amber: bool | None = False
    ):
        return run_alphafold2_prediction_job(
            fasta_content=fasta_content,
            job_id=job_id,
            use_templates=use_templates,
            use_amber=use_amber,
            results_dir=RESULTS_DIR
        )
    
    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Define the models
        class PredictionRequest(BaseModel):
            user_id: str
            job_id: str
            sequence: str
            use_templates: bool = True
            use_amber: bool = True

        class PredictionResponse(BaseModel):
            user_id: str
            job_id: str
            success: bool
            pdb_content: str = None

        @app.get("/status")
        async def status():
            return {"status": "online"}
        
        @app.post("/predict")
        async def predict_pdb(request: PredictionRequest) -> PredictionResponse:
            try:
                result = self.predict.remote(
                    job_id=request.job_id,
                    fasta_content=request.sequence,
                    use_templates=request.use_templates,
                    use_amber=request.use_amber
                )
                if result["status"] != "success" or not result.get("pdb_content"):
                    return {
                        "user_id": request.user_id,
                        "success": False,
                        "job_id": request.job_id,
                        "pdb_content": None
                    }
                return {
                    "user_id": request.user_id,
                    "job_id": request.job_id,
                    "success": True,
                    "pdb_content": result["pdb_content"]
                }
            except Exception as e:
                import traceback
                logger.error(f"Error processing prediction: {e}")
                logger.error(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
                return {
                        "user_id": request.user_id,
                        "success": False,
                        "job_id": request.job_id,
                        "pdb_content": None
                    }
        return app