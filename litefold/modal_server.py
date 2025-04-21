import modal
import logging
import requests
from pathlib import Path
import torch
from fold_models import ESMFold
from tqdm.auto import tqdm
import numpy as np
from Bio.PDB import PDBParser
import biotite.structure.io as bsio

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
    concurrency_limit=10
)
class LiteFoldServer:
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
    def predict(self, user_id: str, job_id: str, sequence: str) -> dict:
        try:
            with torch.no_grad():
                output = self.model.infer_pdb(sequence)
                
                # Save output to temp PDB file
                output_path = Path(RESULTS_DIR) / f"{job_id}.pdb"
                with open(output_path, "w") as f:
                    f.write(output)
            
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
                dist_matrix = np.linalg.norm(
                    coords[:, None, :] - coords[None, :, :], axis=-1
                )
                return dist_matrix.tolist()
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", output_path)
            coords = get_ca_coordinates(structure)
            distogram = calculate_distogram(coords)
            struct = bsio.load_structure(output_path, extra_fields=["b_factor"])
            plddt_score = float(struct.b_factor.mean())
            
            # Read PDB content to return to client
            with open(output_path, "r") as f:
                pdb_content = f.read()
            
            return {
                "distogram": distogram,
                "plddt": plddt_score,
                "coords": coords,
                "pdb_content": pdb_content,
                "user_id": user_id,
                "success": True,
                "job_id": job_id
            }
        except Exception as e:
            import traceback
            logger.error(f"Internal Server error: {e}\nTraceback: {''.join(traceback.format_tb(e.__traceback__))}")
            return {
                "distogram": None,
                "plddt": None,
                "coords": None,
                "pdb_content": None,
                "user_id": user_id,
                "success": False,
                "job_id": job_id
            }
