import os
import shutil
import logging
import subprocess

def setup_console_logger(name, level=logging.INFO):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)

    return logger

logger = setup_console_logger("alphafold2-utils")

def check_nvcc():
    try:
        version_output = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True)
        logger.info(f"nvcc version info: {version_output.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Failed to run 'nvcc --version'")
        return False

def setup_alphafold2_runnable_env(
    cuda_path: str, bin_path: str, lib_path: str, colabfold_dir: str,
    model_dir: str
):
    # Setup some dirs here
    colabfold_bin = f"{colabfold_dir}/colabfold-conda/bin"
    model_colabfold_dir = os.path.join(model_dir, "colabfold")
    colabfold_data_dir = os.path.join(colabfold_dir, "colabfold")

    if bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{bin_path}:{os.environ.get('PATH', '')}"
        logger.info(f"Added {bin_path} to PATH")
    
    if lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        logger.info(f"Added {lib_path} to LD_LIBRARY_PATH")
    
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = cuda_path
        logger.info(f"Set CUDA_HOME to {cuda_path}")
    
    if colabfold_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{colabfold_bin}:{os.environ.get('PATH', '')}"
        logger.info(f"Added {colabfold_bin} to PATH")
    
    if not os.path.exists(model_colabfold_dir):
        os.makedirs(model_colabfold_dir, exist_ok=True)
    
    if os.path.islink(colabfold_data_dir):
        os.unlink(colabfold_data_dir)
    elif os.path.exists(colabfold_data_dir):
        shutil.rmtree(colabfold_data_dir)

    os.symlink(model_colabfold_dir, colabfold_data_dir)
    logger.info(f"Linked {model_colabfold_dir} to {colabfold_data_dir}")

    if not os.listdir(model_colabfold_dir):
        logger.info("Downloading AlphaFold weights...")
        try:
            subprocess.run(
                [f"{colabfold_dir}/colabfold-conda/bin/python3", "-m", "colabfold.download"],
                    check=True
            )
            logger.info("Download of alphafold2 weights finished.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download weights: {e}")
    check_nvcc()

    
def run_alphafold2_prediction_job(
    fasta_content: str,
    job_id: str,
    use_templates: bool,
    use_amber: bool,
    results_dir: str
):
    try:
        if not fasta_content:
            return {
                "status": "error",
                "message": "No FASTA content provided"
            }

        input_dir = os.path.join(results_dir, f"{job_id}_input")
        output_dir = os.path.join(results_dir, f"{job_id}_output")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        fasta_path = os.path.join(input_dir, f"{job_id}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(fasta_content)
        
        cmd = ["colabfold_batch"]
        if use_templates: cmd.append("--templates")
        if use_amber: cmd.append("--amber")
        cmd.append("--use-gpu")
        cmd.append(fasta_path)
        cmd.append(output_dir)

        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Code to stream the logs for debugging
        stdout_lines = []
        stderr_lines = []

        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if stdout_line:
                stdout_lines.append(stdout_line.strip())
                print(f"STDOUT: {stdout_line.strip()}")
                
            if stderr_line:
                stderr_lines.append(stderr_line.strip())
                print(f"STDERR: {stderr_line.strip()}")
                
            if not stdout_line and not stderr_line and process.poll() is not None:
                break
        
        return_code = process.poll()
        if return_code == 0:
            output_files = []
            if os.path.exists(output_dir):
                output_files = os.listdir(output_dir)
            
            best_pdb_content = None
            best_pdb_file = None

            for file in output_files:
                if "relaxed_rank_001_" in file and file.endswith(".pdb"):
                    best_pdb_file = os.path.join(output_dir, file)
                    break
                elif file.endswith(".pdb"):
                    best_pdb_file = os.path.join(output_dir, file)
                    break
            
            if best_pdb_file and os.path.exists(best_pdb_file):
                with open(best_pdb_file, "r") as f:
                    best_pdb_content = f.read()
            
            return {
                "status": "success",
                "job_id": job_id,
                "pdb_content": best_pdb_content,
                "message": "Colabfold prediction successful"
            }
        else:
            return {
                "status": "error",
                "job_id": job_id,
                "pdb_content": None,
                "message": "ColabFold prediction failed"
            }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "job_id": job_id,
            "pdb_content": None,
            "message": f"{str(e)}\nTraceback: {str(traceback.format_exc())}",
        }