def load_model():
    global model
    if model is None:
        try:
            logger.info("Starting model loading process...")
            model_name = "esmfold_3B_v1"
            
            # Check GPU first
            if not check_gpu():
                raise RuntimeError("GPU check failed")

            model_path = Path(f"{MODEL_DIR}/{model_name}.pt")
            if model_path.exists():
                logger.info(f"Loading model from local path: {model_path}")
                model_data = torch.load(str(model_path), map_location="cpu")
            else:
                url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
                logger.info(f"Downloading model from: {url}")
                model_data = torch.hub.load_state_dict_from_url(
                    url=url,
                    progress=True,
                    map_location="cpu",
                    model_dir=MODEL_DIR
                )

            logger.info("Extracting model configuration and state...")
            cfg = model_data["cfg"]["model"]
            model_state = model_data["model"]
            
            logger.info("Initializing ESMFold model...")
            model = ESMFold(esmfold_config=cfg)
            
            logger.info("Loading model state...")
            model.load_state_dict(model_state, strict=False)
            
            logger.info(f"Moving model to device: {CUDA_DEVICE}")
            model = model.eval().to(CUDA_DEVICE)
            
            # Verify model is on GPU
            if next(model.parameters()).is_cuda:
                logger.info("Model successfully loaded on GPU")
            else:
                raise RuntimeError("Model failed to move to GPU")
            
            return model
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            model = None  # Reset model on failure
            raise 