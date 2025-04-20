# from fold_models.esmfold.main import ESMFold
# import torch
# from pathlib import Path
# import biotite.structure.io as bsio

# def _load_model(model_name):
#     if model_name.endswith(".pt"):  # local, treat as filepath
#         model_path = Path(model_name)
#         model_data = torch.load(str(model_path), map_location="cpu")
#     else:  # load from hub
#         url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
#         model_data = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")

#     cfg = model_data["cfg"]["model"]
#     model_state = model_data["model"]
#     model = ESMFold(esmfold_config=cfg)

#     expected_keys = set(model.state_dict().keys())
#     found_keys = set(model_state.keys())

#     missing_essential_keys = []
#     for missing_key in expected_keys - found_keys:
#         if not missing_key.startswith("esm."):
#             missing_essential_keys.append(missing_key)

#     if missing_essential_keys:
#         raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

#     model.load_state_dict(model_state, strict=False)

#     return model


# def esmfold_v1():
#     return _load_model("esmfold_3B_v1")

# model = esmfold_v1()
# model = model.eval().to("cuda:5")

# sequence = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"

# # sequence = "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE"

# # sequence = "MKCLLLALALTCGAQALIVTQTMKGLDIQKVAGTWYSLAMAASDISLLDAQSAPLRVYVEELKPTPEGDLEILLQKWENGECAQKKIIAEKTKIPAVFKIDALNENKVLVLDTDYKKYLLFCMENSAEPEQSLACQCLVRTPEVDDEALEKFDKALKALPMHIRLSFNPTQLEEQCHI"


# with torch.no_grad():
#     output = model.infer_pdb(sequence)

# with open("result.pdb", "w") as f:
#     f.write(output)

# struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
# print(struct.b_factor.mean())  # this will be the pLDDT
# # 88.3


import requests
import time

# Submit a prediction
print("Enter job id:")
job_id = input()
print("Enter user id:")
user_id = input()

prediction_request = {
    "job_name": "test_prediction",
    "job_id": job_id,  # Make sure this matches the status check
    "model": "esmfold",
    "sequence": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "user_id": user_id
}

# First submit the prediction
print("Submitting prediction request...")
url1 = "https://f4dd-194-105-248-9.ngrok-free.app"
url2 = "http://localhost:8000"

response = requests.post(url1 + "/predict", json=prediction_request)
if response.status_code != 200:
    print(f"Error submitting prediction: {response.text}")
    exit(1)
print("Prediction submitted successfully")

# Then check status
print("\nChecking prediction status...")
while True:
    try:
        status = requests.get(f"http://localhost:8000/status/{prediction_request['job_id']}")
        if status.status_code != 200:
            print(f"Error checking status: {status.text}")
            time.sleep(1)
            continue
            
        response = status.json()
        if response["status"] == "successful":
            pdb_content = response["pdb_content"]  # PDB file as string
            distogram = response["distogram"]      # 2D distance matrix
            plddt = response["plddt_score"]
            print(f"User ID: {response['user_id']}")  # Print user ID in response
            break
        elif response["status"] == "crashed":
            print(f"Prediction failed: {response.get('error_message', 'Unknown error')}")
            exit(1)
        else:
            print(f"Status: {response['status']}... waiting")
            time.sleep(1)
            
    except Exception as e:
        print(f"Error occurred while checking status: {str(e)}")
        time.sleep(1)

print("\nPrediction completed successfully!")
print(f"PDB content (first 100 chars): {pdb_content[:100]}...")
print(f"Distogram shape: {len(distogram)}x{len(distogram[0])}")
print(f"First few distogram values: {distogram[0][:5]}")
print(f"pLDDT score: {plddt:.2f}")

