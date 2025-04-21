import requests

# Submit a prediction
print("Enter job id:")
job_id = input()
print("Enter user id:")
user_id = input()

prediction_request = {
    "job_name": "test_prediction",
    "job_id": job_id,
    "model": "esmfold",
    "sequence": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "user_id": user_id
}

# Submit the prediction
print("Submitting prediction request...")
url = "https://anindyadeep--predict.modal.run"  # The endpoint is automatically included

try:
    response = requests.post(url, json=prediction_request)  # Removed "/predict" from URL
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction response received!")
        print(f"Response: {result}")
    else:
        print(f"Error submitting prediction: {response.text}")
        print(f"Status code: {response.status_code}")
except Exception as e:
    print(f"An error occurred: {str(e)}")