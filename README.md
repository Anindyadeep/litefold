## API Documentation
I'll add the successful jobs endpoint to the README.

# ESMFold-Lite API Documentation

## Endpoints

### 1. Health Check
```http
GET /health
```
Returns the current status of the system, including GPU and model information.

### 2. Submit Prediction
```http
POST /predict
```
Submit a protein sequence for structure prediction.

**Request Body:**
```json
{
    "job_id": "string",
    "job_name": "string",
    "model": "string",
    "sequence": "string"
}
```

**Response:**
```json
{
    "job_id": "string",
    "status": "string",
    "message": "string"
}
```

### 3. Check Job Status
```http
GET /status/{job_id}
```
Get the status and results of a prediction job.

**Response:**
```json
{
    "job_id": "string",
    "job_name": "string",
    "status": "string",
    "created_at": "datetime",
    "completed_at": "datetime",
    "error_message": "string",
    "pdb_content": "string",         // Only present if job is successful
    "distogram": "array",            // Only present if job is successful
    "plddt_score": "float"          // Only present if job is successful
}
```

### 4. List Successful Jobs
```http
GET /successful-jobs/{user_id}
```
Retrieve a list of all successfully completed jobs for a specific user.

**Parameters:**
- `user_id`: The ID of the user whose successful jobs you want to retrieve

**Response:**
```json
[
    {
        "job_id": "string",
        "job_name": "string",
        "created_at": "datetime",
        "completed_at": "datetime",
        "result_path": "string",
        "user_id": "string"
    }
]
```

## Status Values
The job status can be one of the following:
- `pending`: Job is queued
- `processing`: Job is currently being processed
- `successful`: Job completed successfully
- `crashed`: Job failed with an error

## Notes
- The API uses a background worker to process predictions asynchronously
- All successful predictions include the PDB structure, distance matrix (distogram), and pLDDT confidence scores
- The service requires GPU availability for predictions
- All timestamps are in UTC
- Job IDs must be unique - attempting to reuse a job ID will result in an error
