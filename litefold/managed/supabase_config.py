import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Optional

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME")

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL and key must be set in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def init_storage() -> bool:
    try:
        client = get_supabase_client()
        try:
            client.storage.get_bucket(BUCKET_NAME)
            logger.info(f"Bucket '{BUCKET_NAME}' already exists")
            return True
        except Exception as e:
            if "Not Found" not in str(e):
                logger.error(f"Error checking bucket: {e}")
                return False
            
            try:
                client.storage.create_bucket(
                    BUCKET_NAME,
                    options={
                        "public": True,
                        "allowed_mime_types": ["chemical/x-pdb", "text/plain"],  # Allow PDB files and plain text
                        "file_size_limit": 5242880,  # 5MB limit
                    }
                )
                logger.info(f"Successfully created bucket '{BUCKET_NAME}'")
                client.rpc(
                    'enable_storage_rls',
                    {'bucket_name': BUCKET_NAME}
                ).execute()
                
                # Create storage policy for authenticated users
                client.rpc(
                    'create_storage_policy',
                    {
                        'bucket_name': BUCKET_NAME,
                        'policy_name': 'authenticated_access',
                        'definition': """(
                            role = 'authenticated' AND 
                            (bucket_id = '${bucket_name}' AND 
                             (storage.foldername(name))[1] = auth.uid()::text)
                        )"""
                    }
                ).execute()
                
                logger.info("Successfully configured storage policies")
                return True
                
            except Exception as create_err:
                logger.error(f"Error creating bucket: {create_err}")
                return False
                
    except Exception as e:
        logger.error(f"Error initializing Supabase storage: {e}")
        return False

def get_file_url(user_id: str, job_id: str) -> Optional[str]:
    try:
        client = get_supabase_client()
        file_path = f"{user_id}/{job_id}.pdb"
        return client.storage.from_(BUCKET_NAME).get_public_url(file_path)
    except Exception as e:
        logger.error(f"Error getting file URL: {e}")
        return None

def upload_file(user_id: str, job_id: str, content: bytes) -> Optional[str]:
    try:
        client = get_supabase_client()
        file_path = f"{user_id}/{job_id}.pdb"
        
        client.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=content,
            file_options={"content-type": "chemical/x-pdb"}
        )
        
        return get_file_url(user_id, job_id)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return None

def download_file(user_id: str, job_id: str) -> Optional[bytes]:
    try:
        client = get_supabase_client()
        file_path = f"{user_id}/{job_id}.pdb"
        return client.storage.from_(BUCKET_NAME).download(file_path)
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return None

def delete_file(user_id: str, job_id: str) -> bool:
    try:
        client = get_supabase_client()
        file_path = f"{user_id}/{job_id}.pdb"
        client.storage.from_(BUCKET_NAME).remove([file_path])
        logger.info(f"Successfully deleted file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return False

if __name__ == "__main__":
    if init_storage():
        logger.info("Successfully initialized Supabase storage")
    else:
        logger.error("Failed to initialize Supabase storage")