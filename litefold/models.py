from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Check if we're running in Modal
IN_MODAL = os.environ.get('MODAL_ENVIRONMENT') == 'true'

if IN_MODAL:
    # Use Modal volume path in deployment
    VOLUME_PATH = "/data"
    DB_PATH = f"{VOLUME_PATH}/jobs.db"
else:
    # Use local path in development
    DB_PATH = "jobs.db"  # This will create the DB in the current directory

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    job_name = Column(String)
    model = Column(String)
    sequence = Column(String)
    status = Column(String)  # pending, successful, crashed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    user_id = Column(String, index=True)  # Add index for faster user-based queries

# Create tables only in development
if not IN_MODAL:
    Base.metadata.create_all(bind=engine)

