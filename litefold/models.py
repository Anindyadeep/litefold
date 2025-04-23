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
from constants import SQLALCHEMY_DATABASE_URL

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
    status = Column(String) 
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    result_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    user_id = Column(String, index=True) 

# Create tables if not exists
Base.metadata.create_all(bind=engine) 