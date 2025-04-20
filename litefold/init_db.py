from models import Base, engine
import os

def init_db():
    # Remove existing database if it exists
    if os.path.exists("jobs.db"):
        os.remove("jobs.db")
        print("Removed existing database")
    
    # Create new database with all tables
    Base.metadata.create_all(bind=engine)
    print("Created new database with all tables")

if __name__ == "__main__":
    init_db() 