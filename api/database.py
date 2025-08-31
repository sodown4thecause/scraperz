from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database URL - constructed from environment variables set in docker-compose
DATABASE_URL = "postgresql://{user}:{password}@{host}:{port}/{db}".format(
    user=os.getenv("POSTGRES_USER", "user"),
    password=os.getenv("POSTGRES_PASSWORD", "password"),
    host=os.getenv("DB_HOST", "db"), # The service name in docker-compose
    port=os.getenv("DB_PORT", "5432"),
    db=os.getenv("POSTGRES_DB", "scraperz")
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get the DB session in API endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
