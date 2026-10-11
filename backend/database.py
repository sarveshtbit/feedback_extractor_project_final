import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Uses DATABASE_URL environment variable if set (e.g., postgresql://user:pass@localhost:5432/bannari_db)
# Defaults to SQLite out-of-the-box for instant zero-config local execution.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./paper_form_digitizer.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
