import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from backend.database import Base

class FormRecord(Base):
    __tablename__ = "forms"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    file_path = Column(String, nullable=False)
    status = Column(String, default="PENDING")  # AUTO_ACCEPTED, NEEDS_REVIEW, NEEDS_REVIEW_MOBILE_PRIORITY, VERIFIED
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    mobile_verification = relationship("MobileVerification", back_populates="form", uselist=False, cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="form", uselist=False, cascade="all, delete-orphan")
    corrections = relationship("Correction", back_populates="form", cascade="all, delete-orphan")
    processing_logs = relationship("ProcessingLog", back_populates="form", cascade="all, delete-orphan")

class MobileVerification(Base):
    __tablename__ = "mobile_verification"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    form_id = Column(Integer, ForeignKey("forms.id"), nullable=False, unique=True)
    extracted_mobile = Column(String(10), nullable=True)
    verified_mobile = Column(String(10), nullable=True)
    confidence = Column(Float, nullable=False)
    is_valid = Column(Boolean, default=False)
    review_status = Column(String, default="PENDING_REVIEW")  # AUTO_ACCEPT, PENDING_REVIEW, HUMAN_VERIFIED

    form = relationship("FormRecord", back_populates="mobile_verification")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    form_id = Column(Integer, ForeignKey("forms.id"), nullable=False, unique=True)
    omr_data_json = Column(Text, nullable=False)
    omr_status = Column(String, default="CLEAN")  # CLEAN, NEEDS_REVIEW

    form = relationship("FormRecord", back_populates="feedback")

class Correction(Base):
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    form_id = Column(Integer, ForeignKey("forms.id"), nullable=False)
    field_name = Column(String, nullable=False)
    original_value = Column(Text, nullable=True)
    corrected_value = Column(Text, nullable=False)
    corrected_by = Column(String, default="human_operator")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    form = relationship("FormRecord", back_populates="corrections")

class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    form_id = Column(Integer, ForeignKey("forms.id"), nullable=False)
    step = Column(String, nullable=False)  # MARKER_ALIGNMENT, GUIDE_DOT_REMOVAL, OMR_DETECTION, CLASSIFICATION, CONFIDENCE_SCORING
    status = Column(String, nullable=False)  # SUCCESS, WARNING, ERROR
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    form = relationship("FormRecord", back_populates="processing_logs")
