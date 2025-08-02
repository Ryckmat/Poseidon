# src/db/models.py
import os
import uuid
from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    ForeignKey,
    Integer,
    Numeric,
    String,
    create_engine,
    func,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in environment")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class RawFile(Base):
    __tablename__ = "raw_files"
    __table_args__ = (UniqueConstraint("filename", name="uq_rawfile_filename"),)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    source_url = Column(String, nullable=True)
    file_metadata = Column("metadata", JSON, nullable=True)
    sessions = relationship("Session", back_populates="raw_file")


class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_file_id = Column(UUID(as_uuid=True), ForeignKey("raw_files.id"), nullable=False)
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    duration_s = Column(Numeric)
    distance_km = Column(Numeric)
    elevation_gain_m = Column(Numeric)
    avg_heart_rate = Column(Numeric)
    avg_speed_kmh = Column(Numeric)
    ftp_estimated = Column(Numeric, nullable=True)
    normalized_power = Column(Numeric, nullable=True)
    tss = Column(Numeric, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    raw_file = relationship("RawFile", back_populates="sessions")
    trackpoints = relationship("Trackpoint", back_populates="session")
    stable_segments = relationship("StableSegment", back_populates="session")
    regressions = relationship("Regression", back_populates="session")


class Trackpoint(Base):
    __tablename__ = "trackpoints"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    time = Column(TIMESTAMP(timezone=True))
    distance_m = Column(Numeric)
    altitude_m = Column(Numeric)
    heart_rate = Column(Integer)
    cadence = Column(Integer)
    power = Column(Numeric)
    power_filtered = Column(Numeric)
    speed_calc_kmh = Column(Numeric)
    pace_min_per_km = Column(Numeric)
    elevation_diff = Column(Numeric)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="trackpoints")


class StableSegment(Base):
    __tablename__ = "stable_segments"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    duration_s = Column(Numeric)
    avg_power = Column(Numeric)
    std_power = Column(Numeric)
    avg_cadence = Column(Numeric)
    avg_speed_kmh = Column(Numeric)
    points_count = Column(Integer)
    label = Column(String, nullable=True)

    session = relationship("Session", back_populates="stable_segments")


class Regression(Base):
    __tablename__ = "regressions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    type = Column(String)
    slope = Column(Numeric)
    intercept = Column(Numeric)
    r2 = Column(Numeric)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="regressions")


def init_db():
    Base.metadata.create_all(bind=engine)