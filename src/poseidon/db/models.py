"""Schéma de la base (SQLAlchemy)."""

import uuid

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    BigInteger,
    Column,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# NUMERIC côté base, float côté Python.
Number = Numeric(asdecimal=False)
# BIGSERIAL sous PostgreSQL ; SQLite n'auto-incrémente que les INTEGER.
BigId = BigInteger().with_variant(Integer, "sqlite")


class RawFile(Base):
    __tablename__ = "raw_files"
    __table_args__ = (UniqueConstraint("filename", name="uq_rawfile_filename"),)

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    source_url = Column(String, nullable=True)
    file_metadata = Column("metadata", JSON, nullable=True)

    sessions = relationship("Session", back_populates="raw_file")


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_start_time", "start_time"),)

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    raw_file_id = Column(Uuid, ForeignKey("raw_files.id"), nullable=False)
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    duration_s = Column(Number)
    distance_km = Column(Number)
    elevation_gain_m = Column(Number)
    avg_heart_rate = Column(Number)
    avg_speed_kmh = Column(Number)
    ftp_estimated = Column(Number, nullable=True)
    normalized_power = Column(Number, nullable=True)
    tss = Column(Number, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    raw_file = relationship("RawFile", back_populates="sessions")
    trackpoints = relationship("Trackpoint", back_populates="session")
    stable_segments = relationship("StableSegment", back_populates="session")
    regressions = relationship("Regression", back_populates="session")


class Trackpoint(Base):
    __tablename__ = "trackpoints"
    __table_args__ = (Index("ix_trackpoints_session_time", "session_id", "time"),)

    id = Column(BigId, primary_key=True, autoincrement=True)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=False)
    time = Column(TIMESTAMP(timezone=True))
    distance_m = Column(Number)
    altitude_m = Column(Number)
    heart_rate = Column(Integer)
    cadence = Column(Integer)
    power = Column(Number)
    power_filtered = Column(Number)
    speed_calc_kmh = Column(Number)
    pace_min_per_km = Column(Number)
    elevation_diff = Column(Number)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="trackpoints")


class StableSegment(Base):
    __tablename__ = "stable_segments"
    __table_args__ = (Index("ix_stable_segments_session", "session_id"),)

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=False)
    start_time = Column(TIMESTAMP(timezone=True))
    end_time = Column(TIMESTAMP(timezone=True))
    duration_s = Column(Number)
    avg_power = Column(Number)
    std_power = Column(Number)
    avg_cadence = Column(Number)
    avg_speed_kmh = Column(Number)
    points_count = Column(Integer)
    label = Column(String, nullable=True)

    session = relationship("Session", back_populates="stable_segments")


class Regression(Base):
    __tablename__ = "regressions"
    __table_args__ = (Index("ix_regressions_session", "session_id"),)

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=False)
    type = Column(String)
    slope = Column(Number)
    intercept = Column(Number)
    r2 = Column(Number)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="regressions")
