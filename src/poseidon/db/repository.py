"""Requêtes réutilisées par la CLI, l'analyse et le dashboard."""

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session as OrmSession

from poseidon.db.models import RawFile, Regression, Session, StableSegment, Trackpoint


@dataclass(frozen=True)
class SessionInfo:
    id: uuid.UUID
    name: str
    start_time: datetime | None
    duration_s: float | None
    distance_km: float | None
    analyzed: bool


def find_session_by_filename(db: OrmSession, filename: str) -> uuid.UUID | None:
    return db.scalar(
        select(Session.id)
        .join(RawFile, Session.raw_file_id == RawFile.id)
        .where(RawFile.filename == filename)
    )


def list_sessions(db: OrmSession, limit: int | None = None) -> list[SessionInfo]:
    query = (
        select(
            Session.id,
            RawFile.filename,
            Session.start_time,
            Session.duration_s,
            Session.distance_km,
            Session.normalized_power.is_not(None) | Session.tss.is_not(None),
        )
        .join(RawFile, Session.raw_file_id == RawFile.id)
        .order_by(Session.start_time.desc())
        .limit(limit)
    )
    return [SessionInfo(*row[:5], analyzed=bool(row[5])) for row in db.execute(query)]


def all_session_ids(db: OrmSession) -> list[uuid.UUID]:
    return list(db.scalars(select(Session.id).order_by(Session.start_time)))


def clear_analysis(db: OrmSession, session_id: uuid.UUID) -> None:
    """Supprime les résultats d'une analyse précédente (segments, régressions)."""
    db.execute(delete(StableSegment).where(StableSegment.session_id == session_id))
    db.execute(delete(Regression).where(Regression.session_id == session_id))


def delete_session(db: OrmSession, session_id: uuid.UUID) -> bool:
    """Supprime une séance, ses points, ses résultats et son fichier source."""
    raw_file_id = db.scalar(select(Session.raw_file_id).where(Session.id == session_id))
    if raw_file_id is None:
        return False
    clear_analysis(db, session_id)
    db.execute(delete(Trackpoint).where(Trackpoint.session_id == session_id))
    db.execute(delete(Session).where(Session.id == session_id))
    remaining = db.scalar(
        select(func.count())
        .select_from(Session)
        .where(Session.raw_file_id == raw_file_id)
    )
    if not remaining:
        db.execute(delete(RawFile).where(RawFile.id == raw_file_id))
    return True
