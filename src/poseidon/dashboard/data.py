"""Accès base du dashboard. Les fonctions fetch_* sont pures (testables),
les load_* ajoutent le cache Streamlit."""

import uuid

import pandas as pd
import streamlit as st
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session as OrmSession

from poseidon.db import Session, Trackpoint, get_session
from poseidon.db.repository import SessionInfo, list_sessions

CACHE_TTL_S = 300

TRACKPOINT_COLUMNS = [
    "time",
    "power",
    "cadence",
    "heart_rate",
    "speed_kmh",
    "distance_m",
    "altitude_m",
]

# Agrégation par tranches de :bucket secondes pour limiter le volume transféré.
_DOWNSAMPLE_SQL = text(
    """
    SELECT
      to_timestamp(floor(extract(epoch from time) / :bucket) * :bucket)
        AT TIME ZONE 'UTC'     AS time,
      AVG(power)               AS power,
      AVG(cadence)             AS cadence,
      AVG(heart_rate)          AS heart_rate,
      AVG(speed_calc_kmh)      AS speed_kmh,
      MAX(distance_m)          AS distance_m,
      AVG(altitude_m)          AS altitude_m
    FROM trackpoints
    WHERE session_id = :sid
    GROUP BY 1
    ORDER BY 1
    """
)


def fetch_trackpoints(
    db: OrmSession, session_id: uuid.UUID, bucket_seconds: int
) -> pd.DataFrame:
    rows = db.execute(
        _DOWNSAMPLE_SQL, {"sid": session_id, "bucket": int(bucket_seconds)}
    ).all()
    df = pd.DataFrame(rows, columns=TRACKPOINT_COLUMNS)
    numeric = df.columns.drop("time")
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def fetch_history(db: OrmSession) -> pd.DataFrame:
    """Indicateurs de toutes les séances, pour la progression."""
    per_session = (
        select(
            Trackpoint.session_id,
            func.avg(Trackpoint.power_filtered).label("avg_power"),
            func.avg(Trackpoint.cadence).label("avg_cadence"),
        )
        .group_by(Trackpoint.session_id)
        .subquery()
    )
    query = select(
        Session.id,
        Session.start_time,
        Session.duration_s,
        Session.distance_km,
        Session.avg_speed_kmh,
        Session.avg_heart_rate,
        Session.ftp_estimated,
        Session.normalized_power,
        Session.tss,
        per_session.c.avg_power,
        per_session.c.avg_cadence,
    ).outerjoin(per_session, per_session.c.session_id == Session.id)
    df = pd.DataFrame(
        db.execute(query).all(), columns=list(query.selected_columns.keys())
    )
    if df.empty:
        return df
    numeric = df.columns.drop(["id", "start_time"])
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    return df.sort_values("start_time").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_sessions(limit: int) -> list[SessionInfo]:
    with get_session() as db:
        return list_sessions(db, limit=limit)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_trackpoints(session_id: uuid.UUID, bucket_seconds: int) -> pd.DataFrame:
    with get_session() as db:
        return fetch_trackpoints(db, session_id, bucket_seconds)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_history() -> pd.DataFrame:
    with get_session() as db:
        return fetch_history(db)


def clear_cache() -> None:
    load_sessions.clear()
    load_trackpoints.clear()
    load_history.clear()
