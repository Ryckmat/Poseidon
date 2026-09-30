"""Accès base pour le dashboard (mis en cache par Streamlit)."""

import pandas as pd
import streamlit as st
from sqlalchemy import select, text

from poseidon.db import Session, get_session

_DOWNSAMPLE_SQL = text(
    """
    SELECT
      to_timestamp(floor(extract(epoch from time) / :bucket) * :bucket)
        AT TIME ZONE 'UTC'     AS time,
      AVG(power)               AS power,
      AVG(cadence)             AS cadence,
      AVG(speed_calc_kmh)      AS speed_kmh,
      MAX(distance_m)          AS distance_m,
      AVG(altitude_m)          AS altitude_m
    FROM trackpoints
    WHERE session_id = :sid
    GROUP BY 1
    ORDER BY 1
    """
)


@st.cache_data(ttl=3600)
def load_sessions(limit: int = 30) -> list:
    """Dernières séances, sous forme de dicts (sérialisables par le cache)."""
    with get_session() as db:
        rows = db.execute(
            select(Session.id, Session.start_time)
            .order_by(Session.start_time.desc())
            .limit(limit)
        ).all()
    return [{"id": r.id, "start_time": r.start_time} for r in rows]


@st.cache_data(ttl=3600)
def load_trackpoints(session_id, bucket_seconds: int = 5) -> pd.DataFrame:
    """Points agrégés côté SQL par tranches de `bucket_seconds`."""
    with get_session() as db:
        rows = db.execute(
            _DOWNSAMPLE_SQL, {"sid": session_id, "bucket": int(bucket_seconds)}
        ).all()
    df = pd.DataFrame(
        rows,
        columns=["time", "power", "cadence", "speed_kmh", "distance_m", "altitude_m"],
    )
    numeric = df.columns.drop("time")
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df
