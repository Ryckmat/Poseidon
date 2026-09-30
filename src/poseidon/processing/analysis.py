"""Analyse d'une séance : dérivés par point, segments stables, régressions et
indicateurs de séance. Relancer l'analyse remplace les résultats précédents."""

import logging
import uuid

import pandas as pd
from sqlalchemy import select, update

from poseidon.config import AnalysisParams
from poseidon.db import Regression, Session, StableSegment, Trackpoint, get_session
from poseidon.db.repository import clear_analysis
from poseidon.processing import metrics

log = logging.getLogger(__name__)

REGRESSIONS = (("power_vs_cadence", "cadence"), ("power_vs_speed", "speed_kmh"))


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def run_analysis(session_id: uuid.UUID, params: AnalysisParams) -> None:
    with get_session() as db:
        session = db.get(Session, session_id)
        if session is None:
            raise LookupError(f"Séance introuvable : {session_id}")
        rows = db.execute(
            select(
                Trackpoint.id,
                Trackpoint.time,
                Trackpoint.power,
                Trackpoint.cadence,
                Trackpoint.heart_rate,
                Trackpoint.distance_m,
                Trackpoint.altitude_m,
            )
            .where(Trackpoint.session_id == session_id)
            .order_by(Trackpoint.time, Trackpoint.id)
        ).all()
        if not rows:
            raise LookupError(f"Aucun point pour la séance {session_id}")

        raw = pd.DataFrame(
            rows,
            columns=[
                "id",
                "time",
                "power",
                "cadence",
                "heart_rate",
                "distance_m",
                "altitude_m",
            ],
        )
        df = metrics.prepare(raw, params)
        summary = metrics.summarize(df, params.reference_ftp_w)
        segments = metrics.detect_stable_segments(df, params)

        clear_analysis(db, session_id)
        for seg in segments:
            db.add(
                StableSegment(
                    session_id=session_id,
                    start_time=seg["start_time"].to_pydatetime(),
                    end_time=seg["end_time"].to_pydatetime(),
                    duration_s=seg["duration_s"],
                    avg_power=_num(seg["avg_power"]),
                    std_power=_num(seg["std_power"]),
                    avg_cadence=_num(seg["avg_cadence"]),
                    avg_speed_kmh=_num(seg["avg_speed_kmh"]),
                    points_count=seg["points"],
                    label="stable_power",
                )
            )
        for label, target in REGRESSIONS:
            fit = metrics.linear_fit(df["power_filtered"], df[target])
            if fit:
                db.add(
                    Regression(
                        session_id=session_id,
                        type=label,
                        slope=fit.slope,
                        intercept=fit.intercept,
                        r2=_num(fit.r2),
                    )
                )

        db.execute(
            update(Trackpoint),
            [
                {
                    "id": int(row.id),
                    "speed_calc_kmh": _num(row.speed_kmh),
                    "pace_min_per_km": _num(row.pace_min_per_km),
                    "elevation_diff": _num(row.elevation_diff),
                    "power_filtered": _num(row.power_filtered),
                }
                for row in df.itertuples()
            ],
        )

        session.elevation_gain_m = summary.elevation_gain_m
        session.avg_heart_rate = summary.avg_heart_rate
        session.avg_speed_kmh = summary.avg_speed_kmh
        session.ftp_estimated = summary.ftp
        session.normalized_power = summary.normalized_power
        session.tss = summary.tss
        log.info(
            "Séance %s analysée : %d segments stables, NP %s W, TSS %s",
            session_id,
            len(segments),
            f"{summary.normalized_power:.0f}" if summary.normalized_power else "-",
            f"{summary.tss:.0f}" if summary.tss else "-",
        )
