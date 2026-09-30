"""Job d'analyse d'une séance : dérivés par point, segments stables, régressions."""

import argparse
import uuid
from typing import Optional

import pandas as pd
from sqlalchemy import select

from poseidon.config import AnalysisParams
from poseidon.db import Regression, Session, StableSegment, Trackpoint, get_session
from poseidon.processing import metrics


def _none_if_nan(x) -> Optional[float]:
    return None if pd.isna(x) else float(x)


def run_analysis(session_id: uuid.UUID, params: AnalysisParams) -> None:
    with get_session() as db:
        if db.get(Session, session_id) is None:
            raise LookupError(f"Séance introuvable: {session_id}")
        tps = (
            db.execute(
                select(Trackpoint)
                .where(Trackpoint.session_id == session_id)
                .order_by(Trackpoint.time)
            )
            .scalars()
            .all()
        )
        if not tps:
            raise LookupError(f"Aucun point pour la séance {session_id}")

        raw = pd.DataFrame(
            {
                "power": [tp.power for tp in tps],
                "cadence": [tp.cadence for tp in tps],
                "distance_m": [tp.distance_m for tp in tps],
                "altitude_m": [tp.altitude_m for tp in tps],
            },
            dtype=float,
        )
        raw.insert(0, "time", [tp.time for tp in tps])
        df = metrics.prepare(raw, params)

        for seg in metrics.detect_stable_segments(df, params):
            db.add(
                StableSegment(
                    session_id=session_id,
                    start_time=seg["start_time"].to_pydatetime(),
                    end_time=seg["end_time"].to_pydatetime(),
                    duration_s=seg["duration_s"],
                    avg_power=_none_if_nan(seg["avg_power"]),
                    std_power=_none_if_nan(seg["std_power"]),
                    avg_cadence=_none_if_nan(seg["avg_cadence"]),
                    avg_speed_kmh=_none_if_nan(seg["avg_speed_kmh"]),
                    points_count=seg["points"],
                    label="stable_power",
                )
            )

        for label, target in (
            ("power_vs_cadence", "cadence"),
            ("power_vs_speed", "speed_kmh"),
        ):
            fit = metrics.linear_fit(df["power_filtered"], df[target])
            if fit:
                db.add(
                    Regression(
                        session_id=session_id,
                        type=label,
                        slope=fit.slope,
                        intercept=fit.intercept,
                        r2=_none_if_nan(fit.r2),
                    )
                )

        # Les points sont triés par temps dans les deux listes.
        for tp, row in zip(tps, df.itertuples()):
            tp.speed_calc_kmh = _none_if_nan(row.speed_kmh)
            tp.pace_min_per_km = _none_if_nan(row.pace_min_per_km)
            tp.elevation_diff = _none_if_nan(row.elevation_diff)
            tp.power_filtered = _none_if_nan(row.power_filtered)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Analyse une séance ingérée")
    parser.add_argument("session_id", type=uuid.UUID, help="UUID de la séance")
    args = parser.parse_args(argv)
    run_analysis(args.session_id, AnalysisParams.from_env())
    print(f"Analyse terminée pour la séance {args.session_id}")


if __name__ == "__main__":
    main()
