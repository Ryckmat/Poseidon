# src/processing/analysis.py
import argparse
import os
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import select
from db.models import Regression, Session, SessionLocal, StableSegment, Trackpoint
import datetime

load_dotenv()

def run_analysis(session_id):
    db = SessionLocal()
    try:
        sess = db.get(Session, session_id)
        if not sess:
            print("Session not found:", session_id)
            return

        # Load trackpoints into ordered list
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
            print("No trackpoints for session")
            return

        # Build arrays
        # Correction: remove tzinfo from datetime before converting to np.datetime64
        times = np.array([
            tp.time.replace(tzinfo=None) if tp.time.tzinfo else tp.time for tp in tps
        ], dtype="datetime64[ns]")

        distances = np.array([float(tp.distance_m or 0) for tp in tps])
        altitudes = np.array([float(tp.altitude_m or 0) for tp in tps])
        power = np.array(
            [float(tp.power) if tp.power is not None else np.nan for tp in tps]
        )
        cadence = np.array(
            [float(tp.cadence) if tp.cadence is not None else np.nan for tp in tps]
        )

        # Compute time diffs in seconds
        time_diff_s = np.diff(times).astype("timedelta64[s]").astype(float)
        time_diff_s = np.insert(time_diff_s, 0, 0.0)

        # Speed m/s and km/h
        delta_dist = np.diff(distances)
        delta_dist = np.insert(delta_dist, 0, 0.0)
        speed_m_s = np.divide(
            delta_dist,
            time_diff_s,
            out=np.zeros_like(delta_dist),
            where=time_diff_s > 0,
        )
        speed_kmh = speed_m_s * 3.6

        # Correction division by zero warning (ensure speed_m_s > 0 and not nan)
        pace = np.where((speed_m_s > 1e-6) & (~np.isnan(speed_m_s)),
                        (1 / speed_m_s) / 60 * 1000, np.nan)

        # Elevation diff
        elev_diff = np.diff(altitudes)
        elev_diff = np.insert(elev_diff, 0, 0.0)

        # Filter power > threshold
        max_power = float(os.getenv("MAX_POWER", "250"))
        power_filtered = np.where(
            (~np.isnan(power)) & (power <= max_power), power, np.nan
        )

        # Rolling std over approx 30s window (assume roughly uniform sampling)
        # estimate window size in points
        median_dt = (
            np.median(time_diff_s[time_diff_s > 0]) if np.any(time_diff_s > 0) else 1
        )
        window_pts = max(1, int(round(30 / median_dt)))

        def rolling_std(arr, w):
            out = np.full_like(arr, np.nan)
            for i in range(len(arr)):
                start = max(0, i - w + 1)
                window = arr[start : i + 1]
                if np.isnan(window).all():
                    out[i] = np.nan
                else:
                    out[i] = np.nanstd(window)
            return out

        power_std = rolling_std(power_filtered, window_pts)

        # Detect stable segments: power_filtered >=50, std <= threshold, duration >= min
        std_thresh = float(os.getenv("STABLE_STD_THRESHOLD", "5"))
        min_dur = float(os.getenv("MIN_STABLE_DURATION_S", "60"))

        stable_mask = (
            (~np.isnan(power_filtered))
            & (power_filtered >= 50)
            & (power_std <= std_thresh)
        )
        segments = []
        current = None
        for idx, flag in enumerate(stable_mask):
            if flag:
                if current is None:
                    current = {"start": idx, "end": idx}
                else:
                    current["end"] = idx
            else:
                if current:
                    segments.append(current)
                    current = None
        if current:
            segments.append(current)

        # Filter and store segments
        for seg in segments:
            start_idx = seg["start"]
            end_idx = seg["end"]
            start_time = tps[start_idx].time
            end_time = tps[end_idx].time
            duration = (end_time - start_time).total_seconds()
            if duration < min_dur:
                continue
            seg_power = power_filtered[start_idx : end_idx + 1]
            seg_cadence = cadence[start_idx : end_idx + 1]
            seg_speed = speed_kmh[start_idx : end_idx + 1]
            avg_power = np.nanmean(seg_power)
            std_power = np.nanstd(seg_power)
            avg_cad = np.nanmean(seg_cadence)
            avg_spd = np.nanmean(seg_speed)
            pts = end_idx - start_idx + 1

            stable = StableSegment(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                duration_s=duration,
                avg_power=avg_power,
                std_power=std_power,
                avg_cadence=avg_cad,
                avg_speed_kmh=avg_spd,
                points_count=pts,
                label="stable_power",
            )
            db.add(stable)

        # Regression: power vs cadence
        def compute_reg(x, y, label):
            # Correction: Cast x and y to float ndarray, ignore errors
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if np.sum(mask) < 2:
                return
            xs = x[mask]
            ys = y[mask]
            coeffs = np.polyfit(xs, ys, 1)
            pred = np.polyval(coeffs, xs)
            ss_res = np.sum((ys - pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            reg = Regression(
                session_id=session_id,
                type=label,
                slope=float(coeffs[0]),
                intercept=float(coeffs[1]),
                r2=float(r2),
            )
            db.add(reg)

        compute_reg(power_filtered, cadence, "power_vs_cadence")
        compute_reg(power_filtered, speed_kmh, "power_vs_speed")

        # Update each trackpoint with derived fields
        for i, tp in enumerate(tps):
            tp.speed_calc_kmh = (
                float(speed_kmh[i]) if not np.isnan(speed_kmh[i]) else None
            )
            tp.pace_min_per_km = float(pace[i]) if not np.isnan(pace[i]) else None
            tp.elevation_diff = (
                float(elev_diff[i]) if not np.isnan(elev_diff[i]) else None
            )
            tp.power_filtered = (
                float(power_filtered[i]) if not np.isnan(power_filtered[i]) else None
            )

        # Optionally you could update session summary (avg speed, avg hr)
        # commit all
        db.commit()
        print(f"Analysis complete for session {session_id}")
    except Exception as e:
        db.rollback()
        print("Error in analysis:", e)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on a session")
    parser.add_argument(
        "--session-id", required=True, help="UUID of session to analyze"
    )
    args = parser.parse_args()
    run_analysis(args.session_id)
