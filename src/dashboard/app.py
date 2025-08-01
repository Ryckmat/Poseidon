# src/dashboard/app.py
import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import select
import matplotlib.pyplot as plt

# Import des modèles (attention : tu utilises l'import sans "src." parce que tu as installé editable)
from db.models import SessionLocal, Session, Trackpoint, Regression, StableSegment

load_dotenv()

st.set_page_config(page_title="Poseidon Session Viewer", layout="wide")


def to_number(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def metric_value(val, fmt="{:.2f}"):
    num = to_number(val)
    if num is None:
        return "—"
    return fmt.format(num)


def get_sessions(db):
    return db.execute(select(Session).order_by(Session.start_time.desc())).scalars().all()


def fetch_trackpoints(db, session_id):
    return db.execute(
        select(Trackpoint).where(Trackpoint.session_id == session_id).order_by(Trackpoint.time)
    ).scalars().all()


def fetch_regressions(db, session_id):
    return db.execute(
        select(Regression).where(Regression.session_id == session_id)
    ).scalars().all()


def fetch_segments(db, session_id):
    return db.execute(
        select(StableSegment).where(StableSegment.session_id == session_id)
    ).scalars().all()


def main():
    db = SessionLocal()
    try:
        sessions = get_sessions(db)
        if not sessions:
            st.warning("No sessions found in the database.")
            return

        # Sidebar: selection
        session_map = {f"{s.start_time} | {s.id}": s for s in sessions}
        selected_label = st.sidebar.selectbox("Select session", list(session_map.keys()))
        session = session_map[selected_label]

        st.title("Poseidon — Session Overview")
        st.subheader(f"Session ID: {session.id}")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration (s)", metric_value(session.duration_s, "{:.0f}"))
        col2.metric("Distance (km)", metric_value(session.distance_km, "{:.2f}"))
        col3.metric("Elevation Gain (m)", metric_value(session.elevation_gain_m, "{:.1f}"))
        col4.metric("Avg Speed (km/h)", metric_value(session.avg_speed_kmh, "{:.2f}"))

        # Optional extra metrics if present
        extra1, extra2, extra3 = st.columns(3)
        extra1.metric("FTP est.", metric_value(session.ftp_estimated, "{:.1f}"))
        extra2.metric("Normalized Power", metric_value(session.normalized_power, "{:.1f}"))
        extra3.metric("TSS", metric_value(session.tss, "{:.1f}"))

        st.markdown("### Regressions")
        regs = fetch_regressions(db, session.id)
        if regs:
            for r in regs:
                st.write(f"- **{r.type}**: slope = {to_number(r.slope):.4f}, intercept = {to_number(r.intercept):.2f}, R² = {to_number(r.r2):.3f}")
        else:
            st.info("No regression data available for this session.")

        tps = fetch_trackpoints(db, session.id)
        if not tps:
            st.warning("No trackpoints for this session.")
            return

        # Build DataFrame safely converting decimals
        df = pd.DataFrame([{
            "time": tp.time,
            "power": to_number(tp.power),
            "power_filtered": to_number(tp.power_filtered),
            "cadence": to_number(tp.cadence),
            "speed_kmh": to_number(tp.speed_calc_kmh),
            "pace_min_per_km": to_number(tp.pace_min_per_km),
            "elevation": to_number(tp.altitude_m)
        } for tp in tps])

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        st.markdown("### Power / Cadence / Speed over time")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["time"], df["power"], label="Power raw", alpha=0.4)
        ax1.plot(df["time"], df["power_filtered"], label="Power filtered", linestyle="--")
        ax1.set_ylabel("Power (W)")
        ax1.set_xlabel("Time")
        ax1.legend()
        st.pyplot(fig1)

        col_a, col_b = st.columns(2)
        with col_a:
            fig2, ax2 = plt.subplots()
            ax2.plot(df["time"], df["cadence"], label="Cadence")
            ax2.set_ylabel("Cadence (rpm)")
            ax2.set_xlabel("Time")
            ax2.legend()
            st.pyplot(fig2)
        with col_b:
            fig3, ax3 = plt.subplots()
            ax3.plot(df["time"], df["speed_kmh"], label="Speed (km/h)")
            ax3.set_ylabel("Speed (km/h)")
            ax3.set_xlabel("Time")
            ax3.legend()
            st.pyplot(fig3)

        st.markdown("### Scatter correlations")
        corr_col1, corr_col2 = st.columns(2)
        with corr_col1:
            st.write("Power vs Cadence")
            fig_pc, ax_pc = plt.subplots()
            ax_pc.scatter(df["power_filtered"], df["cadence"], s=10)
            ax_pc.set_xlabel("Power filtered (W)")
            ax_pc.set_ylabel("Cadence (rpm)")
            st.pyplot(fig_pc)
        with corr_col2:
            st.write("Power vs Speed")
            fig_ps, ax_ps = plt.subplots()
            ax_ps.scatter(df["power_filtered"], df["speed_kmh"], s=10)
            ax_ps.set_xlabel("Power filtered (W)")
            ax_ps.set_ylabel("Speed (km/h)")
            st.pyplot(fig_ps)

        st.markdown("### Stable Segments")
        segs = fetch_segments(db, session.id)
        if segs:
            for s in segs:
                st.write(f"- Segment {s.id}: duration {to_number(s.duration_s):.1f}s, avg_power {to_number(s.avg_power):.1f}W, std {to_number(s.std_power):.1f}, avg_cadence {to_number(s.avg_cadence):.1f}, avg_speed {to_number(s.avg_speed_kmh):.2f} km/h")
        else:
            st.info("No stable segments detected.")

        # Option to download cleaned data
        st.markdown("### Export")
        if st.button("Download cleaned trackpoints CSV"):
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name=f"session_{session.id}_trackpoints.csv", mime="text/csv")
    finally:
        db.close()


if __name__ == "__main__":
    main()
