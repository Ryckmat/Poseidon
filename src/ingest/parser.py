import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import sys

from sqlalchemy import select, func
from sqlalchemy.orm import Session as OrmSession

from db.models import RawFile, Session as SessionModel, SessionLocal, Trackpoint


def parse_tcx_and_store(input_path: str, force_new: bool = False):
    input_path = Path(input_path)
    db: OrmSession = SessionLocal()
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to parse XML from {input_path}: {e}")
        raise

    # === RawFile upsert ===
    try:
        existing_raw = db.execute(
            select(RawFile).where(RawFile.filename == input_path.name)
        ).scalar_one_or_none()

        if existing_raw and not force_new:
            raw_file = existing_raw
            print(f"[INFO] RawFile '{input_path.name}' already exists (id={raw_file.id})")
        else:
            raw_file = RawFile(filename=input_path.name)
            db.add(raw_file)
            db.commit()
            db.refresh(raw_file)
            print(f"[INFO] Created RawFile '{input_path.name}' (id={raw_file.id})")
    except Exception as e:
        print(f"[ERROR] Failed to insert or retrieve RawFile: {e}")
        db.rollback()
        raise

    # === Session selection / creation ===
    session_obj = None
    if not force_new:
        existing_sessions = (
            db.execute(
                select(SessionModel)
                .where(SessionModel.raw_file_id == raw_file.id)
                .order_by(SessionModel.start_time.desc())
            )
            .scalars()
            .all()
        )
        if existing_sessions:
            session_obj = existing_sessions[0]
            print(f"[INFO] Found existing session (id={session_obj.id}) for file {input_path.name}")

    if session_obj is None:
        # Extract minimal start/end from trackpoints
        start_time = None
        end_time = None
        times = []
        for tp in root.findall(".//Trackpoint"):
            time_el = tp.find("Time")
            if time_el is not None and time_el.text:
                try:
                    t = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
                    times.append(t)
                except Exception:
                    continue
        if times:
            start_time = min(times)
            end_time = max(times)
        session_obj = SessionModel(
            raw_file_id=raw_file.id,
            start_time=start_time,
            end_time=end_time,
            duration_s=((end_time - start_time).total_seconds() if start_time and end_time else None),
            distance_km=None,
            elevation_gain_m=None,
            avg_heart_rate=None,
            avg_speed_kmh=None,
        )
        db.add(session_obj)
        db.commit()
        db.refresh(session_obj)
        print(f"[INFO] Created new session (id={session_obj.id}) for file {input_path.name}")

    # === Early exit if already ingéré ===
    has_trackpoints = (
        db.execute(
            select(func.count(Trackpoint.id)).where(Trackpoint.session_id == session_obj.id)
        )
        .scalar_one()
    )
    if has_trackpoints and not force_new:
        print(f"ALREADY_PROCESSED {session_obj.id}")
        db.close()
        return str(session_obj.id)

    # === (Re)ingest trackpoints ===
    try:
        if force_new and has_trackpoints:
            db.execute(
                Trackpoint.__table__.delete().where(Trackpoint.session_id == session_obj.id)
            )
            db.commit()

        for tp in root.findall(".//Trackpoint"):
            time_el = tp.find("Time")
            power_el = tp.find(".//Power")
            cadence_el = tp.find(".//Cadence")
            distance_el = tp.find(".//DistanceMeters")
            altitude_el = tp.find("AltitudeMeters")

            if time_el is None or time_el.text is None:
                continue
            try:
                timestamp = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
            except Exception:
                continue

            power = (
                int(power_el.text) if power_el is not None and power_el.text else None
            )
            cadence = (
                int(cadence_el.text)
                if cadence_el is not None and cadence_el.text
                else None
            )
            distance_m = (
                float(distance_el.text)
                if distance_el is not None and distance_el.text
                else None
            )
            altitude_m = (
                float(altitude_el.text)
                if altitude_el is not None and altitude_el.text
                else None
            )

            trackpoint = Trackpoint(
                session_id=session_obj.id,
                time=timestamp,
                power=power,
                cadence=cadence,
                distance_m=distance_m,
                altitude_m=altitude_m,
            )
            db.add(trackpoint)
        db.commit()
        print(f"[INFO] Ingested trackpoints for session {session_obj.id}")
    except Exception as e:
        print(f"[ERROR] Failed to ingest trackpoints: {e}")
        db.rollback()
        raise
    finally:
        db.close()

    print(f"{session_obj.id}")
    return str(session_obj.id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCX and ingest into DB")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    parser.add_argument("--force-new", action="store_true", help="Force new session even if existing")
    args = parser.parse_args()
    sid = parse_tcx_and_store(args.input, force_new=args.force_new)
    # si c'était déjà ingéré, on a déjà imprimé ALREADY_PROCESSED en tête
    # on réimprime pour que le workflow puisse capturer l'ID proprement
    if not sid.startswith("ALREADY_PROCESSED"):
        print(f"{sid}")
