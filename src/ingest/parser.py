# src/ingest/parser.py
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from db.models import RawFile
from db.models import Session as SessionModel
from db.models import SessionLocal, Trackpoint, init_db

# assure que les tables existent si jamais
init_db()


def parse_tcx_and_store(input_path: str):
    input_path = Path(input_path)
    db: OrmSession = SessionLocal()
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to parse XML from {input_path}: {e}")
        raise

    # 1. RawFile existence check
    try:
        existing_raw = db.execute(
            select(RawFile).where(RawFile.filename == input_path.name)
        ).scalar_one_or_none()

        if existing_raw:
            print(
                f"[INFO] RawFile '{input_path.name}' already exists (id={existing_raw.id})"
            )
            # check existing session(s)
            existing_session = (
                db.execute(
                    select(SessionModel).where(
                        SessionModel.raw_file_id == existing_raw.id
                    )
                )
                .scalars()
                .first()
            )
            if existing_session:
                print(
                    f"[INFO] Found existing session (id={existing_session.id}) for file {input_path.name}"
                )
                print(f"ALREADY_PROCESSED {existing_session.id}")
                print(f"SESSION_ID:{existing_session.id}")
                return str(existing_session.id)
            raw_file = existing_raw
        else:
            # créer nouveau rawfile
            raw_file = RawFile(
                filename=input_path.name,
            )
            db.add(raw_file)
            db.commit()
            db.refresh(raw_file)
            print(f"[INFO] Created RawFile '{input_path.name}' (id={raw_file.id})")
    except Exception as e:
        print(f"[ERROR] Failed to upsert RawFile: {e}")
        db.rollback()
        raise

    # 2. Créer session associée
    try:
        session_obj = SessionModel(
            raw_file_id=raw_file.id,
            start_time=None,  # à compléter si tu parses ça depuis le TCX
            end_time=None,
            duration_s=None,
            distance_km=None,
            elevation_gain_m=None,
            avg_heart_rate=None,
            avg_speed_kmh=None,
        )
        db.add(session_obj)
        db.commit()
        db.refresh(session_obj)
        print(
            f"[INFO] Created new session (id={session_obj.id}) for file {input_path.name}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to create Session: {e}")
        db.rollback()
        raise

    # 3. Parse trackpoints
    trackpoint_count = 0
    try:
        # utile pour debug : combien de Trackpoint dans le XML
        all_tps = root.findall(".//Trackpoint")
        print(f"[DEBUG] Found {len(all_tps)} Trackpoint elements in {input_path.name}")
        for tp in all_tps:
            time_el = tp.find("Time")
            power_el = tp.find(".//Power")
            cadence_el = tp.find(".//Cadence")
            distance_el = tp.find(".//DistanceMeters")
            altitude_el = tp.find(".//AltitudeMeters")
            hr_el = tp.find(".//HeartRateBpm/Value")
            if time_el is None or time_el.text is None:
                continue
            try:
                timestamp = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
            except Exception:
                continue

            def to_int(el):
                try:
                    return (
                        int(el.text) if el is not None and el.text is not None else None
                    )
                except Exception:
                    return None

            def to_float(el):
                try:
                    return (
                        float(el.text)
                        if el is not None and el.text is not None
                        else None
                    )
                except Exception:
                    return None

            trackpoint = Trackpoint(
                session_id=session_obj.id,
                time=timestamp,
                power=to_float(power_el),
                cadence=to_int(cadence_el),
                distance_m=to_float(distance_el),
                altitude_m=to_float(altitude_el),
                heart_rate=to_int(hr_el),
                # les champs dérivés (power_filtered, speed_calc_kmh, etc.) sont calculés plus tard
            )
            db.add(trackpoint)
            trackpoint_count += 1

        db.commit()
        print(f"[INFO] Ingested trackpoints for session {session_obj.id}")
    except Exception as e:
        print(f"[ERROR] Failed to ingest trackpoints: {e}")
        db.rollback()
        raise
    finally:
        db.close()

    print(f"SESSION_ID:{session_obj.id}")
    return str(session_obj.id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCX and ingest into DB")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    args = parser.parse_args()
    parse_tcx_and_store(args.input)
