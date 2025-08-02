# src/ingest/parser.py
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db.models import RawFile
from db.models import Session as SessionModel
from db.models import SessionLocal, Trackpoint


def parse_tcx_and_store(input_path: str):
    input_path = Path(input_path)
    db: Session = SessionLocal()
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to parse XML from {input_path}: {e}")
        raise

    # Vérifie si raw_file existe déjà
    try:
        existing_raw = (
            db.execute(select(RawFile).where(RawFile.filename == input_path.name))
            .scalars()
            .first()
        )
    except SQLAlchemyError as e:
        print(f"[ERROR] DB lookup RawFile failed: {e}")
        db.rollback()
        db.close()
        raise

    if existing_raw:
        print(
            f"[INFO] RawFile '{input_path.name}' already exists (id={existing_raw.id})"
        )
        # voir si une session liée existe
        existing_session = None
        try:
            # prend la première session associée
            if existing_raw.sessions:
                existing_session = existing_raw.sessions[0]
        except Exception:
            pass

        if existing_session:
            print(
                f"[INFO] Found existing session (id={existing_session.id}) for file {input_path.name}"
            )
            print(f"ALREADY_PROCESSED {existing_session.id}")
            db.close()
            return existing_session.id
        # sinon on continue et créer nouvelle session

    # Création du RawFile
    try:
        new_raw = RawFile(filename=input_path.name)
        db.add(new_raw)
        db.commit()
        db.refresh(new_raw)
        print(f"[INFO] Created RawFile '{input_path.name}' (id={new_raw.id})")
    except Exception as e:
        print(f"[ERROR] Failed to insert RawFile: {e}")
        db.rollback()
        db.close()
        raise

    # Exemple d'extraction d'altitudes (non critique)
    altitudes = []
    for tp in root.findall(".//Trackpoint"):
        ele = tp.find("AltitudeMeters")
        if ele is not None and ele.text:
            try:
                altitudes.append(float(ele.text))
            except ValueError:
                continue
    if altitudes:
        avg_altitude = sum(altitudes) / len(altitudes)
        print(f"[INFO] Average altitude: {avg_altitude:.2f} meters")

    # Création de la session associée
    try:
        session_obj = SessionModel(raw_file_id=new_raw.id)
        db.add(session_obj)
        db.commit()
        db.refresh(session_obj)
        print(
            f"[INFO] Created new session (id={session_obj.id}) for file {input_path.name}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to create Session: {e}")
        db.rollback()
        db.close()
        raise

    # Ingestion des trackpoints
    try:
        for tp in root.findall(".//Trackpoint"):
            time_el = tp.find("Time")
            power_el = tp.find(".//Power")
            cadence_el = tp.find(".//Cadence")
            distance_el = tp.find(".//DistanceMeters")
            altitude_el = tp.find(".//AltitudeMeters")

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

    print(session_obj.id)
    return session_obj.id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCX and ingest into DB")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    args = parser.parse_args()
    parse_tcx_and_store(args.input)
