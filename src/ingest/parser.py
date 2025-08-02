# src/ingest/parser.py
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import sys

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy import select

from db.models import (
    RawFile,
    Session as SessionModel,
    SessionLocal,
    Trackpoint,
)


def to_float(text):
    try:
        return float(text)
    except Exception:
        return None


def find_child(elem, name):
    """Retourne le premier enfant quel que soit le namespace via local-name()."""
    for child in elem:
        if child.tag.endswith("}" + name) or child.tag == name or child.tag.split("}")[-1] == name:
            return child
    return None


def find_descendants(elem, name):
    """Itère sur tous les descendants avec matching local-name."""
    for descendant in elem.iter():
        if descendant.tag.endswith("}" + name) or descendant.tag == name or descendant.tag.split("}")[-1] == name:
            yield descendant


def get_text_by_path(parent, *names):
    """Traverse une mini-arborescence avec local-name matching."""
    cur = parent
    for name in names:
        if cur is None:
            return None
        cur = find_child(cur, name)
    if cur is not None and cur.text:
        return cur.text.strip()
    return None


def parse_tcx_and_store(input_path: str):
    input_path = Path(input_path)
    db: OrmSession = SessionLocal()
    session_id = None
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML from {input_path}: {e}")
        db.close()
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error parsing {input_path}: {e}")
        db.close()
        raise

    # Vérifier si RawFile existe déjà
    try:
        existing_raw = db.execute(
            select(RawFile).where(RawFile.filename == input_path.name)
        ).scalars().first()
    except Exception as e:
        print(f"[ERROR] DB lookup for RawFile failed: {e}")
        db.close()
        raise

    if existing_raw:
        print(
            f"[INFO] RawFile '{input_path.name}' already exists (id={existing_raw.id})"
        )
        existing_session = (
            db.execute(
                select(SessionModel).where(SessionModel.raw_file_id == existing_raw.id)
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
            db.close()
            return existing_session.id
        raw_file = existing_raw
    else:
        raw_file = RawFile(
            filename=input_path.name,
        )
        db.add(raw_file)
        try:
            db.commit()
        except Exception as e:
            print(f"[ERROR] Failed to insert RawFile: {e}")
            db.rollback()
            db.close()
            raise
        print(f"[INFO] Created RawFile '{input_path.name}' (id={raw_file.id})")

    # Extraction des trackpoints (robuste au namespace)
    trackpoint_elements = list(find_descendants(root, "Trackpoint"))
    print(
        f"[DEBUG] Found {len(trackpoint_elements)} Trackpoint elements in {input_path.name}"
    )

    if not trackpoint_elements:
        print("[WARN] No trackpoints found, aborting session creation.")
        db.close()
        return None

    # Préparer les collections pour agrégats
    times = []
    distances = []
    altitudes = []
    heart_rates = []
    cadences = []
    powers = []

    # Créer la session (vide pour l'instant)
    new_session = SessionModel(raw_file_id=raw_file.id)
    db.add(new_session)
    try:
        db.commit()
    except Exception as e:
        print(f"[ERROR] Failed to create Session row: {e}")
        db.rollback()
        db.close()
        raise
    session_id = new_session.id
    print(f"[INFO] Created new session (id={session_id}) for file {input_path.name}")

    # Parcours des trackpoints
    try:
        for tp in trackpoint_elements:
            time_text = get_text_by_path(tp, "Time")
            if not time_text:
                continue
            try:
                timestamp = datetime.fromisoformat(
                    time_text.replace("Z", "+00:00")
                )
            except Exception:
                continue

            distance_m = to_float(get_text_by_path(tp, "DistanceMeters"))
            altitude_m = to_float(get_text_by_path(tp, "AltitudeMeters"))
            # HeartRateBpm/Value
            hr_text = get_text_by_path(tp, "HeartRateBpm")
            if hr_text is None:
                hr_text = get_text_by_path(tp, "Value")  # fallback path if nested differently
            # But often HeartRateBpm has child Value
            hr_val = None
            hr_value_node = find_child(find_child(tp, "HeartRateBpm") or tp, "Value")
            if hr_value_node is not None and hr_value_node.text and hr_value_node.text.isdigit():
                hr_val = int(hr_value_node.text)
            cadence_val = None
            cad_node = find_child(tp, "Cadence")
            if cad_node is not None and cad_node.text and cad_node.text.isdigit():
                cadence_val = int(cad_node.text)
            power_val = to_float(get_text_by_path(tp, "Power"))

            # Création du trackpoint
            trackpoint = Trackpoint(
                session_id=session_id,
                time=timestamp,
                distance_m=distance_m,
                altitude_m=altitude_m,
                heart_rate=hr_val,
                cadence=cadence_val,
                power=power_val,
                power_filtered=power_val,
                speed_calc_kmh=None,
                pace_min_per_km=None,
                elevation_diff=None,
            )
            db.add(trackpoint)

            # Collecte pour agrégats
            times.append(timestamp)
            if distance_m is not None:
                distances.append(distance_m)
            if altitude_m is not None:
                altitudes.append(altitude_m)
            if hr_val is not None:
                heart_rates.append(hr_val)
            if cadence_val is not None:
                cadences.append(cadence_val)
            if power_val is not None:
                powers.append(power_val)
        db.commit()
    except Exception as e:
        print(f"[ERROR] Failed to ingest trackpoints: {e}")
        db.rollback()
        db.close()
        raise

    # Calculs sommaires de la session
    try:
        start_time = min(times) if times else None
        end_time = max(times) if times else None
        duration_s = None
        if start_time and end_time:
            duration_s = (end_time - start_time).total_seconds()

        distance_km = None
        if distances:
            distance_km = max(distances) / 1000.0

        elevation_gain_m = None
        if altitudes:
            gain = 0.0
            prev = None
            for a in altitudes:
                if prev is not None and a > prev:
                    gain += a - prev
                prev = a
            elevation_gain_m = gain

        avg_speed_kmh = None
        if duration_s and distance_km:
            hours = duration_s / 3600.0 if duration_s else None
            if hours and hours > 0:
                avg_speed_kmh = distance_km / hours

        avg_heart_rate = None
        if heart_rates:
            avg_heart_rate = sum(heart_rates) / len(heart_rates)

        # Reload et mise à jour
        sess_obj = db.execute(
            select(SessionModel).where(SessionModel.id == session_id)
        ).scalars().first()
        if not sess_obj:
            print(f"[ERROR] Could not reload session {session_id} to update aggregates.")
        else:
            sess_obj.start_time = start_time
            sess_obj.end_time = end_time
            sess_obj.duration_s = duration_s
            sess_obj.distance_km = distance_km
            sess_obj.elevation_gain_m = elevation_gain_m
            sess_obj.avg_heart_rate = avg_heart_rate
            sess_obj.avg_speed_kmh = avg_speed_kmh
            db.add(sess_obj)
            db.commit()
    except Exception as e:
        print(f"[ERROR] Failed to update session aggregates: {e}")
        db.rollback()

    print(f"[INFO] Ingested trackpoints for session {session_id}")
    print(f"SESSION_ID:{session_id}")
    return session_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCX and ingest into DB")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    args = parser.parse_args()
    try:
        parse_tcx_and_store(args.input)
    except Exception as e:
        print(f"[FATAL] parsing failed: {e}")
        raise
