import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from db.models import RawFile
from db.models import Session as SessionModel
from db.models import SessionLocal, Trackpoint


def to_float(text):
    try:
        return float(text)
    except Exception:
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
        existing_raw = (
            db.execute(select(RawFile).where(RawFile.filename == input_path.name))
            .scalars()
            .first()
        )
    except Exception as e:
        print(f"[ERROR] DB lookup for RawFile failed: {e}")
        db.close()
        raise

    if existing_raw:
        print(
            f"[INFO] RawFile '{input_path.name}' already exists (id={existing_raw.id})"
        )
        # Voir si une session déjà plantée / créée pour ce rawfile
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

    # Extraction des trackpoints
    trackpoint_elements = root.findall(".//Trackpoint")
    print(
        f"[DEBUG] Found {len(trackpoint_elements)} Trackpoint elements in {input_path.name}"
    )

    if not trackpoint_elements:
        print("[WARN] No trackpoints found, aborting session creation.")
        db.close()
        return None

    # Pour déterminer start_time / end_time et distance et elevation gain
    times = []
    distances = []
    altitudes = []
    heart_rates = []
    cadences = []
    powers = []

    # Créer la session (temporaire) — on remplira plus tard
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

    # Parcours trackpoints
    try:
        for tp in trackpoint_elements:
            time_el = tp.find("Time")
            if time_el is None or time_el.text is None:
                continue
            try:
                timestamp = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
            except Exception:
                continue

            dist_el = tp.find("DistanceMeters")
            altitude_el = tp.find("AltitudeMeters")
            hr_el = tp.find(".//HeartRateBpm/Value")
            cadence_el = tp.find("Cadence")
            # Power souvent dans Extensions selon appareil; on essaie simple
            power_el = tp.find(".//Power")

            distance_m = (
                to_float(dist_el.text) if dist_el is not None and dist_el.text else None
            )
            altitude_m = (
                to_float(altitude_el.text)
                if altitude_el is not None and altitude_el.text
                else None
            )
            heart_rate = (
                int(hr_el.text)
                if hr_el is not None and hr_el.text and hr_el.text.isdigit()
                else None
            )
            cadence = (
                int(cadence_el.text)
                if cadence_el is not None
                and cadence_el.text
                and cadence_el.text.isdigit()
                else None
            )
            power = (
                to_float(power_el.text)
                if power_el is not None and power_el.text
                else None
            )

            # Reconstruction prudente des autres champs (pace/speed) : laissés à l'analyse ultérieure
            trackpoint = Trackpoint(
                session_id=session_id,
                time=timestamp,
                distance_m=distance_m,
                altitude_m=altitude_m,
                heart_rate=heart_rate,
                cadence=cadence,
                power=power,
                power_filtered=power,  # initialement identique, les filtres viendront après
                speed_calc_kmh=None,
                pace_min_per_km=None,
                elevation_diff=None,
            )
            db.add(trackpoint)

            # Collectes pour agrégats
            times.append(timestamp)
            if distance_m is not None:
                distances.append(distance_m)
            if altitude_m is not None:
                altitudes.append(altitude_m)
            if heart_rate is not None:
                heart_rates.append(heart_rate)
            if cadence is not None:
                cadences.append(cadence)
            if power is not None:
                powers.append(power)
        db.commit()
    except Exception as e:
        print(f"[ERROR] Failed to ingest trackpoints: {e}")
        db.rollback()
        db.close()
        raise

    # Calculs sommaires de session
    try:
        # start / end
        start_time = min(times) if times else None
        end_time = max(times) if times else None
        duration_s = None
        if start_time and end_time:
            duration_s = (end_time - start_time).total_seconds()

        # distance en km depuis dernier point (si progression)
        distance_km = None
        if distances:
            # suppose que la distance est cumulative : on prend max
            distance_km = max(distances) / 1000.0

        # elevation gain : somme des deltas positifs
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

        # Mise à jour de la session
        sess_obj = (
            db.execute(select(SessionModel).where(SessionModel.id == session_id))
            .scalars()
            .first()
        )
        if not sess_obj:
            print(
                f"[ERROR] Could not reload session {session_id} to update aggregates."
            )
        else:
            sess_obj.start_time = start_time
            sess_obj.end_time = end_time
            sess_obj.duration_s = duration_s
            sess_obj.distance_km = distance_km
            sess_obj.elevation_gain_m = elevation_gain_m
            sess_obj.avg_heart_rate = avg_heart_rate
            sess_obj.avg_speed_kmh = avg_speed_kmh
            # ftp_estimated / normalized_power / tss pourront être calculés ensuite
            db.add(sess_obj)
            db.commit()
    except Exception as e:
        print(f"[ERROR] Failed to update session aggregates: {e}")
        db.rollback()
        # on ne raise pas forcément, on continue

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
