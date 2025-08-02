# src/ingest/parser.py
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from db.models import RawFile
from db.models import Session as SessionModel
from db.models import SessionLocal, Trackpoint


def _strip_namespace(tree: ET.ElementTree):
    """
    Enlève tous les namespaces pour permettre des requêtes simples.
    Modifie l'arbre en place.
    """
    for elem in tree.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
        # aussi les attributs si besoin (rare dans TCX)
    return tree


def _get_text(el: Optional[ET.Element]) -> Optional[str]:
    return el.text.strip() if el is not None and el.text else None


def _extract_power(tp: ET.Element) -> Optional[int]:
    # Cas simple : <Power>
    power = tp.find("Power")
    if power is not None and power.text:
        try:
            return int(power.text)
        except ValueError:
            pass

    # Variante Garmin : dans Extensions/TPX/Watts ou Power under TPX
    ext = tp.find("Extensions")
    if ext is not None:
        tpx = ext.find(".//TPX")
        if tpx is not None:
            watts = tpx.find("Watts")
            if watts is not None and watts.text:
                try:
                    return int(watts.text)
                except ValueError:
                    pass
    return None


def _extract_cadence(tp: ET.Element) -> Optional[int]:
    cad = tp.find("Cadence")
    if cad is not None and cad.text:
        try:
            return int(cad.text)
        except ValueError:
            pass
    # parfois dans TPX (rare)
    ext = tp.find("Extensions")
    if ext is not None:
        tpx = ext.find(".//TPX")
        if tpx is not None:
            cad2 = tpx.find("Cadence")
            if cad2 is not None and cad2.text:
                try:
                    return int(cad2.text)
                except ValueError:
                    pass
    return None


def _extract_heart_rate(tp: ET.Element) -> Optional[int]:
    hr = tp.find("HeartRateBpm/Value")
    if hr is not None and hr.text:
        try:
            return int(hr.text)
        except ValueError:
            pass
    return None


def parse_tcx_and_store(input_path: str):
    input_path = Path(input_path)
    with SessionLocal() as db:
        # Dédupe raw file
        existing_raw = (
            db.execute(select(RawFile).where(RawFile.filename == input_path.name))
            .scalars()
            .first()
        )
        if existing_raw:
            print(
                f"[INFO] RawFile '{input_path.name}' already exists (id={existing_raw.id})"
            )
            # tenter de récupérer la session existante avec des trackpoints
            existing_session = None
            if existing_raw.sessions:
                # choisir la plus récente
                existing_session = sorted(
                    existing_raw.sessions,
                    key=lambda s: s.created_at or datetime.min,
                    reverse=True,
                )[0]
            if existing_session:
                # vérifier s’il y a déjà des trackpoints
                tp_exists = (
                    db.execute(
                        select(Trackpoint).where(
                            Trackpoint.session_id == existing_session.id
                        )
                    )
                    .scalars()
                    .first()
                )
                if tp_exists:
                    print(
                        f"[INFO] Found existing session (id={existing_session.id}) for file {input_path.name}"
                    )
                    print(f"ALREADY_PROCESSED {existing_session.id}")
                    return str(existing_session.id)
            raw_file = existing_raw
        else:
            raw_file = RawFile(filename=input_path.name)
            db.add(raw_file)
            db.commit()
            db.refresh(raw_file)
            print(f"[INFO] Created RawFile '{input_path.name}' (id={raw_file.id})")

        # Crée une nouvelle session
        session_id = str(uuid.uuid4())
        session_obj = SessionModel(id=session_id, raw_file_id=raw_file.id)
        db.add(session_obj)
        db.commit()
        db.refresh(session_obj)
        print(
            f"[INFO] Created new session (id={session_obj.id}) for file {input_path.name}"
        )

        # Parse XML
        try:
            tree = ET.parse(input_path)
            _strip_namespace(tree)
            root = tree.getroot()
        except Exception as e:
            print(f"[ERROR] Failed to parse XML from {input_path}: {e}")
            raise

        trackpoints = root.findall(".//Trackpoint")
        print(
            f"[DEBUG] Found {len(trackpoints)} Trackpoint elements in {input_path.name}"
        )

        if not trackpoints:
            print(f"[WARNING] No trackpoints found in {input_path.name}")
        try:
            for tp in trackpoints:
                time_el = tp.find("Time")
                if time_el is None or not time_el.text:
                    continue
                try:
                    timestamp = datetime.fromisoformat(
                        time_el.text.replace("Z", "+00:00")
                    )
                except Exception:
                    # fallback parser
                    continue

                power = _extract_power(tp)
                cadence = _extract_cadence(tp)
                heart_rate = _extract_heart_rate(tp)

                distance_el = tp.find("DistanceMeters")
                altitude_el = tp.find("AltitudeMeters")

                trackpoint = Trackpoint(
                    session_id=session_obj.id,
                    time=timestamp,
                    power=power,
                    cadence=cadence,
                    heart_rate=heart_rate,
                    distance_m=float(distance_el.text)
                    if distance_el is not None and distance_el.text
                    else None,
                    altitude_m=float(altitude_el.text)
                    if altitude_el is not None and altitude_el.text
                    else None,
                )
                db.add(trackpoint)
            db.commit()
        except Exception as e:
            print(f"[ERROR] Failed to ingest trackpoints: {e}")
            db.rollback()
            raise

        print(f"[INFO] Ingested trackpoints for session {session_obj.id}")
        return session_obj.id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCX and ingest into DB")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    args = parser.parse_args()
    parse_tcx_and_store(args.input)
