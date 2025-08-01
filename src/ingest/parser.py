import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from db.models import (  # adapte l'import si tu ranges différemment
    RawFile,
    SessionLocal,
    Trackpoint,
)


def parse_tcx_and_store(input_path: str):
    input_path = Path(input_path)
    db: Session = SessionLocal()
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to parse XML from {input_path}: {e}")
        raise

    # Exemple: extraction basique, à adapter selon ton TCX
    try:
        session_id = str(uuid.uuid4())
        raw_file = RawFile(
            id=session_id, filename=input_path.name, created_at=datetime.utcnow()
        )
        db.add(raw_file)
        db.commit()
    except Exception as e:
        print(f"[ERROR] Failed to insert RawFile: {e}")
        db.rollback()
        raise

    # Exemple d'altitudes : si tu n'en as pas besoin, supprime ; sinon utilise
    altitudes = []
    for tp in root.findall(".//Trackpoint"):
        ele = tp.find("AltitudeMeters")
        if ele is not None and ele.text:
            try:
                altitudes.append(float(ele.text))
            except ValueError:
                continue  # ignore malformé

    # Utilisation minimale pour éviter F841
    avg_altitude = None
    if altitudes:
        avg_altitude = sum(altitudes) / len(altitudes)
        # tu peux logger ou enregistrer avg_altitude quelque part
        print(f"[INFO] Average altitude: {avg_altitude:.2f} meters")

    # Ici : parse les trackpoints et insère
    try:
        for tp in root.findall(".//Trackpoint"):
            time_el = tp.find("Time")
            power_el = tp.find(".//Power")  # selon ton schema
            cadence_el = tp.find(".//Cadence")
            if time_el is None or time_el.text is None:
                continue
            timestamp = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
            power = (
                int(power_el.text) if power_el is not None and power_el.text else None
            )
            cadence = (
                int(cadence_el.text)
                if cadence_el is not None and cadence_el.text
                else None
            )

            trackpoint = Trackpoint(
                id=str(uuid.uuid4()),
                session_id=session_id,
                time=timestamp,
                power=power,
                cadence=cadence,
            )
            db.add(trackpoint)
        db.commit()
    except Exception as e:
        print(f"[ERROR] Failed to ingest trackpoints: {e}")
        db.rollback()
        raise
    finally:
        db.close()

    print(f"Ingested TCX file as session {session_id}")
    return session_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCX and ingest into DB")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    args = parser.parse_args()
    parse_tcx_and_store(args.input)
