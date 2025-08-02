import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from db.models import (
    RawFile,
    SessionLocal,
    Trackpoint,
)

def find_descendants(elem, name):
    for descendant in elem.iter():
        if descendant.tag.endswith("}" + name) or descendant.tag == name or descendant.tag.split("}")[-1] == name:
            yield descendant

def get_text_by_path(parent, *names):
    cur = parent
    for name in names:
        if cur is None:
            return None
        found = None
        for child in cur:
            if child.tag.endswith("}" + name) or child.tag == name or child.tag.split("}")[-1] == name:
                found = child
                break
        cur = found
    if cur is not None and cur.text:
        return cur.text.strip()
    return None

def to_float(text):
    try:
        return float(text)
    except Exception:
        return None

def parse_tcx_and_store(input_path: str):
    input_path = Path(input_path)
    db: Session = SessionLocal()
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to parse XML from {input_path}: {e}")
        db.close()
        raise

    try:
        session_id = uuid.uuid4()
        raw_file = RawFile(
            id=session_id,
            filename=input_path.name,
        )
        db.add(raw_file)
        db.commit()
        print(f"[INFO] Inserted RawFile with id {session_id}")
    except Exception as e:
        print(f"[ERROR] Failed to insert RawFile: {e}")
        db.rollback()
        db.close()
        raise

    # Extraction Trackpoints
    trackpoint_elements = list(find_descendants(root, "Trackpoint"))
    print(f"[DEBUG] Found {len(trackpoint_elements)} Trackpoint elements in {input_path.name}")

    if not trackpoint_elements:
        print("[WARN] No trackpoints found, aborting session creation.")
        db.close()
        return None

    new_points = []
    for tp in trackpoint_elements:
        time_text = get_text_by_path(tp, "Time")
        if not time_text:
            continue
        try:
            timestamp = datetime.fromisoformat(time_text.replace("Z", "+00:00"))
        except Exception as e:
            print(f"[WARN] Failed to parse time: {time_text} ({e})")
            continue

        power_val = to_float(get_text_by_path(tp, "Power"))
        cadence_val = None
        cad_node = None
        for child in tp:
            if child.tag.endswith("Cadence") or child.tag.split("}")[-1] == "Cadence":
                cad_node = child
                break
        if cad_node is not None and cad_node.text and cad_node.text.isdigit():
            cadence_val = int(cad_node.text)

        trackpoint = Trackpoint(
            id=uuid.uuid4(),
            session_id=session_id,  # UUID type
            time=timestamp,
            power=power_val,
            cadence=cadence_val,
        )
        print(f"[INFO] Prepared trackpoint: time={timestamp}, power={power_val}, cadence={cadence_val}")
        new_points.append(trackpoint)

    try:
        db.add_all(new_points)
        print(f"[DEBUG] Trackpoints to commit: {len(new_points)}")
        db.commit()
        print(f"[INFO] Successfully inserted {len(new_points)} trackpoints.")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to ingest trackpoints: {e}")
        traceback.print_exc()
        db.rollback()
        db.close()
        raise
    finally:
        db.close()

    print(f"Ingested TCX file as session {session_id}")
    return str(session_id)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse TCX and ingest into DB")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    args = parser.parse_args()
    parse_tcx_and_store(args.input)
