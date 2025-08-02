# src/ingest/parser.py
import os
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from db.models import SessionLocal, RawFile, Session, Trackpoint
from sqlalchemy.exc import IntegrityError

load_dotenv()

NS = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}


def parse_tcx(path):
    tree = ET.parse(path)
    root = tree.getroot()

    data = []
    for tp in root.findall('.//ns:Trackpoint', NS):
        time_elem = tp.find('ns:Time', NS)
        time = None
        if time_elem is not None:
            txt = time_elem.text
            if txt.endswith("Z"):
                txt = txt.replace("Z", "+00:00")
            time = datetime.fromisoformat(txt)
        altitude = tp.find('ns:AltitudeMeters', NS)
        distance = tp.find('ns:DistanceMeters', NS)
        hr_elem = tp.find('.//ns:HeartRateBpm/ns:Value', NS)
        cadence = tp.find('ns:Cadence', NS)

        # power might live in Extensions
        power = None
        ext = tp.find('ns:Extensions', NS)
        if ext is not None:
            for child in ext.iter():
                tag_lower = child.tag.lower()
                if 'watts' in tag_lower or 'power' in tag_lower:
                    try:
                        power = float(child.text)
                    except:
                        pass

        row = {
            'time': time,
            'altitude': float(altitude.text) if altitude is not None else np.nan,
            'distance': float(distance.text) if distance is not None else np.nan,
            'heart_rate': int(hr_elem.text) if hr_elem is not None else None,
            'cadence': int(cadence.text) if cadence is not None else None,
            'power': float(power) if power is not None else None
        }
        data.append(row)
    return data


def ingest_file(tcx_path):
    session_db = SessionLocal()
    try:
        # create or register raw file
        raw = RawFile(filename=os.path.basename(tcx_path), metadata={})
        session_db.add(raw)
        session_db.flush()  # get raw.id

        # parse points
        points = parse_tcx(tcx_path)
        if not points:
            print("No trackpoints found.")
            return

        # derive session-level bounds
        times = [p['time'] for p in points if p['time'] is not None]
        distances = [p['distance'] for p in points if p['distance'] is not None]
        altitudes = [p['altitude'] for p in points if p['altitude'] is not None]

        start_time = min(times)
        end_time = max(times)
        duration_s = (end_time - start_time).total_seconds() if start_time and end_time else 0
        total_distance_km = (max(distances) / 1000.0) if distances else 0
        elevation_gain = 0  # placeholder, can improve later
        avg_hr = np.nan
        avg_speed = np.nan

        session = Session(
            raw_file_id=raw.id,
            start_time=start_time,
            end_time=end_time,
            duration_s=duration_s,
            distance_km=total_distance_km,
            elevation_gain_m=elevation_gain,
            avg_heart_rate=avg_hr,
            avg_speed_kmh=avg_speed,
        )
        session_db.add(session)
        session_db.flush()

        # insert trackpoints (derived fields will be computed later)
        for p in points:
            tp = Trackpoint(
                session_id=session.id,
                time=p['time'],
                distance_m=p['distance'],
                altitude_m=p['altitude'],
                heart_rate=p['heart_rate'],
                cadence=p['cadence'],
                power=p['power'],
                power_filtered=None,
                speed_calc_kmh=None,
                pace_min_per_km=None,
                elevation_diff=None,
            )
            session_db.add(tp)

        session_db.commit()
        print(f"Ingested TCX file as session {session.id}")
    except Exception as e:
        session_db.rollback()
        print("Error ingesting:", e)
    finally:
        session_db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a TCX file into Postgres")
    parser.add_argument("--input", required=True, help="Path to .tcx file")
    args = parser.parse_args()
    ingest_file(args.input)