import os
import sys
from pathlib import Path

# Ensure database URL is set before importing parser and make src importable
os.environ.setdefault("DATABASE_URL", "sqlite://")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ingest import parser


def test_ingest_file_no_timestamps(tmp_path, monkeypatch, capsys):
    class DummySession:
        def add(self, _):
            pass

        def flush(self):
            pass

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(parser, "SessionLocal", lambda: DummySession())

    tcx_content = (
        """<?xml version='1.0' encoding='UTF-8'?>
<TrainingCenterDatabase xmlns='http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'>
  <Activities>
    <Activity Sport='Running'>
      <Id>2023-01-01T00:00:00Z</Id>
      <Lap StartTime='2023-01-01T00:00:00Z'>
        <Track>
          <Trackpoint>
            <AltitudeMeters>0</AltitudeMeters>
            <DistanceMeters>0</DistanceMeters>
          </Trackpoint>
        </Track>
      </Lap>
    </Activity>
  </Activities>
</TrainingCenterDatabase>
"""
    )

    tcx_path = tmp_path / "no_times.tcx"
    tcx_path.write_text(tcx_content)

    parser.ingest_file(str(tcx_path))
    captured = capsys.readouterr()
    assert "No valid timestamps found." in captured.out
