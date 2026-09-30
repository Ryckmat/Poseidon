from pathlib import Path

import pytest

TCX_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase
    xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2">
  <Activities><Activity Sport="Other"><Lap><Track>
{points}
  </Track></Lap></Activity></Activities>
</TrainingCenterDatabase>
"""

POINT = """    <Trackpoint>
      <Time>{time}</Time>
      <DistanceMeters>{distance}</DistanceMeters>
      <Cadence>{cadence}</Cadence>
      <Extensions><ns3:TPX><ns3:Watts>{power}</ns3:Watts></ns3:TPX></Extensions>
    </Trackpoint>"""


@pytest.fixture
def write_tcx(tmp_path: Path):
    """Écrit un TCX de `n` points à 1 s d'intervalle, 4 m/s, puissance fixe."""

    def _write(name, start="2025-01-01T10:00:00Z", n=10, power=150, dist0=0.0):
        from datetime import datetime, timedelta

        t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
        points = "\n".join(
            POINT.format(
                time=(t0 + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
                distance=dist0 + 4.0 * i,
                cadence=26,
                power=power,
            )
            for i in range(n)
        )
        path = tmp_path / name
        path.write_text(TCX_TEMPLATE.format(points=points))
        return str(path)

    return _write
