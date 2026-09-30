from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from poseidon.db import init_db, reset_engine

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
      <HeartRateBpm><Value>{hr}</Value></HeartRateBpm>
      <Cadence>{cadence}</Cadence>
      <Extensions><ns3:TPX><ns3:Watts>{power}</ns3:Watts></ns3:TPX></Extensions>
    </Trackpoint>"""


@pytest.fixture
def write_tcx(tmp_path: Path) -> Callable[..., str]:
    """Écrit un TCX de `n` points à 1 s d'intervalle, 4 m/s, 26 coups/min.

    `power` est une valeur fixe ou une fonction de l'index du point.
    """

    def _write(
        name: str,
        start: str = "2025-01-01T10:00:00Z",
        n: int = 10,
        power=150,
        dist0: float = 0.0,
        hr: int = 140,
    ) -> str:
        t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
        points = "\n".join(
            POINT.format(
                time=(t0 + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
                distance=dist0 + 4.0 * i,
                hr=hr,
                cadence=26,
                power=power(i) if callable(power) else power,
            )
            for i in range(n)
        )
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TCX_TEMPLATE.format(points=points))
        return str(path)

    return _write


@pytest.fixture
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Base SQLite vierge, branchée via DATABASE_URL."""
    url = f"sqlite:///{tmp_path / 'poseidon.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine()
    init_db()
    yield url
    reset_engine()
