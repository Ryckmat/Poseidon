import math
from datetime import timedelta

from poseidon.ingest.merge import merge_continuous, smooth_boundary_power
from poseidon.ingest.tcx import last_distance, parse_tcx, time_bounds


def test_parse_tcx_reads_all_fields(write_tcx):
    points = parse_tcx(write_tcx("a.tcx", n=3, power=180))
    assert len(points) == 3
    assert points[0].time.tzinfo is not None
    assert points[1].distance == 4.0
    assert points[2].cadence == 26
    assert points[0].power == 180.0
    assert math.isnan(points[0].altitude)
    assert points[0].heart_rate is None


def test_merge_is_continuous_and_chains_distance(write_tcx):
    # Le 2e fichier démarre 1 h plus tard et est passé en premier.
    late = parse_tcx(write_tcx("b.tcx", start="2025-01-01T11:00:00Z", dist0=500))
    early = parse_tcx(write_tcx("a.tcx"))
    merged, boundaries = merge_continuous([late, early])

    start, end = time_bounds(merged)
    assert start == early[0].time
    assert end - start == timedelta(seconds=18)
    assert boundaries == [early[-1].time]
    assert last_distance(merged) == 72.0


def test_smooth_boundary_power_fills_dip(write_tcx):
    a = parse_tcx(write_tcx("a.tcx", power=200))
    b = parse_tcx(write_tcx("b.tcx", start="2025-01-01T11:00:00Z", power=200))
    b[0].power = 0.0
    merged, boundaries = merge_continuous([a, b])
    smooth_boundary_power(merged, boundaries)
    assert all(p.power >= 190 for p in merged)
