import math
from datetime import timedelta

import pytest

from poseidon.ingest.merge import merge_continuous, smooth_boundary_power
from poseidon.ingest.tcx import (
    TcxError,
    expand_paths,
    last_distance,
    parse_tcx,
    time_bounds,
)


def test_parse_tcx_reads_all_fields(write_tcx):
    points = parse_tcx(write_tcx("a.tcx", n=3, power=180, hr=150))
    assert len(points) == 3
    assert points[0].time.tzinfo is not None
    assert points[1].distance == 4.0
    assert points[2].cadence == 26
    assert points[0].power == 180.0
    assert points[0].heart_rate == 150
    assert math.isnan(points[0].altitude)


def test_parse_tcx_rejects_invalid_xml(tmp_path):
    bad = tmp_path / "bad.tcx"
    bad.write_text("<TrainingCenterDatabase><oops>")
    with pytest.raises(TcxError, match="XML invalide"):
        parse_tcx(str(bad))


def test_parse_tcx_rejects_missing_file(tmp_path):
    with pytest.raises(TcxError, match="lecture impossible"):
        parse_tcx(str(tmp_path / "absent.tcx"))


def test_parse_tcx_rejects_file_without_points(tmp_path):
    empty = tmp_path / "empty.tcx"
    empty.write_text(
        '<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/'
        'TrainingCenterDatabase/v2"/>'
    )
    with pytest.raises(TcxError, match="aucun point"):
        parse_tcx(str(empty))


def test_parse_tcx_rejects_entity_expansion(tmp_path):
    bomb = tmp_path / "bomb.tcx"
    bomb.write_text(
        '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>'
        "<TrainingCenterDatabase>&a;</TrainingCenterDatabase>"
    )
    with pytest.raises(TcxError):
        parse_tcx(str(bomb))


def test_expand_paths_lists_directory_sorted(write_tcx, tmp_path):
    write_tcx("seance/b.tcx")
    write_tcx("seance/a.tcx")
    single = write_tcx("c.tcx")
    paths = expand_paths([str(tmp_path / "seance"), single, single])
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["a.tcx", "b.tcx", "c.tcx"]


def test_expand_paths_rejects_empty_directory(tmp_path):
    (tmp_path / "vide").mkdir()
    with pytest.raises(TcxError):
        expand_paths([str(tmp_path / "vide")])


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


def test_smooth_boundary_power_copies_previous_without_valid_next(write_tcx):
    a = parse_tcx(write_tcx("a.tcx", power=200))
    b = parse_tcx(write_tcx("b.tcx", start="2025-01-01T11:00:00Z", power=0))
    merged, boundaries = merge_continuous([a, b])
    smooth_boundary_power(merged, boundaries, window_after_s=2)
    (boundary,) = boundaries
    for p in merged:
        offset = (p.time - boundary).total_seconds()
        expected = 0.0 if offset > 2 else 200.0
        assert p.power == expected, offset


def test_merge_drops_first_point_of_next_file_at_junction(write_tcx):
    # Fusion bout à bout : le 1er point du fichier suivant tombe sur
    # l'horodatage du dernier point du précédent et n'est pas dupliqué.
    a = parse_tcx(write_tcx("a.tcx", n=10))
    b = parse_tcx(write_tcx("b.tcx", start="2025-01-01T11:00:00Z", n=10))
    merged, _ = merge_continuous([a, b])
    assert len(merged) == 19
