"""Import, analyse et CLI de bout en bout sur une base SQLite."""

import uuid

import pytest
from sqlalchemy import func, select

from poseidon.cli import main
from poseidon.config import AnalysisParams
from poseidon.db import (
    RawFile,
    Regression,
    Session,
    StableSegment,
    Trackpoint,
    get_session,
)
from poseidon.db.repository import delete_session, list_sessions
from poseidon.ingest.store import DuplicateSessionError, ingest
from poseidon.processing.analysis import run_analysis

PARAMS = AnalysisParams(min_stable_duration_s=60, reference_ftp_w=160)


def _stable_session(write_tcx, name="seance.tcx", **kw):
    # 30 s d'échauffement irrégulier, 3 min stables à 150 W, 30 s de retour.
    def power(i):
        return 150 if 30 <= i < 210 else 60 + (i % 7) * 20

    return write_tcx(name, n=240, power=power, **kw)


def _count(db, model, session_id):
    return db.scalar(
        select(func.count()).select_from(model).where(model.session_id == session_id)
    )


def test_ingest_then_analyze_persists_results(database, write_tcx):
    session_id = ingest([_stable_session(write_tcx)])
    run_analysis(session_id, PARAMS)

    with get_session() as db:
        session = db.get(Session, session_id)
        assert session.duration_s == 239
        assert session.distance_km == pytest.approx(0.956)
        assert session.normalized_power > 0
        # Séance de 4 min : pas de FTP estimée, TSS sur la FTP de référence.
        assert session.ftp_estimated is None
        assert session.tss > 0
        assert session.avg_heart_rate == pytest.approx(140)
        assert _count(db, Trackpoint, session_id) == 240
        assert _count(db, StableSegment, session_id) == 1
        assert _count(db, Regression, session_id) == 2
        filtered = db.scalar(
            select(func.count())
            .select_from(Trackpoint)
            .where(
                Trackpoint.session_id == session_id,
                Trackpoint.speed_calc_kmh.is_not(None),
            )
        )
        assert filtered == 239
        raw = db.get(RawFile, session.raw_file_id)
        assert raw.file_metadata["files"][0]["sha256"]


def test_reanalysis_replaces_previous_results(database, write_tcx):
    session_id = ingest([_stable_session(write_tcx)])
    run_analysis(session_id, PARAMS)
    run_analysis(session_id, PARAMS)
    with get_session() as db:
        assert _count(db, StableSegment, session_id) == 1
        assert _count(db, Regression, session_id) == 2


def test_duplicate_import_is_rejected(database, write_tcx):
    path = _stable_session(write_tcx)
    first = ingest([path])
    with pytest.raises(DuplicateSessionError) as exc:
        ingest([path])
    assert exc.value.session_id == first


def test_merge_import_and_listing(database, write_tcx):
    a = write_tcx("part1.tcx", n=60)
    b = write_tcx("part2.tcx", start="2025-01-01T12:00:00Z", n=60, dist0=1000)
    session_id = ingest([b, a], name="fractionne")
    with get_session() as db:
        (info,) = list_sessions(db)
        raw = db.scalar(select(RawFile).where(RawFile.filename == "fractionne"))
    assert info.id == session_id
    assert info.name == "fractionne"
    assert info.duration_s == 118
    assert not info.analyzed
    assert raw.file_metadata["mode"] == "continuous_no_gap"


def test_delete_session_removes_everything(database, write_tcx):
    session_id = ingest([_stable_session(write_tcx)])
    run_analysis(session_id, PARAMS)
    with get_session() as db:
        assert delete_session(db, session_id)
    with get_session() as db:
        for model in (RawFile, Session, Trackpoint, StableSegment, Regression):
            assert db.scalar(select(func.count()).select_from(model)) == 0
        assert not delete_session(db, session_id)


def test_analysis_of_unknown_session_fails(database):
    with pytest.raises(LookupError):
        run_analysis(uuid.uuid4(), PARAMS)


# --------------------------------------------------------------------- CLI
def test_cli_full_cycle(database, write_tcx, tmp_path, capsys):
    folder = tmp_path / "2025-01-01"
    write_tcx("2025-01-01/a.tcx", n=60)
    write_tcx("2025-01-01/b.tcx", start="2025-01-01T11:00:00Z", n=60)

    assert main(["init-db"]) == 0
    assert main(["ingest", str(folder), "--name", "dossier", "--analyze"]) == 0
    session_id = capsys.readouterr().out.strip()
    uuid.UUID(session_id)

    # Réimport : erreur, sauf avec --skip-existing.
    assert main(["ingest", str(folder), "--name", "dossier"]) == 1
    assert main(["ingest", str(folder), "--name", "dossier", "--skip-existing"]) == 0
    capsys.readouterr()

    assert main(["list"]) == 0
    listing = capsys.readouterr().out
    assert session_id in listing and "analysée" in listing

    assert main(["analyze", "--all"]) == 0
    assert main(["analyze", str(uuid.uuid4())]) == 1
    assert main(["delete", session_id, "--yes"]) == 0
    assert main(["delete", session_id, "--yes"]) == 1


def test_cli_analyze_arguments(database):
    # Base vide : --all n'a rien à faire mais réussit (workflow de maintenance).
    assert main(["analyze", "--all"]) == 0
    # Ni id ni --all : erreur d'usage.
    assert main(["analyze"]) == 2


def test_cli_reports_bad_file(database, tmp_path, caplog):
    bad = tmp_path / "bad.tcx"
    bad.write_text("pas du xml")
    assert main(["ingest", str(bad)]) == 1
    assert "XML invalide" in caplog.text


def test_cli_without_database_url(monkeypatch, caplog):
    from poseidon.db import reset_engine

    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_engine()
    assert main(["list"]) == 1
    assert "DATABASE_URL" in caplog.text
