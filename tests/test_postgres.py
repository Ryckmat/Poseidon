"""Tests sur PostgreSQL : requêtes spécifiques et dashboard complet.

Lancés seulement si POSEIDON_TEST_DATABASE_URL est défini (base jetable,
vidée à chaque test).
"""

import os
from pathlib import Path

import pytest

from poseidon.config import AnalysisParams
from poseidon.db import Base, get_engine, get_session, init_db, reset_engine
from poseidon.ingest.store import ingest
from poseidon.processing.analysis import run_analysis

pytestmark = pytest.mark.postgres

TEST_URL = os.getenv("POSEIDON_TEST_DATABASE_URL")
APP = Path(__file__).parents[1] / "src" / "poseidon" / "dashboard" / "app.py"

if not TEST_URL:
    pytest.skip("POSEIDON_TEST_DATABASE_URL non défini", allow_module_level=True)


@pytest.fixture
def pg(monkeypatch):
    import streamlit as st

    # Le cache Streamlit est global au processus : repartir de zéro.
    st.cache_data.clear()
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    reset_engine()
    Base.metadata.drop_all(get_engine())
    init_db()
    yield
    reset_engine()


@pytest.fixture
def two_sessions(pg, write_tcx):
    def power(i):
        return 150 if 60 <= i < 400 else 80 + (i % 9) * 15

    ids = [
        ingest([write_tcx("s1.tcx", n=600, power=power)]),
        ingest([write_tcx("s2.tcx", start="2025-01-08T10:00:00Z", n=600, power=power)]),
    ]
    for session_id in ids:
        run_analysis(session_id, AnalysisParams())
    return ids


def test_init_db_is_idempotent(pg):
    init_db()
    init_db()


def test_downsampled_trackpoints(two_sessions):
    from poseidon.dashboard.data import TRACKPOINT_COLUMNS, fetch_trackpoints

    with get_session() as db:
        df = fetch_trackpoints(db, two_sessions[0], bucket_seconds=5)
    assert list(df.columns) == TRACKPOINT_COLUMNS
    assert len(df) == 120
    assert df["distance_m"].max() == pytest.approx(2396)
    assert df["heart_rate"].mean() == pytest.approx(140)
    assert df["speed_kmh"].notna().all()


def test_history(two_sessions):
    from poseidon.dashboard.data import fetch_history

    with get_session() as db:
        history = fetch_history(db)
    assert len(history) == 2
    assert history["normalized_power"].notna().all()
    assert history["avg_power"].between(80, 200).all()


def test_dashboard_renders_every_tab(two_sessions, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("POSEIDON_ENABLE_UPLOAD", "1")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception, app.exception
    assert len(app.tabs) == 3
    assert any("Distance" in m.label for m in app.metric)

    # Comparaison avec la seconde séance, puis passage en français.
    app.sidebar.selectbox[2].set_value(app.sidebar.selectbox[2].options[2]).run()
    assert not app.exception, app.exception
    app.sidebar.selectbox[0].set_value("fr").run()
    assert not app.exception, app.exception
    assert app.title[0].value.startswith("Poseidon : ")


def test_dashboard_without_sessions(pg):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.info
