import pytest

from poseidon.config import AnalysisParams, ConfigError, upload_enabled
from poseidon.dashboard import i18n
from poseidon.dashboard.charts import elapsed_ticks
from poseidon.dashboard.formatting import fmt, hhmmss, human_duration, split


def test_formatting():
    assert hhmmss(3725) == "01:02:05"
    assert hhmmss(None) == "00:00:00"
    assert human_duration(3725) == "1h 2m 5s"
    assert human_duration(float("nan")) == "-"
    assert split(125.04) == "2:05.0"
    assert split(0) == "-"
    assert fmt(1.234) == "1.23"
    assert fmt(None) == "-"
    assert fmt(float("inf")) == "-"


def test_translations_complete_and_formattable():
    for lang in i18n.LANGUAGES:
        t = i18n.Translator(lang)
        for key in i18n.all_keys():
            assert t(key)
    assert "5" in i18n.Translator("fr")("max_streak", duration="5")
    assert i18n.Translator("de").lang == "en"


def test_elapsed_ticks_bounded():
    values, labels = elapsed_ticks(0, 7200)
    assert len(values) <= 14
    assert labels[1] == "00:10:00"


def test_params_from_env(monkeypatch):
    monkeypatch.setenv("MAX_POWER", "300")
    monkeypatch.setenv("FTP_W", "180")
    monkeypatch.setenv("STABLE_WINDOW_S", "")
    params = AnalysisParams.from_env()
    assert params.max_power == 300
    assert params.reference_ftp_w == 180
    assert params.std_window_s == AnalysisParams.std_window_s


def test_params_from_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv("MAX_POWER", "beaucoup")
    with pytest.raises(ConfigError, match="MAX_POWER"):
        AnalysisParams.from_env()


def test_database_url_normalization():
    from poseidon.db.session import normalize_url

    assert normalize_url("postgres://u:p@h/db") == "postgresql://u:p@h/db"
    assert normalize_url("postgresql+psycopg2://h/db") == "postgresql://h/db"
    assert normalize_url("postgresql://h/db") == "postgresql://h/db"
    assert normalize_url("sqlite:///x.db") == "sqlite:///x.db"


def test_upload_flag(monkeypatch):
    monkeypatch.delenv("POSEIDON_ENABLE_UPLOAD", raising=False)
    assert not upload_enabled()
    monkeypatch.setenv("POSEIDON_ENABLE_UPLOAD", "true")
    assert upload_enabled()
