"""Configuration lue depuis l'environnement (un fichier .env est accepté)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_TRUE = {"1", "true", "yes", "on"}


class ConfigError(RuntimeError):
    """Configuration absente ou invalide."""


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} doit être un nombre, reçu {raw!r}") from exc


def _optional_float(name: str) -> float | None:
    value = _float(name, 0.0)
    return value if value > 0 else None


@dataclass(frozen=True)
class AnalysisParams:
    """Paramètres de filtrage, de détection des segments stables et de charge."""

    max_power: float = 250.0
    min_stable_power: float = 50.0
    std_window_s: float = 30.0
    std_threshold: float = 5.0
    min_stable_duration_s: float = 60.0
    reference_ftp_w: float | None = None

    @classmethod
    def from_env(cls) -> "AnalysisParams":
        return cls(
            max_power=_float("MAX_POWER", cls.max_power),
            min_stable_power=_float("MIN_STABLE_POWER", cls.min_stable_power),
            std_window_s=_float("STABLE_WINDOW_S", cls.std_window_s),
            std_threshold=_float("STABLE_STD_THRESHOLD", cls.std_threshold),
            min_stable_duration_s=_float(
                "MIN_STABLE_DURATION_S", cls.min_stable_duration_s
            ),
            reference_ftp_w=_optional_float("FTP_W"),
        )


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise ConfigError("DATABASE_URL n'est pas défini (voir .env.example)")
    return url


def upload_enabled() -> bool:
    """Import de fichiers depuis le dashboard (désactivé par défaut)."""
    return os.getenv("POSEIDON_ENABLE_UPLOAD", "").strip().lower() in _TRUE
