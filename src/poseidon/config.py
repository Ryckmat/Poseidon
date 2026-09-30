"""Configuration lue depuis l'environnement (fichier .env accepté)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class AnalysisParams:
    """Paramètres de filtrage et de détection des segments stables."""

    max_power: float = 250.0
    min_stable_power: float = 50.0
    std_window_s: float = 30.0
    std_threshold: float = 5.0
    min_stable_duration_s: float = 60.0

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
        )


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL n'est pas défini (voir .env.example)")
    return url
