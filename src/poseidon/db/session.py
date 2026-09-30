"""Connexion à la base, créée à la première utilisation."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from poseidon.config import database_url
from poseidon.db.models import Base

# Préfixes hérités, redirigés vers le pilote psycopg 3 (défaut de SQLAlchemy
# pour postgresql://) : postgres:// (Heroku, certains hébergeurs) et
# postgresql+psycopg2:// (pilote retiré des dépendances).
_LEGACY_PREFIXES = ("postgres://", "postgresql+psycopg2://")


def normalize_url(url: str) -> str:
    for prefix in _LEGACY_PREFIXES:
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    # pool_pre_ping : les hébergeurs (Supabase, Neon...) coupent les
    # connexions inactives, on les teste avant réutilisation.
    return create_engine(normalize_url(database_url()), pool_pre_ping=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def reset_engine() -> None:
    """Oublie le moteur courant (changement de DATABASE_URL, tests)."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
    _session_factory.cache_clear()


@contextmanager
def get_session() -> Iterator[OrmSession]:
    """Session transactionnelle : commit en sortie, rollback sur exception."""
    db = _session_factory()()
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Crée les tables et index manquants. Idempotent."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    # create_all ne crée les index que des tables nouvelles.
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            index.create(engine, checkfirst=True)
