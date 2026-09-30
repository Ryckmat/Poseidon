"""Connexion à la base, créée à la première utilisation."""

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from poseidon.config import database_url
from poseidon.db.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(database_url(), future=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Iterator[OrmSession]:
    """Session transactionnelle : commit en sortie, rollback sur exception."""
    db = _session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())
