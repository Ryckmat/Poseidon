"""Ligne de commande `poseidon`.

poseidon init-db
poseidon ingest FICHIER_OU_DOSSIER... [--analyze] [--skip-existing]
poseidon analyze SESSION_ID... | --all
poseidon list [--limit N]
poseidon delete SESSION_ID [--yes]
"""

import argparse
import logging
import sys
import uuid

from sqlalchemy.exc import SQLAlchemyError

from poseidon.config import AnalysisParams, ConfigError
from poseidon.db import get_session, init_db
from poseidon.db.repository import all_session_ids, delete_session, list_sessions
from poseidon.ingest.store import BoundaryFix, DuplicateSessionError, ingest
from poseidon.ingest.tcx import TcxError, expand_paths
from poseidon.processing.analysis import run_analysis

log = logging.getLogger("poseidon")


# ------------------------------------------------------------------ commandes
def cmd_init_db(args: argparse.Namespace) -> int:
    init_db()
    log.info("Tables et index à jour")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = expand_paths(args.paths)
    try:
        session_id = ingest(
            paths,
            name=args.name,
            boundary_fix=BoundaryFix(
                enabled=not args.no_fix_boundary_spikes,
                window_after_s=args.spike_window_after_s,
                seek_next_valid_s=args.spike_seek_next_valid_s,
                min_valid_w=args.spike_min_valid_w,
            ),
        )
    except DuplicateSessionError as exc:
        if not args.skip_existing:
            raise
        log.warning("%s, ignoré", exc)
        return 0
    if args.analyze:
        run_analysis(session_id, AnalysisParams.from_env())
    # Sortie standard : l'id seul, exploitable en script.
    print(session_id)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    if args.all:
        with get_session() as db:
            ids = all_session_ids(db)
    else:
        ids = args.session_ids
    if not ids:
        log.error("Aucune séance à analyser (donner des ids ou --all)")
        return 2
    params = AnalysisParams.from_env()
    failures = 0
    for session_id in ids:
        try:
            run_analysis(session_id, params)
        except (LookupError, ValueError) as exc:
            failures += 1
            log.error("%s", exc)
    return 1 if failures else 0


def cmd_list(args: argparse.Namespace) -> int:
    with get_session() as db:
        sessions = list_sessions(db, limit=args.limit)
    for s in sessions:
        start = s.start_time.strftime("%Y-%m-%d %H:%M") if s.start_time else "?"
        minutes = (s.duration_s or 0) / 60
        status = "analysée" if s.analyzed else "à analyser"
        print(
            f"{s.id}  {start}  {minutes:6.1f} min  {s.distance_km or 0:6.2f} km"
            f"  {status:<10}  {s.name}"
        )
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    if not args.yes:
        answer = input(f"Supprimer la séance {args.session_id} ? [o/N] ")
        if answer.strip().lower() not in {"o", "oui", "y", "yes"}:
            log.info("Abandon")
            return 1
    with get_session() as db:
        deleted = delete_session(db, args.session_id)
    if not deleted:
        log.error("Séance introuvable : %s", args.session_id)
        return 1
    log.info("Séance %s supprimée", args.session_id)
    return 0


# -------------------------------------------------------------------- parseur
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poseidon", description="Analyse de séances de rameur (TCX)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="crée les tables et index manquants")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser(
        "ingest",
        help="importe une séance (plusieurs fichiers ou un dossier = fusion)",
    )
    p.add_argument("paths", nargs="+", help="fichiers .tcx ou dossier de .tcx")
    p.add_argument("--name", help="nom de la séance (défaut : nom du fichier)")
    p.add_argument("--analyze", action="store_true", help="analyse dans la foulée")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="ignore sans erreur une séance déjà importée",
    )
    fix = p.add_argument_group("correction des jonctions (fusion)")
    fix.add_argument("--no-fix-boundary-spikes", action="store_true")
    fix.add_argument("--spike-window-after-s", type=float, default=3.0)
    fix.add_argument("--spike-seek-next-valid-s", type=float, default=6.0)
    fix.add_argument("--spike-min-valid-w", type=float, default=20.0)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("analyze", help="(re)calcule l'analyse de séances")
    p.add_argument("session_ids", nargs="*", type=uuid.UUID, metavar="SESSION_ID")
    p.add_argument("--all", action="store_true", help="toutes les séances")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("list", help="liste les séances")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("delete", help="supprime une séance et ses données")
    p.add_argument("session_id", type=uuid.UUID)
    p.add_argument("-y", "--yes", action="store_true", help="sans confirmation")
    p.set_defaults(func=cmd_delete)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        return args.func(args)
    except (ConfigError, TcxError, DuplicateSessionError, LookupError) as exc:
        log.error("%s", exc)
        return 1
    except SQLAlchemyError as exc:
        detail = str(getattr(exc, "orig", None) or exc).strip().splitlines()[0]
        log.error("Erreur base de données : %s", detail)
        log.debug("Détail", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
