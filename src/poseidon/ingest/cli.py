"""CLI : poseidon-ingest FICHIER.tcx [FICHIER.tcx ...]"""

import argparse

from poseidon.ingest.store import BoundaryFix, ingest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingère un ou plusieurs TCX comme une seule séance. "
            "Plusieurs fichiers sont fusionnés en timeline continue."
        )
    )
    parser.add_argument("paths", nargs="+", help="Fichiers .tcx de la séance")
    parser.add_argument("--name", help="Nom de la séance (défaut: nom du fichier)")
    fix = parser.add_argument_group("correction des jonctions (multi-fichiers)")
    fix.add_argument("--no-fix-boundary-spikes", action="store_true")
    fix.add_argument("--spike-window-after-s", type=float, default=3.0)
    fix.add_argument("--spike-seek-next-valid-s", type=float, default=6.0)
    fix.add_argument("--spike-min-valid-w", type=float, default=20.0)
    args = parser.parse_args(argv)

    paths = list(dict.fromkeys(args.paths))
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
    # Dernière ligne = id de séance, lue par le workflow CI.
    print(session_id)


if __name__ == "__main__":
    main()
