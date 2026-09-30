![CI](https://github.com/Ryckmat/Poseidon/actions/workflows/lint.yml/badge.svg)

# Poseidon

Analyse de séances de rameur à partir de fichiers TCX : puissance, cadence,
vitesse, segments d'effort stable, FTP / NP / TSS et tendances hebdomadaires.

```
TCX ──> poseidon-ingest ──> PostgreSQL ──> poseidon-analyze ──> dashboard Streamlit
         (1 ou N fichiers,     (raw_files,      (dérivés par point,
          fusion continue)      sessions,        segments stables,
                                trackpoints)     régressions)
```

## Structure

```
src/poseidon/
├── config.py              # variables d'environnement, AnalysisParams
├── db/                    # modèles SQLAlchemy, connexion
├── ingest/
│   ├── tcx.py             # lecture TCX
│   ├── merge.py           # fusion multi-fichiers + lissage aux jonctions
│   ├── store.py           # écriture en base
│   └── cli.py             # poseidon-ingest
├── processing/
│   ├── metrics.py         # calculs purs (pandas), partagés job / dashboard
│   └── analysis.py        # poseidon-analyze
└── dashboard/             # Streamlit : app, graphiques, PDF, libellés en/fr
tests/                     # pytest
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # renseigner DATABASE_URL
poseidon-init-db          # crée les tables
```

## Utilisation

```bash
# Une séance = un fichier
poseidon-ingest data/seance.tcx

# Une séance coupée en plusieurs fichiers : fusion en timeline continue
poseidon-ingest data/part1.tcx data/part2.tcx --name "2025-03-08 fractionné"

# Analyse (l'id est affiché en dernière ligne par poseidon-ingest)
poseidon-analyze <session_id>

# Dashboard
streamlit run src/poseidon/dashboard/app.py
```

Options de fusion : `--no-fix-boundary-spikes`, `--spike-window-after-s`,
`--spike-seek-next-valid-s`, `--spike-min-valid-w` (voir `poseidon-ingest -h`).

## Paramètres d'analyse

Lus depuis l'environnement par `poseidon-analyze`, modifiables en direct dans
le dashboard.

| Variable                | Défaut | Rôle                                              |
| ----------------------- | ------ | ------------------------------------------------- |
| `MAX_POWER`             | 250    | Puissance au-delà de laquelle un point est écarté |
| `MIN_STABLE_POWER`      | 50     | Puissance min d'un segment stable                 |
| `STABLE_WINDOW_S`       | 30     | Fenêtre de l'écart-type glissant                  |
| `STABLE_STD_THRESHOLD`  | 5      | Écart-type max pour être « stable »               |
| `MIN_STABLE_DURATION_S` | 60     | Durée min d'un segment stable                     |

## CI

- `lint.yml` : black, isort, flake8, pytest.
- `process-tcx.yml` : à chaque push d'un `.tcx` dans `data/`, ingère puis
  analyse les fichiers ajoutés. Nécessite le secret `DATABASE_URL`.

## Développement

```bash
black src tests && isort src tests && flake8 src tests && pytest
```
