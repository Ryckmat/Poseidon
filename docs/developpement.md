# Développement

## Environnement

```bash
python -m venv .venv && source .venv/bin/activate
make install        # paquet en mode éditable, outils, hooks pre-commit
make db-up          # PostgreSQL local : bases poseidon et poseidon_test
cp .env.example .env
```

Un devcontainer (`.devcontainer/`) fournit le même environnement dans
Codespaces ou VS Code.

`make help` liste les commandes disponibles.

## Tests

```bash
make test           # unitaires et pipeline complet sur SQLite
make test-all       # + PostgreSQL et rendu du dashboard (make db-up avant)
```

| Fichier                         | Portée                                                   |
| ------------------------------- | -------------------------------------------------------- |
| `test_tcx_merge.py`             | lecture TCX, fichiers invalides, fusion, lissage         |
| `test_metrics.py`               | chaque indicateur sur des séances synthétiques           |
| `test_dashboard_helpers.py`     | formatage, traductions, configuration                    |
| `test_pipeline.py`              | import, analyse, suppression et CLI sur SQLite           |
| `test_postgres.py`              | agrégation SQL, progression, dashboard complet (`AppTest`) |

Les tests PostgreSQL ne tournent que si `POSEIDON_TEST_DATABASE_URL` est
défini ; ils vident la base à chaque test, ne jamais la faire pointer vers une
base de production. La CI les exécute avec un service PostgreSQL.

Les fichiers TCX de test sont générés par la fixture `write_tcx`
(`tests/conftest.py`) : pas de fichier binaire dans le dépôt.

## Conventions

- Format et lint : `ruff` (`make format`, `make lint`), vérifiés par
  pre-commit et par la CI.
- Toute nouvelle formule va dans `processing/metrics.py`, avec un test.
- Tout nouveau libellé du dashboard va dans `dashboard/i18n.py`, en anglais et
  en français (un test vérifie que chaque clé est traduite).
- Dépendances figées dans `pyproject.toml`, seule source de vérité (ruff
  compris : les hooks pre-commit utilisent la version installée).
- Dependabot (`.github/dependabot.yml`) propose chaque mois une PR groupée
  pour les versions mineures et correctives des dépendances Python et des
  actions GitHub, et une PR séparée par version majeure. La CI valide chaque
  PR ; relire le changelog du paquet avant de merger une version majeure.
- Messages de commit à l'impératif, décrivant le pourquoi.

## Publier une version

1. Mettre à jour `version` dans `pyproject.toml` et `CHANGELOG.md`.
2. Fusionner sur `main` une fois la CI verte.
3. Taguer : `git tag vX.Y.Z && git push origin vX.Y.Z`.
