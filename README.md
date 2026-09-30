# Poseidon

[![CI](https://github.com/Ryckmat/Poseidon/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryckmat/Poseidon/actions/workflows/ci.yml)

Analyse de séances de rameur à partir de fichiers TCX : puissance, cadence,
allure au 500 m, distance par coup, fréquence cardiaque, segments d'effort
stable, FTP / NP / TSS, tendances hebdomadaires et records.

```
 fichiers TCX            PostgreSQL                    Streamlit
 ───────────  ingest ─> raw_files, sessions,  <─ lit ─ dashboard
 (1 ou N)                trackpoints                   (séance, progression,
                  analyze ─> segments, régressions,     analyse avancée,
                             indicateurs de séance      exports CSV / PDF)
```

## Fonctionnalités

- **Import** d'un fichier, de plusieurs fichiers ou d'un dossier : une séance
  coupée en plusieurs enregistrements est fusionnée en timeline continue, avec
  lissage des chutes de puissance aux jonctions. Réimport détecté.
- **Analyse** : filtrage des pics de puissance, segments stables, régressions
  puissance/cadence et puissance/vitesse, NP, FTP estimée, TSS (FTP de
  référence optionnelle). Relançable sans doublon.
- **Dashboard** : indicateurs clés avec écart vs une séance de comparaison,
  séries temporelles, zoom sur segment, distributions, zones de puissance,
  meilleurs efforts, progression hebdomadaire, records personnels, export CSV
  et rapport PDF, interface en français ou en anglais, import de fichiers
  optionnel.
- **Automatisation** : un `.tcx` poussé dans `data/` est importé et analysé par
  GitHub Actions.

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
make install                 # paquet + outils de dev + hooks pre-commit
make db-up                   # PostgreSQL local (docker compose)
cp .env.example .env         # DATABASE_URL déjà adapté à la base locale
poseidon init-db
poseidon ingest data/seance.tcx --analyze
make run                     # dashboard sur http://localhost:8501
```

Commandes principales :

| Commande                                   | Rôle                                        |
| ------------------------------------------ | ------------------------------------------- |
| `poseidon ingest FICHIER... [--analyze]`   | importe une séance (fichiers ou dossier)    |
| `poseidon analyze ID... \| --all`          | (re)calcule l'analyse                       |
| `poseidon list`                            | liste les séances                           |
| `poseidon delete ID`                       | supprime une séance et ses données          |

## Documentation

| Page                                        | Contenu                                           |
| ------------------------------------------- | ------------------------------------------------- |
| [Utilisation](docs/utilisation.md)          | CLI, dashboard, configuration, dossier `data/`    |
| [Métriques](docs/metriques.md)              | définition de chaque indicateur et de la fusion   |
| [Architecture](docs/architecture.md)        | modules, flux, modèle de données                  |
| [Déploiement](docs/deploiement.md)          | base hébergée, Streamlit Cloud, GitHub Actions    |
| [Développement](docs/developpement.md)      | environnement, tests, conventions                 |
| [Changelog](CHANGELOG.md)                   | historique des versions                           |
