# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Ajouté

- Workflow `Maintenance base`, lancé à la main depuis GitHub : `init-db` puis,
  en option, `analyze --all`.
- Dependabot : mises à jour mensuelles des dépendances Python et des actions
  GitHub.
- Licence MIT.

### Modifié

- Les hooks pre-commit utilisent le ruff installé par le projet (version
  unique dans `pyproject.toml`).
- Pilote PostgreSQL psycopg 3 à la place de psycopg2 (SQLAlchemy 2.1) ; les
  URL `postgres://` et `postgresql+psycopg2://` restent acceptées.
- plotly 7 et kaleido 1 : l'image du graphique du PDF nécessite désormais
  Chrome ou Chromium sur la machine.
- numpy 2.4, SQLAlchemy 2.1.

### Corrigé

- `poseidon analyze --all` sur une base vide réussit au lieu de renvoyer une
  erreur.

## [0.3.0] - 2026-09-30

### Ajouté

- Commande unique `poseidon` : `init-db`, `ingest`, `analyze`, `list`,
  `delete`.
- Import d'un dossier de fichiers, option `--analyze`, détection des séances
  déjà importées (`--skip-existing`).
- Métriques d'aviron : allure au 500 m, distance par coup ; fréquence
  cardiaque dans l'analyse et le dashboard.
- FTP de référence (`FTP_W`) pour un TSS comparable entre séances.
- Indicateurs de séance enregistrés en base par l'analyse (NP, FTP, TSS,
  vitesse et FC moyennes, dénivelé).
- Dashboard : écarts vs séance de comparaison, tableau des segments stables,
  graphiques allure et FC, progression hebdomadaire (volume, nombre de
  séances, TSS, FTP, NP) sur tout l'historique, records personnels, import de
  fichiers optionnel, bouton de rafraîchissement.
- Documentation (`docs/`), Makefile, PostgreSQL local (`docker-compose.yml`),
  pre-commit, tests d'intégration PostgreSQL et dashboard en CI.

### Modifié

- Relancer l'analyse remplace les résultats précédents au lieu de les
  dupliquer.
- Le workflow d'import traite tous les fichiers d'un push (plus seulement le
  dernier commit) et fusionne les sous-dossiers de `data/`.
- Le dashboard reprend les paramètres d'analyse de l'environnement.
- Vitesse inconnue (premier point, distance absente) laissée vide au lieu de
  0, ce qui ne fausse plus la vitesse moyenne.
- Dépendances mises à jour ; ruff remplace black, isort et flake8.
- Lecture TCX via `defusedxml`, erreurs explicites sur fichier invalide.
- Insertions et mises à jour de points en masse, index sur les tables.

### Corrigé

- FTP estimée, NP et meilleurs efforts calculés sur des fenêtres partielles en
  début de séance : un pic de quelques secondes au démarrage pouvait devenir
  la « meilleure moyenne 20 min ». Seules les fenêtres complètes comptent ;
  une séance de moins de 20 minutes n'a plus de FTP estimée. Relancer
  `poseidon analyze --all` pour recalculer l'historique.
- Distance de séance à « nan » quand le fichier n'a pas de distance.
- Erreur brute (trace Python) quand la base est inaccessible depuis le
  dashboard ou la CLI.
- Nouvelles séances invisibles jusqu'à une heure dans le dashboard (cache).

## [0.2.0] - 2026-09-30

### Modifié

- Restructuration en paquet `poseidon`, calculs partagés entre analyse et
  dashboard, dashboard découpé en modules.

### Corrigé

- Métadonnées de fichier perdues à l'import.
- Chaînage des distances quand les fichiers fusionnés ne sont pas dans
  l'ordre chronologique.
- Point en trop dans les statistiques de segment stable.
- Graphique jamais intégré au rapport PDF.
