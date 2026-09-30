# Utilisation

## Configuration

Variables lues dans l'environnement ou dans un fichier `.env` à la racine
(modèle : `.env.example`).

| Variable                 | Défaut  | Rôle                                                        |
| ------------------------ | ------- | ----------------------------------------------------------- |
| `DATABASE_URL`           | aucun   | connexion PostgreSQL, obligatoire                           |
| `MAX_POWER`              | 250     | puissance (W) au-delà de laquelle un point est écarté       |
| `MIN_STABLE_POWER`       | 50      | puissance minimale d'un segment stable                      |
| `STABLE_WINDOW_S`        | 30      | fenêtre de l'écart-type glissant (s)                        |
| `STABLE_STD_THRESHOLD`   | 5       | écart-type maximal pour être « stable » (W)                 |
| `MIN_STABLE_DURATION_S`  | 60      | durée minimale d'un segment stable (s)                      |
| `FTP_W`                  | vide    | FTP de référence (W) pour le TSS ; vide = FTP estimée       |
| `POSEIDON_ENABLE_UPLOAD` | `false` | active l'import de fichiers depuis le dashboard             |

Les paramètres d'analyse servent de valeurs par défaut au dashboard, où ils
restent modifiables en direct sans toucher à la base.

## Ligne de commande

Toutes les commandes écrivent leurs messages sur la sortie d'erreur et
renvoient un code non nul en cas d'échec. `-v` active les logs détaillés.

### `poseidon init-db`

Crée les tables et index manquants. Sans effet si tout existe déjà : à
relancer après une mise à jour qui ajoute un index.

### `poseidon ingest CHEMIN... [options]`

Importe une séance. `CHEMIN` est un fichier `.tcx` ou un dossier (tous ses
`.tcx`, triés par nom). Plusieurs fichiers donnent une seule séance fusionnée
(voir [Métriques, fusion](metriques.md#fusion-de-plusieurs-fichiers)).

| Option                          | Effet                                                      |
| ------------------------------- | ---------------------------------------------------------- |
| `--name NOM`                    | nom de la séance (défaut : nom du fichier)                 |
| `--analyze`                     | lance l'analyse juste après l'import                       |
| `--skip-existing`               | séance déjà importée : avertissement au lieu d'une erreur  |
| `--no-fix-boundary-spikes`      | désactive le lissage aux jonctions                         |
| `--spike-window-after-s S`      | durée lissée après une jonction (défaut 3)                 |
| `--spike-seek-next-valid-s S`   | recherche d'une puissance valide après jonction (défaut 6) |
| `--spike-min-valid-w W`         | puissance minimale jugée valide (défaut 20)                |

La sortie standard contient uniquement l'identifiant de la séance créée, ce
qui permet `ID=$(poseidon ingest seance.tcx)`.

Une séance est identifiée par son nom : réimporter le même fichier est refusé.
Pour remplacer une séance, la supprimer puis la réimporter.

### `poseidon analyze ID... | --all`

Calcule, pour chaque séance : dérivés par point (vitesse, allure, puissance
filtrée), segments stables, régressions et indicateurs de séance. Les
résultats précédents sont remplacés. `--all` traite toutes les séances, utile
après un changement de paramètres ou de version.

### `poseidon list [--limit N]`

Liste les séances récentes : id, date, durée, distance, état d'analyse, nom.

### `poseidon delete ID [--yes]`

Supprime la séance, ses points, ses résultats et le fichier source associé.
Demande confirmation sauf avec `--yes`.

## Dossier `data/` et import automatique

Pousser des fichiers dans `data/` sur `main` déclenche le workflow
`Process new TCX` :

```
data/
├── 2025-03-01.tcx          -> une séance
└── 2025-03-08-fractionne/  -> une séance fusionnée
    ├── part1.tcx
    └── part2.tcx
```

Seuls les fichiers ajoutés ou modifiés par le push sont traités ; les séances
déjà présentes sont ignorées. Un lancement manuel (onglet Actions) traite tout
le dossier. Configuration : voir [Déploiement](deploiement.md#github-actions).

## Dashboard

```bash
streamlit run src/poseidon/dashboard/app.py
```

**Barre latérale** : langue, rafraîchissement des données (cache de
5 minutes), import de fichiers si activé, choix de la séance et d'une séance
de comparaison, paramètres d'analyse avec presets, pas d'agrégation.

**En-tête** : indicateurs clés. Avec une séance de comparaison, chaque
indicateur affiche l'écart (en vert quand c'est mieux, l'allure étant
meilleure quand elle baisse).

| Onglet      | Contenu                                                                 |
| ----------- | ----------------------------------------------------------------------- |
| Séance      | puissance brute / filtrée et segments stables, tableau et zoom des segments, allure, cadence, fréquence cardiaque, distributions, régressions, exports CSV et PDF |
| Progression | volume, nombre de séances, TSS et tendance FTP / NP par semaine ; records personnels |
| Avancé      | statistiques descriptives, dispersion, temps par zone de puissance, meilleurs efforts, plus longue séquence entre 100 et 250 W |

La progression lit les indicateurs enregistrés par `poseidon analyze` : une
séance importée sans analyse y est signalée.

**Pas d'agrégation** : les points sont moyennés par tranches de N secondes
côté base pour limiter le volume transféré. Les calculs de l'affichage
(segments, NP...) portent sur ces points agrégés et peuvent donc différer
légèrement des valeurs calculées par `poseidon analyze` sur les points bruts.

**Import depuis le navigateur** (`POSEIDON_ENABLE_UPLOAD=true`) : un ou
plusieurs `.tcx`, importés séparément ou fusionnés, puis analysés. N'activer
que si l'accès au dashboard est protégé : toute personne qui y accède peut
alors écrire en base.
