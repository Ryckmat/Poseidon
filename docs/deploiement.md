# Déploiement

Trois briques indépendantes partagent la même base : la CLI (poste local ou
GitHub Actions), le dashboard et PostgreSQL.

## Base PostgreSQL

N'importe quel PostgreSQL convient (local, Supabase, Neon...). La connexion
passe par le pilote psycopg 3 ; les URL `postgres://` et
`postgresql+psycopg2://` sont acceptées et converties. Une fois `DATABASE_URL`
défini :

```bash
poseidon init-db
```

À relancer après chaque mise à jour : la commande ajoute les tables et index
manquants sans toucher aux données.

Après une mise à jour qui change les calculs, recalculer l'historique :

```bash
poseidon analyze --all
```

Sans accès direct à la base, le workflow `Maintenance base` fait les deux
depuis GitHub : onglet **Actions > Maintenance base > Run workflow**. La
liste des séances récentes est affichée dans le résumé de l'exécution.

## Dashboard sur Streamlit Community Cloud

1. Créer l'application depuis le dépôt, fichier principal
   `src/poseidon/dashboard/app.py`. `requirements.txt` installe le paquet.
2. Dans **Settings > Secrets**, déclarer les variables au premier niveau (elles
   sont exposées comme variables d'environnement) :

   ```toml
   DATABASE_URL = "postgresql://..."
   FTP_W = "180"
   ```

3. Laisser `POSEIDON_ENABLE_UPLOAD` absent si l'application est publique :
   l'import ouvrirait l'écriture en base à tout visiteur.

Le graphique du rapport PDF est rendu par `kaleido`, qui pilote un navigateur
Chrome ou Chromium installé sur la machine (emplacements standards, ou chemin
donné par la variable `BROWSER_PATH`). Sans navigateur, le PDF est produit
sans le graphique et l'incident est journalisé. Sur un serveur Debian ou
Ubuntu : `apt install chromium` ; sinon `plotly_get_chrome` télécharge Chrome.

## Dashboard sur un serveur

```bash
pip install .
DATABASE_URL=... streamlit run src/poseidon/dashboard/app.py --server.port 8501
```

Placer un reverse proxy avec authentification devant si l'import est activé.

## GitHub Actions

| Workflow          | Déclencheur                         | Rôle                              |
| ----------------- | ----------------------------------- | --------------------------------- |
| `CI`              | push sur `main`, pull request       | lint, tests SQLite et PostgreSQL  |
| `Process new TCX` | push de `.tcx` dans `data/`, manuel | import et analyse des séances     |
| `Maintenance base`| manuel                              | `init-db` puis, en option, `analyze --all` |

Configuration du dépôt pour `Process new TCX` et `Maintenance base`
(**Settings > Secrets and variables > Actions**) :

| Type     | Nom                                              | Obligatoire |
| -------- | ------------------------------------------------ | ----------- |
| Secret   | `DATABASE_URL`                                   | oui         |
| Variable | `FTP_W`, `MAX_POWER`, `MIN_STABLE_POWER`, `STABLE_WINDOW_S`, `STABLE_STD_THRESHOLD`, `MIN_STABLE_DURATION_S` | non, défauts sinon |

Chaque push est traité par sa propre exécution ; les séances déjà présentes
sont ignorées, ce qui permet de relancer une exécution sans risque.
