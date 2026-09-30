# Métriques

Toutes les formules sont dans `src/poseidon/processing/metrics.py`, partagées
par `poseidon analyze` et le dashboard. Notations : `P` puissance (W),
`v` vitesse (m/s), `c` cadence (coups/min).

## Préparation des points

| Colonne                 | Calcul                                                             |
| ----------------------- | ------------------------------------------------------------------ |
| `elapsed_time_s`        | temps écoulé depuis le premier point                               |
| `speed_kmh`             | Δdistance / Δtemps × 3,6 ; vide si Δtemps ≤ 0 ou distance qui recule |
| `pace_min_per_km`       | 1000 / v / 60                                                      |
| `split_500m_s`          | 500 / v, allure au 500 m en secondes                               |
| `distance_per_stroke_m` | v × 60 / c                                                         |
| `elevation_diff`        | Δaltitude                                                          |
| `power_filtered`        | P si P ≤ `MAX_POWER`, sinon vide                                   |
| `power_std`             | écart-type glissant de `power_filtered` sur `STABLE_WINDOW_S`      |

Allure et distance par coup ne sont calculées que si le rameur avance
(v > 0,01 km/h) et, pour la seconde, si la cadence est positive. Dans le
dashboard, une vitesse déjà calculée en base est prioritaire sur celle
recalculée depuis les points agrégés.

## Indicateurs de séance

| Indicateur            | Définition                                                         |
| --------------------- | ------------------------------------------------------------------ |
| Durée                 | temps écoulé entre le premier et le dernier point                  |
| Distance              | distance cumulée maximale                                          |
| Allure moyenne        | 500 / vitesse moyenne                                              |
| Puissance moyenne     | moyenne de `power_filtered`                                        |
| Distance par coup     | moyenne de `distance_per_stroke_m`                                 |
| NP                    | moyenne glissante 30 s de `power_filtered`, puissance 4, moyenne, racine 4 |
| FTP estimée           | 0,95 × meilleure moyenne de `power_filtered` sur 20 min            |
| TSS                   | durée × NP × IF / (FTP × 3600) × 100, avec IF = NP / FTP           |

Le TSS utilise la FTP de référence (`FTP_W` ou champ du dashboard) si elle est
définie, sinon la FTP estimée sur la séance elle-même. Dans ce second cas le
TSS mesure l'effort relatif à la meilleure portion de la séance, pas à la
forme réelle : définir `FTP_W` donne un TSS comparable d'une séance à l'autre.

**Fenêtres complètes uniquement.** Les moyennes glissantes (NP, FTP,
meilleurs efforts) ne retiennent que les fenêtres couvrant toute la durée :
les premiers points d'une séance, qui n'ont pas encore l'historique voulu, sont
exclus. Sans cette règle, quelques secondes de pic au démarrage passeraient
pour une « meilleure moyenne 20 min ». Conséquence : une séance de moins de
20 minutes n'a pas de FTP estimée, ni de TSS si aucune FTP de référence n'est
définie.

## Segments stables

Plage continue de points où `power_filtered ≥ MIN_STABLE_POWER` et
`power_std ≤ STABLE_STD_THRESHOLD`, retenue si elle dure au moins
`MIN_STABLE_DURATION_S`. Pour chaque segment : durée, puissance moyenne et
écart-type, cadence et vitesse moyennes.

## Régressions

Droite des moindres carrés de la cadence puis de la vitesse en fonction de
`power_filtered`, avec pente, ordonnée à l'origine et R². Non calculée s'il y
a moins de deux points ou une puissance constante.

## Analyse avancée

| Indicateur          | Définition                                                         |
| ------------------- | ------------------------------------------------------------------ |
| Zones de puissance  | temps passé dans chaque zone (bornes en W ci-dessous)              |
| Meilleurs efforts   | meilleure moyenne de `power_filtered` sur 5 s, 1, 5 et 20 min (vide si séance plus courte) |
| Plus longue séquence| plus longue durée continue avec 100 W ≤ P < 250 W                  |

| Zone | Nom              | De (W) | À (W) |
| ---- | ---------------- | ------ | ----- |
| Z1   | Récup. active    | 0      | 34    |
| Z2   | Endurance        | 34     | 47    |
| Z3   | Tempo            | 47     | 56    |
| Z4   | Seuil            | 56     | 66    |
| Z5   | VO2max           | 66     | 75    |
| Z6   | Anaérobie        | 75     | 94    |
| Z7   | Neuromusculaire  | 94     | 250   |

Les bornes sont fixes (constante `POWER_ZONES`). Les temps sont estimés à
partir du pas d'échantillonnage médian.

## Fusion de plusieurs fichiers

Utilisée quand une séance est enregistrée en plusieurs morceaux
(`src/poseidon/ingest/merge.py`).

1. Les fichiers sont ordonnés par heure de début.
2. Chaque fichier est décalé pour commencer à l'instant où finit le
   précédent : la timeline résultante n'a pas de trou.
3. Les distances sont remises à zéro par fichier puis cumulées.
4. Un horodatage déjà présent n'est pas dupliqué : le premier point d'un
   fichier, qui tombe sur le dernier point du précédent, est écarté.
5. Lissage aux jonctions (option, actif par défaut) : les points des
   3 secondes suivant une jonction sont interpolés entre la dernière puissance
   avant la jonction et la première puissance ≥ 20 W trouvée dans les
   6 secondes. Sans puissance valide, la valeur d'avant est recopiée. Cela
   supprime les chutes artificielles vers 0 W au redémarrage de
   l'enregistrement.

Le détail de chaque fichier (nom, empreinte SHA-256, bornes, nombre de points,
distance) et les paramètres de lissage sont conservés dans
`raw_files.metadata`.
