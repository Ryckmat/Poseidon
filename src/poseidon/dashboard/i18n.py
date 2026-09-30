"""Libellés du dashboard (en / fr)."""

LANGUAGES = {"en": "English", "fr": "Français"}

_STRINGS = {
    # Structure
    "title": ("Poseidon: session overview", "Poseidon : vue de séance"),
    "controls": ("Controls", "Contrôles"),
    "none": ("None", "Aucune"),
    "refresh": ("Refresh data", "Rafraîchir les données"),
    "db_error": (
        "Database unavailable: {error}",
        "Base de données indisponible : {error}",
    ),
    "no_sessions": (
        "No sessions yet. Import a TCX file to get started.",
        "Aucune séance. Importez un fichier TCX pour commencer.",
    ),
    "no_trackpoints": ("Session has no trackpoints.", "La séance n'a pas de points."),
    # Sélection
    "sessions_shown": ("Sessions listed", "Séances listées"),
    "primary_session": ("Session", "Séance"),
    "compare_to_optional": ("Compare to (optional)", "Comparer avec (optionnel)"),
    "self_compare_warning": (
        "Comparison session is the same as primary; ignored.",
        "La séance de comparaison est la même que la principale ; ignorée.",
    ),
    # Paramètres
    "stable_params": ("Analysis parameters", "Paramètres d'analyse"),
    "power_threshold": ("Power max threshold (W)", "Seuil max puissance (W)"),
    "min_stable_power": ("Min power, stable segment (W)", "Puissance min stable (W)"),
    "rolling_std_window": ("Rolling std window (s)", "Fenêtre écart-type (s)"),
    "std_threshold": ("Std threshold for stability", "Seuil écart-type stabilité"),
    "min_segment_duration": ("Min segment duration (s)", "Durée min segment (s)"),
    "reference_ftp": (
        "Reference FTP (W, 0 = estimated)",
        "FTP de référence (W, 0 = estimée)",
    ),
    "bucket_seconds": ("Downsample bucket (s)", "Agrégation (s)"),
    "preset_select": ("Saved presets", "Presets enregistrés"),
    "preset_name": ("Preset name", "Nom du preset"),
    "preset_save": ("Save preset", "Sauvegarder preset"),
    "reset_params": ("Reset parameters", "Réinitialiser"),
    # Import
    "upload": ("Import sessions", "Importer des séances"),
    "upload_files": ("TCX files", "Fichiers TCX"),
    "upload_merge": (
        "Merge selected files into one session",
        "Fusionner les fichiers en une seule séance",
    ),
    "upload_name": ("Session name (optional)", "Nom de la séance (optionnel)"),
    "upload_run": ("Import and analyze", "Importer et analyser"),
    "upload_done": ("Imported: {name}", "Importé : {name}"),
    "upload_skipped": ("Already imported: {name}", "Déjà importé : {name}"),
    "upload_failed": (
        "Import failed for {name}: {error}",
        "Échec de l'import de {name} : {error}",
    ),
    # Indicateurs
    "duration": ("Duration", "Durée"),
    "distance": ("Distance (km)", "Distance (km)"),
    "avg_split": ("Avg split (/500m)", "Allure moy. (/500m)"),
    "avg_speed": ("Avg speed (km/h)", "Vitesse moy. (km/h)"),
    "avg_power": ("Avg power (W)", "Puissance moy. (W)"),
    "avg_cadence": ("Avg stroke rate (spm)", "Cadence moy. (cps/min)"),
    "avg_dps": ("Distance per stroke (m)", "Distance par coup (m)"),
    "avg_hr": ("Avg heart rate (bpm)", "FC moy. (bpm)"),
    "elevation_gain": ("Elevation gain (m)", "Dénivelé (m)"),
    "ftp_est": ("FTP est. (W)", "FTP estimée (W)"),
    "normalized_power": ("Normalized power (W)", "Puissance normalisée (W)"),
    "tss": ("TSS", "TSS"),
    "tooltip_ftp": (
        "95% of the best 20-minute average power (sessions of 20 min or more)",
        "95 % de la meilleure puissance moyenne sur 20 min (séances de 20 min et plus)",
    ),
    "tooltip_np": (
        "30 s rolling average, raised to the 4th power, averaged, 4th root",
        "Moyenne glissante 30 s, puissance 4, moyenne, racine 4",
    ),
    "tooltip_tss": (
        "Training Stress Score, based on reference FTP when set, else estimated FTP",
        "Score de charge, calculé avec la FTP de référence si définie, sinon estimée",
    ),
    "filter_info": (
        "Power filter: threshold {threshold:.0f} W, "
        "{removed} point(s) removed ({percent:.1f}%)",
        "Filtre de puissance : seuil {threshold:.0f} W, "
        "{removed} point(s) écarté(s) ({percent:.1f} %)",
    ),
    # Onglets
    "tab_session": ("Session", "Séance"),
    "tab_progression": ("Progression", "Progression"),
    "tab_advanced": ("Advanced", "Avancé"),
    # Graphiques
    "time_series": ("Time series", "Séries temporelles"),
    "elapsed_axis": ("Elapsed time (hh:mm:ss)", "Temps écoulé (hh:mm:ss)"),
    "power_over_time": ("Power", "Puissance"),
    "cadence_over_time": ("Stroke rate", "Cadence"),
    "speed_over_time": ("Speed", "Vitesse"),
    "split_over_time": ("Split /500m", "Allure /500m"),
    "hr_over_time": ("Heart rate", "Fréquence cardiaque"),
    "power_raw": ("Power raw", "Puissance brute"),
    "power_filtered": ("Power filtered", "Puissance filtrée"),
    "primary": ("Primary", "Principale"),
    "compare": ("Compare", "Comparaison"),
    "stable": ("Stable", "Stable"),
    "select_segment": ("Zoom on segment", "Zoom sur segment"),
    "stable_segments": ("Stable segments", "Segments stables"),
    "no_stable": (
        "No stable segment with current parameters.",
        "Aucun segment stable avec ces paramètres.",
    ),
    "start": ("Start", "Début"),
    "distributions": ("Distributions", "Distributions"),
    "correlations": ("Correlations and regression", "Corrélations et régressions"),
    "power_vs_cadence": ("Power vs stroke rate", "Puissance vs cadence"),
    "power_vs_speed": ("Power vs speed", "Puissance vs vitesse"),
    "fit": ("Fit", "Régression"),
    "power_w": ("Power (W)", "Puissance (W)"),
    "cadence_spm": ("Stroke rate (spm)", "Cadence (cps/min)"),
    "speed_kmh": ("Speed (km/h)", "Vitesse (km/h)"),
    "split_s": ("Split (s/500m)", "Allure (s/500m)"),
    "hr_bpm": ("Heart rate (bpm)", "FC (bpm)"),
    "dps_m": ("Distance per stroke (m)", "Distance par coup (m)"),
    # Export
    "export": ("Export", "Export"),
    "cleaned_trackpoints": ("Download cleaned CSV", "Télécharger le CSV nettoyé"),
    "full_export_csv": ("Download full CSV", "Télécharger le CSV complet"),
    "build_pdf": ("Build PDF report", "Générer le rapport PDF"),
    "download_pdf": ("Download PDF report", "Télécharger le rapport PDF"),
    # Progression
    "weekly_trends": ("Weekly trends", "Tendances hebdomadaires"),
    "week": ("Week", "Semaine"),
    "not_analyzed": (
        "{count} session(s) not analyzed yet: run `poseidon analyze --all`.",
        "{count} séance(s) non analysée(s) : lancer `poseidon analyze --all`.",
    ),
    "no_progression": ("No data for progression.", "Pas de données de progression."),
    "weekly_volume": ("Weekly volume (km)", "Volume hebdomadaire (km)"),
    "weekly_sessions": ("Sessions per week", "Séances par semaine"),
    "ftp_trend": ("FTP trend", "Tendance FTP"),
    "np_trend": ("Normalized power trend", "Tendance puissance normalisée"),
    "training_load": ("Training load (TSS)", "Charge d'entraînement (TSS)"),
    "records": ("Personal records", "Records personnels"),
    "record": ("Record", "Record"),
    "value": ("Value", "Valeur"),
    "date": ("Date", "Date"),
    "rec_distance": ("Longest distance (km)", "Plus longue distance (km)"),
    "rec_duration": ("Longest session", "Plus longue séance"),
    "rec_avg_power": ("Best average power (W)", "Meilleure puissance moyenne (W)"),
    "rec_np": ("Best normalized power (W)", "Meilleure puissance normalisée (W)"),
    "rec_split": ("Best average split (/500m)", "Meilleure allure moyenne (/500m)"),
    "rec_ftp": ("Best estimated FTP (W)", "Meilleure FTP estimée (W)"),
    # Avancé
    "descriptive_stats": ("Descriptive statistics", "Statistiques descriptives"),
    "mean": ("Mean", "Moyenne"),
    "median": ("Median", "Médiane"),
    "min": ("Min", "Min"),
    "max": ("Max", "Max"),
    "std": ("Std", "Écart-type"),
    "boxplots": ("Dispersion", "Dispersion"),
    "power_zones": ("Power zones", "Zones de puissance"),
    "zone": ("Zone", "Zone"),
    "name": ("Name", "Nom"),
    "from": ("From", "De"),
    "to": ("To", "À"),
    "time_in_zone": ("Time in zone", "Temps dans la zone"),
    "best_efforts": (
        "Best average power (5s, 1min, 5min, 20min)",
        "Meilleurs efforts moyens (5s, 1min, 5min, 20min)",
    ),
    "interval": ("Interval", "Intervalle"),
    "longest_streak": (
        "Longest streak between {low}W and {high}W",
        "Plus longue séquence entre {low}W et {high}W",
    ),
    "max_streak": ("Max streak: {duration}", "Séquence max : {duration}"),
    # PDF
    "key_metrics": ("Key metrics", "Indicateurs clés"),
    "chart_unavailable": ("chart unavailable", "graphique indisponible"),
}


class Translator:
    def __init__(self, lang: str):
        self.lang = lang if lang in LANGUAGES else "en"
        self._idx = 0 if self.lang == "en" else 1

    def __call__(self, key: str, **kwargs) -> str:
        text = _STRINGS[key][self._idx]
        return text.format(**kwargs) if kwargs else text


def all_keys() -> list:
    return list(_STRINGS)
