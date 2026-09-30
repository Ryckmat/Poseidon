"""Libellés du dashboard (en / fr)."""

LANGUAGES = {"en": "🇬🇧 English", "fr": "🇫🇷 Français"}

_STRINGS = {
    "language": ("Language", "Langue"),
    "title": ("Poseidon — Session Overview", "Poseidon — Vue de séance"),
    "controls": ("Controls", "Contrôles"),
    "primary_session": ("Primary session", "Séance principale"),
    "compare_to_optional": ("Compare to (optional)", "Comparer avec (optionnel)"),
    "none": ("None", "Aucune"),
    "self_compare_warning": (
        "Comparison session is the same as primary; ignored.",
        "La séance de comparaison est la même que la principale ; ignorée.",
    ),
    "stable_params": (
        "Stable segment & filtering params",
        "Paramètres de segment stable et filtrage",
    ),
    "power_threshold": ("Power max threshold (filter)", "Seuil max puissance (filtre)"),
    "min_stable_power": (
        "Min power for stable segment",
        "Puissance min pour segment stable",
    ),
    "rolling_std_window": ("Rolling std window (s)", "Fenêtre écart-type roulante (s)"),
    "std_threshold": ("Std threshold for stability", "Seuil écart-type pour stabilité"),
    "min_segment_duration": ("Min segment duration (s)", "Durée min du segment (s)"),
    "bucket_seconds": ("Downsample bucket (s)", "Agrégation (s)"),
    "preset_select": ("Saved presets", "Presets enregistrés"),
    "preset_name": ("Preset name", "Nom du preset"),
    "preset_save": ("Save current preset", "Sauvegarder preset"),
    "reset_params": ("Reset filters", "Réinitialiser filtres"),
    "no_sessions": ("No sessions found in database.", "Aucune séance trouvée."),
    "no_trackpoints": ("Session has no trackpoints.", "La séance n'a pas de points."),
    "duration": ("Duration", "Durée"),
    "distance": ("Distance (km)", "Distance (km)"),
    "elevation_gain": ("Elevation Gain (m)", "Dénivelé (m)"),
    "avg_speed": ("Avg Speed (km/h)", "Vitesse moy. (km/h)"),
    "ftp_est": ("FTP Est.", "FTP estimé"),
    "normalized_power": ("Normalized Power", "Puissance normalisée"),
    "tss": ("TSS", "TSS"),
    "tooltip_ftp": (
        "Estimated from best 20-minute average × 0.95",
        "Estimé depuis la meilleure moyenne sur 20 minutes × 0.95",
    ),
    "tooltip_np": (
        "Normalized Power: 30s rolling average to the 4th power",
        "Puissance normalisée : moyenne roulante 30s à la puissance 4",
    ),
    "tooltip_tss": (
        "Training Stress Score approximate",
        "Score de charge d'entraînement approximatif",
    ),
    "power_filtering": ("Power filtering", "Filtrage de puissance"),
    "filter_info": (
        "threshold = {threshold:.1f} W → removed {removed} point(s) ({percent:.1f}%)",
        "seuil = {threshold:.1f} W → {removed} point(s) supprimé(s) ({percent:.1f}%)",
    ),
    "tab_session": ("Session", "Séance"),
    "tab_progression": ("Progression", "Progression"),
    "tab_advanced": ("Advanced", "Avancé"),
    "time_series": ("Time Series", "Séries temporelles"),
    "elapsed_axis": ("Elapsed time (hh:mm:ss)", "Temps écoulé (hh:mm:ss)"),
    "power_over_time": ("Power over Time", "Puissance dans le temps"),
    "cadence_over_time": ("Cadence over Time", "Cadence dans le temps"),
    "speed_over_time": ("Speed over Time", "Vitesse dans le temps"),
    "power_raw": ("Power raw", "Puissance brute"),
    "power_filtered": ("Power filtered", "Puissance filtrée"),
    "primary": ("Primary", "Principale"),
    "compare": ("Compare", "Comparaison"),
    "stable": ("Stable", "Stable"),
    "select_segment": ("Zoom on segment", "Zoom sur segment"),
    "distributions": ("Distributions", "Distributions"),
    "correlations": ("Correlations & Regression", "Corrélations & Régressions"),
    "power_vs_cadence": ("Power vs Cadence", "Puissance vs Cadence"),
    "power_vs_speed": ("Power vs Speed", "Puissance vs Vitesse"),
    "fit": ("Fit", "Régression"),
    "stable_segments": ("Stable segments", "Segments stables"),
    "export": ("Export", "Export"),
    "cleaned_trackpoints": (
        "Download cleaned trackpoints CSV",
        "Télécharger CSV nettoyé",
    ),
    "full_export_csv": (
        "Export all session data (CSV)",
        "Exporter toutes les données de séance (CSV)",
    ),
    "build_pdf": ("Build PDF report", "Générer le rapport PDF"),
    "download_pdf": ("Download PDF report", "Télécharger le rapport PDF"),
    "weekly_trends": ("Weekly Trends", "Tendances hebdo"),
    "trends_info": (
        "Computed on demand to avoid loading every session.",
        "Calcul déclenché manuellement pour éviter de charger toutes les séances.",
    ),
    "trends_run": (
        "Compute trends (may be slow)",
        "Calculer tendances (peut être long)",
    ),
    "no_progression": ("No data for progression.", "Pas de données de progression."),
    "ftp_trend": ("FTP Trend", "Tendance FTP"),
    "np_trend": ("Normalized Power Trend", "Tendance NP"),
    "training_load": ("Training Load (TSS)", "Charge d'entraînement (TSS)"),
    "descriptive_stats": ("Descriptive Statistics", "Statistiques descriptives"),
    "mean": ("Mean", "Moyenne"),
    "median": ("Median", "Médiane"),
    "min": ("Min", "Min"),
    "max": ("Max", "Max"),
    "std": ("Std", "Écart-type"),
    "boxplots": ("Dispersion (Boxplots)", "Dispersion (Boxplots)"),
    "power_w": ("Power (W)", "Puissance (W)"),
    "cadence_rpm": ("Cadence (rpm)", "Cadence (rpm)"),
    "speed_kmh": ("Speed (km/h)", "Vitesse (km/h)"),
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
