"""
VoltGrid - shared helpers for the scikit-learn masterclass.

Import these in every chapter so lessons stay short and students never
re-type boilerplate on stream.
"""
from .loading import (
    DATA_DIR,
    load_starter,
    load_ml_sessions,
    load_sessions_raw,
    load_stations,
    station_profiles,
    load_anomaly_key,
    load_archetype_key,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
)
from .features import make_preprocessor, split_xy
from .plotting import (
    plot_decision_boundary,
    plot_confusion,
    plot_roc,
    plot_residuals,
    plot_permutation_importance,
    use_style,
    STYLE,
)

__all__ = [
    "DATA_DIR",
    "load_starter",
    "load_ml_sessions",
    "load_sessions_raw",
    "load_stations",
    "station_profiles",
    "load_anomaly_key",
    "load_archetype_key",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_FEATURES",
    "make_preprocessor",
    "split_xy",
    "plot_decision_boundary",
    "plot_confusion",
    "plot_roc",
    "plot_residuals",
    "plot_permutation_importance",
    "use_style",
    "STYLE",
]

__version__ = "1.0.0"
