"""Data loading for the VoltGrid scikit-learn masterclass.

Every loader returns a plain pandas DataFrame. Nothing here does any
modelling - that is the student's job.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# repo_root/voltgrid/loading.py  ->  repo_root/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------- feature sets
NUMERIC_FEATURES = [
    "station_age_years",
    "power_kw",
    "num_bays",
    "hour",
    "day_of_week",
    "is_weekend",
    "ambient_temp_c",
    "grid_load_index",
    "start_soc_pct",
    "battery_capacity_kwh",
]

CATEGORICAL_FEATURES = [
    "connector_type",
    "zone",
    "vehicle_segment",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"Run  python make_voltgrid_ml.py  from the repo root to build it."
        )
    return path


def load_starter(n: int | None = None, random_state: int = 11) -> pd.DataFrame:
    """Small, clean, model-ready table (2,000 rows). Chapters 1-2 start here.

    No missing values, no messy text, no date parsing. The point of chapters
    1-2 is the estimator API, not data cleaning.

    Why 2,000 and not 400: at 400 rows there are only ~27 failures, and a
    single-feature model scores BELOW chance purely from sampling noise.
    A teaching set has to be large enough to tell the truth.
    """
    df = pd.read_csv(_require(DATA_DIR / "ml_starter_2000.csv"))
    if n is not None:
        df = df.sample(n, random_state=random_state).reset_index(drop=True)
    return df


def load_ml_sessions() -> pd.DataFrame:
    """Full clean modelling table (45,000 rows)."""
    return pd.read_csv(_require(DATA_DIR / "ml_sessions.csv"))


def load_sessions_raw() -> pd.DataFrame:
    """The MESSY operational table.

    Deliberately contains unparseable timestamps, money stored as text,
    missing readings and an `error_code` column that leaks the target.
    Used in chapters 15 and 17, and by the pandas masterclass.
    """
    return pd.read_csv(_require(DATA_DIR / "sessions.csv"))


def load_stations() -> pd.DataFrame:
    """Station master table (18 rows)."""
    return pd.read_csv(_require(DATA_DIR / "stations.csv"))


def station_profiles() -> pd.DataFrame:
    """One row per station, aggregated from raw sessions.

    This is the clustering input for chapter 15. The true archetype label is
    deliberately NOT included - students recover it.
    """
    ses = load_sessions_raw()
    ses["ts"] = pd.to_datetime(ses["started_at"], errors="coerce")
    ses = ses.dropna(subset=["ts"]).copy()
    ses["hour"] = ses["ts"].dt.hour

    prof = ses.groupby("station_id").agg(
        sessions=("session_id", "count"),
        med_duration_mins=("duration_mins", "median"),
        med_energy_kwh=("energy_kwh", "median"),
        night_share=("hour", lambda h: float(h.isin([22, 23, 0, 1, 2, 3, 4, 5]).mean())),
        daytime_share=("hour", lambda h: float(h.between(9, 17).mean())),
    )
    meta = load_stations().set_index("station_id")[["power_kw", "num_bays"]]
    return prof.join(meta)


def load_anomaly_key() -> set[str]:
    """INSTRUCTOR ONLY - ground-truth anomalous session ids."""
    p = DATA_DIR.parent / "instructor/data/_anomaly_ANSWER_KEY.csv"
    return set(pd.read_csv(_require(p))["session_id"])


def load_archetype_key() -> pd.Series:
    """INSTRUCTOR ONLY - ground-truth station archetypes."""
    p = DATA_DIR.parent / "instructor/data/_station_archetype_ANSWER_KEY.csv"
    return pd.read_csv(_require(p)).set_index("station_id")["archetype_TRUTH"]
