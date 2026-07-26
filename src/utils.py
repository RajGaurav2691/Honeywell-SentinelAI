"""
utils.py

Honeywell SentinelAI project.

A single, generic collection of reusable helper utilities shared by
generate_data.py, train_model.py, and dashboard.py. This module intentionally
contains NO business logic specific to any one script (no entity profiles,
no attack injection, no chart-specific rendering) — only small, general-
purpose building blocks: logging, path handling, model/CSV I/O, risk-level
mapping, time formatting, validation, dataset statistics, Plotly theming,
and number formatting.

Designed to be imported either explicitly:
    from utils import get_logger, load_csv, risk_level

or via star import:
    from utils import *
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Public API (enables safe `from utils import *`)
# --------------------------------------------------------------------------- #

__all__ = [
    # Logger utilities
    "get_logger",
    # Path utilities
    "ensure_directory",
    # Model utilities
    "save_model",
    "load_model",
    "save_pickle",
    "load_pickle",
    # CSV utilities
    "load_csv",
    "save_csv",
    # Risk utilities
    "risk_level",
    # Time utilities
    "current_timestamp",
    "format_timestamp",
    "days_between",
    # Validation utilities
    "check_required_columns",
    "validate_dataframe",
    "validate_model_files",
    # Statistics utilities
    "dataset_summary",
    # Visualization utilities
    "apply_plot_theme",
    "show_streamlit_error",
    # Formatting utilities
    "format_number",
    "format_percentage",
    "format_duration",
    # Constants
    "RISK_COLORS",
    "THEME_COLORS",
    "DEFAULT_PLOT_LAYOUT",
]


# --------------------------------------------------------------------------- #
# 11. Constants
# --------------------------------------------------------------------------- #

#: Colors associated with each risk level, used consistently across charts
#: and tables throughout the project.
RISK_COLORS: Dict[str, str] = {
    "Low": "#22C55E",
    "Medium": "#FACC15",
    "High": "#FB923C",
    "Critical": "#EF4444",
}

#: Core brand / theme colors for the SentinelAI dark-blue SOC aesthetic.
THEME_COLORS: Dict[str, str] = {
    "navy": "#0B1E3F",
    "navy_deep": "#071431",
    "blue": "#1E56A0",
    "accent": "#3AA0FF",
    "card": "#FFFFFF",
    "text_light": "#E6EEFA",
}

#: Default Plotly layout options applied to every figure in the project.
DEFAULT_PLOT_LAYOUT: Dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(255,255,255,0.04)",
    "font": {"color": THEME_COLORS["text_light"], "family": "Segoe UI, sans-serif"},
    "legend": {
        "bgcolor": "rgba(255,255,255,0.06)",
        "bordercolor": THEME_COLORS["accent"],
        "borderwidth": 1,
    },
}

# Risk level thresholds (upper bound, inclusive) mapped to their labels, in
# ascending order of severity. Used by risk_level().
_RISK_THRESHOLDS: List[tuple] = [
    (24, "Low"),
    (49, "Medium"),
    (74, "High"),
]
_RISK_MAX_LABEL = "Critical"


# --------------------------------------------------------------------------- #
# 1. Logger utilities
# --------------------------------------------------------------------------- #

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a consistent, readable format.

    Safe to call multiple times with the same name: it will not attach
    duplicate handlers to the logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


_logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 2. Path utilities
# --------------------------------------------------------------------------- #

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating parent folders as needed.

    Args:
        path: Directory path, or a file path (in which case its parent
            directory is created).

    Returns:
        The resolved Path object for the directory that now exists.
    """
    directory = Path(path)

    # If the path looks like a file (has a suffix), ensure its parent exists.
    if directory.suffix:
        directory = directory.parent

    directory.mkdir(parents=True, exist_ok=True)
    return directory


# --------------------------------------------------------------------------- #
# 3. Model utilities
# --------------------------------------------------------------------------- #

def save_model(model: Any, path: Union[str, Path]) -> Path:
    """Save a trained model object to disk using joblib.

    Args:
        model: The fitted model or estimator to persist.
        path: Destination file path.

    Returns:
        The Path the model was saved to.
    """
    output_path = Path(path)
    ensure_directory(output_path)
    joblib.dump(model, output_path)
    _logger.info("Model saved to %s", output_path)
    return output_path


def load_model(path: Union[str, Path]) -> Any:
    """Load a previously saved model object from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        The deserialized model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def save_pickle(obj: Any, path: Union[str, Path]) -> Path:
    """Save any Python object to disk using joblib (generic pickle helper).

    Args:
        obj: Any picklable Python object (e.g. a dict of encoders, a scaler).
        path: Destination file path.

    Returns:
        The Path the object was saved to.
    """
    output_path = Path(path)
    ensure_directory(output_path)
    joblib.dump(obj, output_path)
    _logger.info("Object saved to %s", output_path)
    return output_path


def load_pickle(path: Union[str, Path]) -> Any:
    """Load any previously pickled Python object from disk.

    Args:
        path: Path to the saved object file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    object_path = Path(path)
    if not object_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {object_path}")
    return joblib.load(object_path)


# --------------------------------------------------------------------------- #
# 4. CSV utilities
# --------------------------------------------------------------------------- #

def load_csv(path: Union[str, Path], **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a dataframe, validating that it exists first.

    Args:
        path: Path to the CSV file.
        **read_csv_kwargs: Additional keyword arguments forwarded to
            pandas.read_csv (e.g. parse_dates, dtype).

    Returns:
        The loaded dataframe.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file exists but cannot be parsed as CSV.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        return pd.read_csv(csv_path, **read_csv_kwargs)
    except Exception as error:  # noqa: BLE001 - re-raised with clearer context
        raise ValueError(f"Failed to parse CSV file {csv_path}: {error}") from error


def save_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> Path:
    """Save a dataframe to CSV, creating parent folders as needed.

    Args:
        df: Dataframe to save.
        path: Destination file path.
        index: Whether to write the dataframe index. Defaults to False.

    Returns:
        The Path the CSV was saved to.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("save_csv expects a pandas DataFrame.")

    output_path = Path(path)
    ensure_directory(output_path)
    df.to_csv(output_path, index=index)
    _logger.info("CSV saved to %s (%d rows)", output_path, len(df))
    return output_path


# --------------------------------------------------------------------------- #
# 5. Risk utilities
# --------------------------------------------------------------------------- #

def risk_level(score: float) -> str:
    """Convert a numeric risk score (0-100) into a categorical risk level.

    Args:
        score: Risk score between 0 and 100.

    Returns:
        One of "Low", "Medium", "High", or "Critical".
    """
    clamped_score = max(0.0, min(100.0, float(score)))

    for upper_bound, label in _RISK_THRESHOLDS:
        if clamped_score <= upper_bound:
            return label

    return _RISK_MAX_LABEL


# --------------------------------------------------------------------------- #
# 6. Time utilities
# --------------------------------------------------------------------------- #

def current_timestamp() -> datetime:
    """Return the current local timestamp.

    Returns:
        A datetime object representing "now".
    """
    return datetime.now()


def format_timestamp(
    timestamp: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Format a datetime object as a string.

    Args:
        timestamp: The datetime to format. Defaults to the current time.
        fmt: The strftime format string to use.

    Returns:
        The formatted timestamp string.
    """
    moment = timestamp if timestamp is not None else current_timestamp()
    return moment.strftime(fmt)


def days_between(start: datetime, end: datetime) -> int:
    """Compute the whole number of days between two datetimes.

    Args:
        start: The earlier datetime.
        end: The later datetime.

    Returns:
        The absolute number of days between start and end.
    """
    return abs((end - start).days)


# --------------------------------------------------------------------------- #
# 7. Validation utilities
# --------------------------------------------------------------------------- #

def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Verify that a dataframe contains all required columns.

    Args:
        df: Dataframe to check.
        required_columns: List of column names that must be present.

    Returns:
        True if all required columns are present.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def validate_dataframe(df: pd.DataFrame, allow_empty: bool = False) -> bool:
    """Validate that an object is a usable, non-empty pandas DataFrame.

    Args:
        df: Object to validate.
        allow_empty: If False (default), an empty dataframe is rejected.

    Returns:
        True if the dataframe is valid.

    Raises:
        ValueError: If df is not a DataFrame, or is empty when not allowed.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame.")
    if not allow_empty and df.empty:
        raise ValueError("DataFrame is empty.")
    return True


def validate_model_files(paths: List[Union[str, Path]]) -> bool:
    """Validate that all given model/artifact files exist on disk.

    Args:
        paths: List of file paths that must exist.

    Returns:
        True if every file exists.

    Raises:
        FileNotFoundError: If any file in the list is missing.
    """
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model file(s): {missing}")
    return True


# --------------------------------------------------------------------------- #
# 8. Statistics utilities
# --------------------------------------------------------------------------- #

def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute a generic summary of a login-event style dataframe.

    Gracefully handles dataframes that lack risk_score / label / prediction
    columns by omitting the fields that cannot be computed.

    Args:
        df: Dataframe to summarize.

    Returns:
        Dict with rows, columns, null_count, duplicate_count, and,
        where available, average_risk, normal_count, and anomaly_count.
    """
    validate_dataframe(df, allow_empty=True)

    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
    }

    if "risk_score" in df.columns:
        summary["average_risk"] = (
            float(df["risk_score"].mean()) if not df.empty else 0.0
        )

    # Prefer the model's "prediction" column if present, otherwise fall back
    # to the ground-truth "label" column.
    status_column = "prediction" if "prediction" in df.columns else (
        "label" if "label" in df.columns else None
    )
    if status_column is not None:
        summary["normal_count"] = int((df[status_column] == 0).sum())
        summary["anomaly_count"] = int((df[status_column] == 1).sum())

    return summary


# --------------------------------------------------------------------------- #
# 9. Visualization utilities
# --------------------------------------------------------------------------- #

def apply_plot_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared SentinelAI dark-blue theme to a Plotly figure.

    Sets a dark/transparent background, white font, blue accent legend
    styling, and rounded legend borders so every chart in the project
    shares a consistent, professional SOC-dashboard look.

    Args:
        fig: The Plotly figure to theme.

    Returns:
        The same figure, updated in place, for convenient chaining.
    """
    fig.update_layout(
        paper_bgcolor=DEFAULT_PLOT_LAYOUT["paper_bgcolor"],
        plot_bgcolor=DEFAULT_PLOT_LAYOUT["plot_bgcolor"],
        font=DEFAULT_PLOT_LAYOUT["font"],
        legend=dict(
            bgcolor=DEFAULT_PLOT_LAYOUT["legend"]["bgcolor"],
            bordercolor=DEFAULT_PLOT_LAYOUT["legend"]["bordercolor"],
            borderwidth=DEFAULT_PLOT_LAYOUT["legend"]["borderwidth"],
        ),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def show_streamlit_error(message: str) -> None:
    """Display an error message in the running Streamlit app.

    Args:
        message: The error text to display.
    """
    st.error(message)


# --------------------------------------------------------------------------- #
# 10. Formatting utilities
# --------------------------------------------------------------------------- #

def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """Format a number with thousands separators.

    Args:
        value: The numeric value to format.
        decimals: Number of decimal places to include.

    Returns:
        The formatted number as a string, e.g. "10,000" or "10,000.50".
    """
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a fractional or whole-number value as a percentage string.

    Args:
        value: The value to format. Values <= 1 are treated as a fraction
            (e.g. 0.25 -> "25.0%"); values > 1 are treated as an already
            computed percentage (e.g. 25 -> "25.0%").
        decimals: Number of decimal places to include.

    Returns:
        The formatted percentage string, e.g. "25.0%".
    """
    percentage = value * 100 if -1.0 <= value <= 1.0 else value
    return f"{percentage:.{decimals}f}%"


def format_duration(seconds: Union[int, float]) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        A human-readable string such as "45s", "3m 20s", or "1h 5m 10s".
    """
    total_seconds = int(max(0, seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"