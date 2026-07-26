"""
train_model.py

Honeywell SentinelAI hackathon project.

Loads the synthetic login-event dataset produced by generate_data.py, builds
behavioural features, encodes categorical columns, trains an IsolationForest
anomaly detector, generates an explainable risk score for every event, and
evaluates the result against the ground-truth label column.

Input:
    data/raw/login_logs.csv

Outputs:
    data/processed/processed_logs.csv
    models/anomaly_model.pkl
    models/scaler.pkl
    models/label_encoders.pkl

Run with:
    python train_model.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

RAW_DATA_PATH = Path("data/raw/login_logs.csv")
PROCESSED_DATA_PATH = Path("data/processed/processed_logs.csv")
MODEL_PATH = Path("models/anomaly_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
ENCODERS_PATH = Path("models/label_encoders.pkl")

CATEGORICAL_COLUMNS: List[str] = [
    "entity_type",
    "geo_location",
    "resource_accessed",
    "auth_method",
    "command_sequence",
    "browser",
    "operating_system",
    "login_result",
]

# Columns that must never be used as model input.
NON_FEATURE_COLUMNS: List[str] = [
    "entity_id",
    "timestamp",
    "attack_type",
    "label",
    # source_ip and device_fingerprint are high-cardinality identifiers, not
    # behavioural signals; the information they carry is already captured
    # via engineered features such as new_device.
    "source_ip",
    "device_fingerprint",
]

SENSITIVE_RESOURCES = {
    "Database", "Finance", "Payroll", "Production Server", "SCADA", "PLC",
}

LONG_SESSION_THRESHOLD_SECONDS = 1800  # 30 minutes

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.03,
    "random_state": 42,
    "n_jobs": -1,
}

RISK_WEIGHTS = {
    "off_hour": 15,
    "weekend": 10,
    "sensitive_resource": 20,
    "long_session": 15,
    "failed_login": 20,
    "anomaly": 20,
}


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #

def load_data(path: Path) -> pd.DataFrame:
    """Load the raw login-log CSV, validating that it exists.

    Args:
        path: Path to the input CSV file.

    Returns:
        The loaded dataframe.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Run generate_data.py first."
        )

    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape}")
    print("Missing values per column:")
    print(df.isnull().sum().to_string())

    return df


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe and extract basic time-based features.

    Removes duplicate rows, fills missing values, parses the timestamp
    column, and derives hour / day / weekday / weekend / month features.

    Args:
        df: Raw input dataframe.

    Returns:
        Cleaned dataframe with new time-based columns.
    """
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values: numeric columns get median imputation,
    # categorical/object columns get an explicit "unknown" placeholder.
    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna("unknown")

    # Parse timestamp into a real datetime column.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Derive behavioural time features.
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday  # 0 = Monday
    df["is_weekend_flag"] = (df["weekday"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month

    return df


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional behavioural features used by the model and scorer.

    Must be called BEFORE categorical encoding, since it relies on the raw
    text values of resource_accessed, login_result, and device_fingerprint.

    Args:
        df: Preprocessed dataframe (with time features already extracted).

    Returns:
        Dataframe with new engineered feature columns.
    """
    # Office hours: 08:00 - 18:59.
    df["is_office_hours"] = df["hour"].between(8, 18).astype(int)

    # Reuse the weekend flag computed during preprocessing.
    df["is_weekend"] = df["is_weekend_flag"]

    # Sensitive resource flag.
    df["is_sensitive_resource"] = (
        df["resource_accessed"].isin(SENSITIVE_RESOURCES).astype(int)
    )

    # Long session flag.
    df["long_session"] = (
        df["session_duration"] > LONG_SESSION_THRESHOLD_SECONDS
    ).astype(int)

    # Failed login flag.
    df["failed_login"] = (df["login_result"] == "fail").astype(int)

    # New device flag: does this event's device fingerprint differ from the
    # entity's most common (historically typical) fingerprint?
    def _most_common(series: pd.Series) -> str:
        return Counter(series).most_common(1)[0][0]

    typical_fingerprint = df.groupby("entity_id")["device_fingerprint"].transform(
        _most_common
    )
    df["new_device"] = (df["device_fingerprint"] != typical_fingerprint).astype(int)

    # Resource frequency: how often this entity normally accesses this
    # specific resource, as a proportion of that entity's total events.
    pair_counts = df.groupby(["entity_id", "resource_accessed"])[
        "resource_accessed"
    ].transform("count")
    entity_counts = df.groupby("entity_id")["entity_id"].transform("count")
    df["resource_frequency"] = pair_counts / entity_counts

    # Cyclical encoding of the hour of day.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Min-max normalized session duration (bounded 0-1 feature).
    min_duration = df["session_duration"].min()
    max_duration = df["session_duration"].max()
    duration_range = max_duration - min_duration
    if duration_range == 0:
        df["session_duration_normalized"] = 0.0
    else:
        df["session_duration_normalized"] = (
            (df["session_duration"] - min_duration) / duration_range
        )

    return df


# --------------------------------------------------------------------------- #
# 4. Encoding
# --------------------------------------------------------------------------- #

def encode(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode the given categorical columns in place.

    Args:
        df: Dataframe containing the categorical columns (post feature
            engineering, since feature engineering needs the raw text).
        columns: List of categorical column names to encode.

    Returns:
        A tuple of (encoded dataframe, dict of fitted LabelEncoders).
    """
    encoders: Dict[str, LabelEncoder] = {}

    for column in columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        encoders[column] = encoder

    return df, encoders


# --------------------------------------------------------------------------- #
# 5 & 6. Feature selection and scaling
# --------------------------------------------------------------------------- #

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix X, excluding non-feature columns.

    Args:
        df: Fully engineered and encoded dataframe.

    Returns:
        Feature matrix X (numeric columns only).
    """
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df[feature_columns].copy()


def scale(x: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Scale numerical features using StandardScaler.

    Args:
        x: Raw feature matrix.

    Returns:
        A tuple of (scaled feature array, fitted scaler).
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, scaler


# --------------------------------------------------------------------------- #
# 7 & 8. Model training and prediction
# --------------------------------------------------------------------------- #

def train_model(x_scaled: np.ndarray) -> IsolationForest:
    """Train an IsolationForest anomaly detection model.

    Args:
        x_scaled: Scaled feature matrix.

    Returns:
        The fitted IsolationForest model.
    """
    model = IsolationForest(**ISOLATION_FOREST_PARAMS)
    model.fit(x_scaled)
    return model


def predict(model: IsolationForest, x_scaled: np.ndarray) -> np.ndarray:
    """Generate anomaly predictions and convert them to 0/1 labels.

    IsolationForest natively returns -1 (anomaly) / 1 (normal); this is
    converted to the project convention of 0 = normal, 1 = anomaly.

    Args:
        model: Fitted IsolationForest model.
        x_scaled: Scaled feature matrix.

    Returns:
        Array of predictions where 0 = normal and 1 = anomaly.
    """
    raw_predictions = model.predict(x_scaled)
    return np.where(raw_predictions == -1, 1, 0)


# --------------------------------------------------------------------------- #
# 9 & 10. Risk scoring and explanation
# --------------------------------------------------------------------------- #

def calculate_risk(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Compute an explainable, weighted risk score (0-100) for every event.

    Args:
        df: Dataframe with engineered behavioural flag columns.
        predictions: Array of model predictions (0 = normal, 1 = anomaly).

    Returns:
        Dataframe with new risk_score and risk_level columns.
    """
    df = df.copy()
    df["prediction"] = predictions

    risk_score = (
        df["is_office_hours"].apply(lambda v: 0 if v == 1 else RISK_WEIGHTS["off_hour"])
        + df["is_weekend"] * RISK_WEIGHTS["weekend"]
        + df["is_sensitive_resource"] * RISK_WEIGHTS["sensitive_resource"]
        + df["long_session"] * RISK_WEIGHTS["long_session"]
        + df["failed_login"] * RISK_WEIGHTS["failed_login"]
        + df["prediction"] * RISK_WEIGHTS["anomaly"]
    )

    df["risk_score"] = risk_score.clip(0, 100).astype(int)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 24, 49, 74, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    return df


def generate_explanation(df: pd.DataFrame) -> pd.DataFrame:
    """Build a human-readable explanation string for every event's risk score.

    Args:
        df: Dataframe with engineered flag columns and predictions.

    Returns:
        Dataframe with a new explanation column.
    """

    def _explain(row: pd.Series) -> str:
        reasons: List[str] = []
        if row["is_office_hours"] == 0:
            reasons.append("Off-hour login")
        if row["is_weekend"] == 1:
            reasons.append("Weekend login")
        if row["is_sensitive_resource"] == 1:
            reasons.append("Sensitive resource")
        if row["long_session"] == 1:
            reasons.append("Long session")
        if row["failed_login"] == 1:
            reasons.append("Failed login")
        if row.get("new_device", 0) == 1:
            reasons.append("New device")
        if row["prediction"] == 1:
            reasons.append("Isolation Forest anomaly")

        return ", ".join(reasons) if reasons else "No risk indicators"

    df = df.copy()
    df["explanation"] = df.apply(_explain, axis=1)
    return df


# --------------------------------------------------------------------------- #
# 11. Evaluation
# --------------------------------------------------------------------------- #

def evaluate(df: pd.DataFrame) -> None:
    """Print evaluation metrics comparing predictions against ground truth.

    Args:
        df: Dataframe containing both the true "label" column and the
            model's "prediction" column.
    """
    y_true = df["label"]
    y_pred = df["prediction"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n--- Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


# --------------------------------------------------------------------------- #
# 12. Saving artifacts
# --------------------------------------------------------------------------- #

def save_files(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    encoders: Dict[str, LabelEncoder],
) -> None:
    """Persist the processed dataframe and all trained artifacts to disk.

    Args:
        df: Fully processed dataframe (with risk scores and explanations).
        model: Fitted IsolationForest model.
        scaler: Fitted StandardScaler.
        encoders: Dict of fitted LabelEncoders keyed by column name.
    """
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)


# --------------------------------------------------------------------------- #
# 13. Summary
# --------------------------------------------------------------------------- #

def print_summary(df: pd.DataFrame) -> None:
    """Print a final summary of the training run.

    Args:
        df: Fully processed dataframe with predictions and risk scores.
    """
    total = len(df)
    normal_count = int((df["prediction"] == 0).sum())
    anomaly_count = int((df["prediction"] == 1).sum())
    avg_risk = df["risk_score"].mean()

    print("\n--- Summary ---")
    print(f"Dataset size: {total}")
    print(f"Normal count: {normal_count}")
    print(f"Anomaly count: {anomaly_count}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Processed dataset saved: {PROCESSED_DATA_PATH}")
    print(f"Average risk score: {avg_risk:.2f}")


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the full training and scoring pipeline end to end."""
    try:
        df = load_data(RAW_DATA_PATH)
        df = preprocess(df)
        df = feature_engineering(df)
        df, encoders = encode(df, CATEGORICAL_COLUMNS)

        x = build_feature_matrix(df)
        x_scaled, scaler = scale(x)

        model = train_model(x_scaled)
        predictions = predict(model, x_scaled)

        df = calculate_risk(df, predictions)
        df = generate_explanation(df)

        evaluate(df)
        save_files(df, model, scaler, encoders)
        print_summary(df)

    except FileNotFoundError as error:
        print(f"Error: {error}")
    except Exception as error:  # noqa: BLE001 - top-level safety net
        print(f"Unexpected error during pipeline execution: {error}")
        raise


if __name__ == "__main__":
    main()