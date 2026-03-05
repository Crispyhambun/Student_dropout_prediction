from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "dropout_model.pkl"

FEATURE_COLUMNS = [
    "Age at enrollment",
    "Gender",
    "Scholarship holder",
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Debtor",
    "Displaced",
    "Unemployment rate",
    "GDP",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Scholarship holder",
    "Tuition fees up to date",
    "Debtor",
    "Displaced",
]


def load_model_artifacts() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifacts not found at {MODEL_PATH}. Run model/train_model.py first."
        )
    artifacts = joblib.load(MODEL_PATH)
    return artifacts


def prepare_features_from_df(df: pd.DataFrame, artifacts: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare encoded and scaled features from a raw DataFrame using saved artifacts."""

    feature_cols = artifacts["feature_columns"]
    categorical_cols = artifacts["categorical_columns"]
    numeric_cols = artifacts["numeric_columns"]
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]

    X = df[feature_cols].copy()

    for col in categorical_cols:
        le = encoders[col]
        X[col] = le.transform(X[col])

    X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X, X.values


def build_single_input_dataframe(
    age: int,
    gender: str,
    scholarship: str,
    fees_up_to_date: str,
    units_sem1_approved: int,
    units_sem2_approved: int,
    gpa_sem2: float,
    debtor: str,
    displaced: str,
    unemployment_rate: float,
    gdp: float,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature schema."""

    # Dataset uses numeric encodings (0/1) for many binary attributes.
    # We map human-readable inputs to those numeric codes.

    yes_no_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}

    data = {
        "Age at enrollment": age,
        "Gender": gender_map.get(gender, 0),
        "Scholarship holder": yes_no_map.get(scholarship, 0),
        "Tuition fees up to date": yes_no_map.get(fees_up_to_date, 1),
        "Curricular units 1st sem (approved)": units_sem1_approved,
        "Curricular units 2nd sem (approved)": units_sem2_approved,
        "Curricular units 2nd sem (grade)": gpa_sem2,
        "Debtor": yes_no_map.get(debtor, 0),
        "Displaced": yes_no_map.get(displaced, 0),
        "Unemployment rate": unemployment_rate,
        "GDP": gdp,
    }

    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def compute_risk_category(probability: float) -> str:
    if probability < 0.33:
        return "Low Risk"
    elif probability < 0.66:
        return "Moderate Risk"
    else:
        return "High Risk"


def recommendation_for_risk(category: str) -> str:
    if category == "High Risk":
        return "This student shows high dropout indicators. Immediate counseling recommended."
    if category == "Moderate Risk":
        return "This student shows some risk factors. Close monitoring and supportive interventions are advised."
    return "This student is on track. Keep monitoring performance and providing guidance."
