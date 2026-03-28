from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["name", "tempo", "energy", "valence"]
NUMERIC_COLUMNS = ["tempo", "energy", "valence"]


def load_and_preprocess_songs(csv_path: str) -> pd.DataFrame:
    """
    Load songs from CSV, clean missing values, and normalize numeric features.

    Expected columns:
    - name
    - tempo
    - energy
    - valence
    """
    df = pd.read_csv(csv_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # Keep only required columns and coerce numeric values safely.
    df = df[REQUIRED_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing names and numeric values.
    df["name"] = df["name"].fillna("Unknown Song").astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        df[col] = df[col].fillna(median_value)

    # Min-max normalization to [0, 1].
    for col in NUMERIC_COLUMNS:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val == min_val:
            df[col] = 0.0
        else:
            df[col] = (df[col] - min_val) / (max_val - min_val)

    return df
