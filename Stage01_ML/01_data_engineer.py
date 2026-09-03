"""Data Preprocessing Pipeline: Missing Values, Duplicates, and Outlier Handling

Exports Cleaned Master Dataset for EDA and Modeling.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 1. RETRIEVE RAW DATA
base_dir = Path(__file__).resolve().parent
processed_dir = base_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

candidate_paths = [
    processed_dir / "Master_Dataset.csv",
    processed_dir / "Processed_Stage1_Dataset.csv",
    base_dir / "Processed_Stage1_Dataset.csv",
    base_dir / "Master_Dataset.csv",
]

input_file = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
if not input_file.exists():
  raise FileNotFoundError(
      "No dataset file found. Expected one of: "
      + ", ".join(str(path) for path in candidate_paths)
  )

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file)
print(f"Initial Dataset Shape: {df.shape}")

# 2. DUPLICATE HANDLING
duplicates_count = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates_count}")
if duplicates_count > 0:
  df = df.drop_duplicates().reset_index(drop=True)
  print(f"Dataset shape after dropping duplicates: {df.shape}")
else:
  print("No duplicate rows found.")

# 3. MISSING VALUES HANDLING
print("\nChecking for missing values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
  if df[col].isnull().sum() > 0:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"Imputed missing values in '{col}' using median: {median_val}")

categorical_cols = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()
for col in categorical_cols:
  if df[col].isnull().sum() > 0:
    mode_values = df[col].mode()
    if not mode_values.empty:
      mode_val = mode_values.iloc[0]
      df[col].fillna(mode_val, inplace=True)
      print(f"Imputed missing values in '{col}' using mode: {mode_val}")
    else:
      df[col].fillna("Unknown", inplace=True)
      print(f"Imputed missing values in '{col}' using fallback: Unknown")

# Standardize timestamp column if present
if "timestamp" in df.columns:
  df["timestamp"] = pd.to_datetime(
      df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce"
  )
  df["timestamp"] = df["timestamp"].ffill().bfill()
  print("Standardized 'timestamp' column to datetime format.")

# 4. OUTLIER HANDLING (IQR Capping / Winsorization)
target_outlier_cols = [
    "rainfall_mm",
    "emergency_calls",
    "road_closures",
    "bridge_closures",
    "flood_history_count",
    "population_affected",
    "water_level_change_m",
]

print("\nApplying IQR Outlier Capping...")
for col in target_outlier_cols:
  if col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers before capping
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    print(
        f" - {col}: {outliers.sum()} outliers capped to [{lower_bound:.2f},"
        f" {upper_bound:.2f}]"
    )

    # Cap outliers
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# 5. CONVERT & SAVE MASTER PROCESSED DATASET
output_file = processed_dir / "Master_Processed_Dataset.csv"
df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print(f"Master Dataset saved successfully as: {output_file}")
print(f"Final Dataset Shape: {df.shape}")
print("=" * 60)