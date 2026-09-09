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
    df[col] = df[col].fillna(median_val)
    print(f"Imputed missing values in '{col}' using median: {median_val}")

categorical_cols = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()
for col in categorical_cols:
  if df[col].isnull().sum() > 0:
    mode_values = df[col].mode()
    if not mode_values.empty:
      mode_val = mode_values.iloc[0]
      df[col] = df[col].fillna(mode_val)
      print(f"Imputed missing values in '{col}' using mode: {mode_val}")
    else:
      df[col] = df[col].fillna("Unknown")
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

# 4.5. TARGET LABEL (zone_risk) VALIDATION -- DO NOT REGENERATE
#
# An earlier revision of this script OVERWROTE zone_risk with a hand-written
# rule over river_level_m / river_level_threshold_m / emergency_calls /
# rainfall_mm. That rule is a deterministic function of columns that are
# themselves model inputs -- and the engineered feature river_level_margin_m
# (river_level_m - river_level_threshold_m) is literally the rule's own
# decision variable. Training on those labels is target leakage by
# construction: the model would be approximating an if/else using the if/else's
# own inputs, and reported accuracy would approach 100% while measuring
# nothing.
#
# The rule is retained below ONLY as a documented sanity reference. It is never
# written back to the dataset. zone_risk is carried through unchanged from the
# source dataset, which is what the shipped model was trained on.
print("\nValidating 'zone_risk' target label (labels are preserved, not regenerated)...")

if "zone_risk" not in df.columns:
  raise ValueError(
      "Source dataset is missing the 'zone_risk' target column. This pipeline "
      "consumes ground-truth labels; it does not synthesise them."
  )

if df["zone_risk"].isnull().any():
  raise ValueError(
      f"Source dataset has {int(df['zone_risk'].isnull().sum())} rows with a "
      "missing zone_risk label. Fix the source data; do not impute the target."
  )


def reference_risk_rule(row):
  """Documented threshold heuristic. Diagnostic only - NEVER used as the target.

  Reported as an agreement rate so the gap between the operational threshold
  heuristic and the observed labels is visible and auditable.
  """
  if row.get("river_level_m", 0) > row.get("river_level_threshold_m", 999):
    return "Severe"
  if row.get("emergency_calls", 0) >= 35 and row.get("rainfall_mm", 0) >= 40:
    return "Severe"
  if row.get("rainfall_mm", 0) >= 20 or row.get("emergency_calls", 0) >= 25:
    return "Moderate"
  return "Low"


rule_agreement = float((df.apply(reference_risk_rule, axis=1) == df["zone_risk"]).mean())
print(f"Label distribution: {df['zone_risk'].value_counts().to_dict()}")
print(f"Threshold-heuristic agreement with observed labels: {rule_agreement:.4f}")
print(
    "NOTE: labels are preserved from the source dataset. The heuristic above is "
    "a diagnostic reference only and is deliberately NOT used as the target."
)

# 5. CONVERT & SAVE MASTER PROCESSED DATASET
# Restore timestamp to the original "%d-%m-%Y %H:%M" string format so that the
# downstream EDA/ML scripts (which parse that exact format) can consume this file.
if "timestamp" in df.columns:
  df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime(
      "%d-%m-%Y %H:%M"
  )

output_file = processed_dir / "Master_Processed_Dataset.csv"
df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print(f"Master Dataset saved successfully as: {output_file}")
print(f"Final Dataset Shape: {df.shape}")
print("=" * 60)