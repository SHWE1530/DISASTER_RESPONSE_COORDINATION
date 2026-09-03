# ============================================================
# STAGE 01 - EDA ENGINEER
# Mission: Plot leading indicators and hunt down
#          historical false alarms
#
# Input : Master_Dataset.csv
# Output: EDA plots + EDA analysis CSV files
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib

# Prevent plots from opening and blocking the program
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Folder where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# Master dataset
file_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "Master_Dataset.csv"
)

# EDA output folder
OUTPUT_DIR = BASE_DIR / "eda_outputs"

# Create output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 60)
print("EDA ENGINEER - DATASET LOADING")
print("=" * 60)

print("\nDataset path:")
print(file_path)

if not file_path.exists():

    print("\nERROR: Master_Dataset.csv was not found.")
    print("\nExpected location:")
    print(file_path)

    raise FileNotFoundError(
        f"Dataset not found: {file_path}"
    )


df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("=" * 60)
print("EDA ENGINEER - DATASET LOADED")
print("=" * 60)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATA CHECK")
print("=" * 60)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nRisk Categories:")
print(df["zone_risk"].value_counts())


# ============================================================
# 5. CONVERT TIMESTAMP
# ============================================================

print("\n" + "=" * 60)
print("TIMESTAMP PROCESSING")
print("=" * 60)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

print(
    "\nInvalid timestamps:",
    df["timestamp"].isnull().sum()
)

# Sort chronologically
df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print("\nTimestamp range:")
print("Start:", df["timestamp"].min())
print("End  :", df["timestamp"].max())


# ============================================================
# 6. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

numeric_columns = [
    "rainfall_mm",
    "river_level_m",
    "river_level_threshold_m",
    "emergency_calls",
    "road_closures",
    "bridge_closures",
    "flood_history_count",
    "population_affected",
    "water_level_change_m"
]

print(
    df[numeric_columns].describe()
)


# ============================================================
# 7. RISK-WISE LEADING INDICATOR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("LEADING INDICATOR ANALYSIS")
print("=" * 60)

risk_analysis = (
    df
    .groupby("zone_risk")[numeric_columns]
    .mean()
)

print("\nAverage indicator values by risk:")
print(risk_analysis)

# Save analysis
risk_analysis.to_csv(
    OUTPUT_DIR / "leading_indicator_summary.csv"
)


# ============================================================
# 8. LEADING INDICATOR VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("LEADING INDICATOR VISUALIZATION")
print("=" * 60)


indicators = [
    "rainfall_mm",
    "river_level_m",
    "emergency_calls",
    "road_closures",
    "bridge_closures",
    "flood_history_count",
    "population_affected",
    "water_level_change_m"
]


for column in indicators:

    print(
        f"Generating plot: {column}"
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="zone_risk",
        y=column,
        order=[
            "Low",
            "Moderate",
            "Severe"
        ]
    )

    plt.title(
        f"{column} vs Zone Risk"
    )

    plt.xlabel("Zone Risk")
    plt.ylabel(column)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{column}_vs_risk.png",
        dpi=300,
        bbox_inches="tight"
    )

    # IMPORTANT:
    # No plt.show()
    # This prevents the program from getting stuck.

    plt.close()

    print(
        f"Saved: {column}_vs_risk.png"
    )


# ============================================================
# 9. TIME-SERIES ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TIME-SERIES ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# Rainfall over time
# ------------------------------------------------------------

print("Generating rainfall time-series plot...")

plt.figure(figsize=(12, 5))

plt.plot(
    df["timestamp"],
    df["rainfall_mm"]
)

plt.title(
    "Rainfall Over Time"
)

plt.xlabel("Timestamp")
plt.ylabel("Rainfall (mm)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "rainfall_time_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# River level over time
# ------------------------------------------------------------

print("Generating river-level time-series plot...")

plt.figure(figsize=(12, 5))

plt.plot(
    df["timestamp"],
    df["river_level_m"]
)

plt.title(
    "River Level Over Time"
)

plt.xlabel("Timestamp")
plt.ylabel("River Level (m)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "river_level_time_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Emergency calls over time
# ------------------------------------------------------------

print("Generating emergency-calls time-series plot...")

plt.figure(figsize=(12, 5))

plt.plot(
    df["timestamp"],
    df["emergency_calls"]
)

plt.title(
    "Emergency Calls Over Time"
)

plt.xlabel("Timestamp")
plt.ylabel("Emergency Calls")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "emergency_calls_time_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Time-series plots completed.")


# ============================================================
# 10. SEVERE EVENT PATTERN ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SEVERE EVENT PATTERN ANALYSIS")
print("=" * 60)

severe_df = df[
    df["zone_risk"] == "Severe"
]

print(
    "\nNumber of Severe observations:",
    len(severe_df)
)

print(
    "\nAverage conditions during Severe events:"
)

print(
    severe_df[indicators].mean()
)


# ============================================================
# 11. ZONE / DISTRICT-WISE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DISTRICT-WISE ANALYSIS")
print("=" * 60)


district_risk = pd.crosstab(
    df["district"],
    df["zone_risk"]
)

print(
    "\nRisk distribution by district:"
)

print(district_risk)


# Percentage of Severe events by district

severe_percentage = (
    df
    .groupby("district")["zone_risk"]
    .apply(
        lambda x:
        (x == "Severe").mean() * 100
    )
    .sort_values(
        ascending=False
    )
)


print(
    "\nSevere percentage by district:"
)

print(severe_percentage)


# Save district analysis

severe_percentage.to_csv(
    OUTPUT_DIR /
    "severe_risk_percentage_by_district.csv"
)


# District visualization

print(
    "\nGenerating district risk plot..."
)

plt.figure(figsize=(12, 6))

severe_percentage.plot(
    kind="bar"
)

plt.title(
    "Severe Risk Percentage by District"
)

plt.xlabel("District")
plt.ylabel("Severe Risk (%)")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "severe_risk_by_district.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. RIVER LEVEL vs THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("RIVER LEVEL THRESHOLD ANALYSIS")
print("=" * 60)


df["above_river_threshold"] = (
    df["river_level_m"]
    >
    df["river_level_threshold_m"]
)


print(
    "\nObservations above river threshold:",
    df["above_river_threshold"].sum()
)


threshold_risk = pd.crosstab(
    df["above_river_threshold"],
    df["zone_risk"],
    normalize="index"
) * 100


print(
    "\nRisk percentage based on threshold:"
)

print(threshold_risk)


threshold_risk.to_csv(
    OUTPUT_DIR /
    "river_threshold_risk_analysis.csv"
)


# ============================================================
# 13. CORRELATION FOR LEADING INDICATORS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Supporting analysis only.
# This is NOT model feature importance.


risk_numeric = df.copy()


risk_numeric["risk_numeric"] = (
    risk_numeric["zone_risk"]
    .map({
        "Low": 0,
        "Moderate": 1,
        "Severe": 2
    })
)


correlation_data = (
    df[indicators]
    .copy()
)


correlation_data[
    "risk_numeric"
] = risk_numeric[
    "risk_numeric"
]


correlation_matrix = (
    correlation_data.corr()
)


print(
    "\nCorrelation matrix:"
)

print(correlation_matrix)


plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Leading Indicator Correlation Analysis"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "leading_indicator_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. OUTLIER INVESTIGATION
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER INVESTIGATION")
print("=" * 60)


outlier_records = []


for column in indicators:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = (
        Q1 - 1.5 * IQR
    )

    upper_bound = (
        Q3 + 1.5 * IQR
    )


    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ].copy()


    print(
        f"{column}: "
        f"{len(outliers)} outliers"
    )


    for index in outliers.index:

        outlier_records.append({

            "row_index": index,

            "feature": column,

            "value":
            df.loc[index, column],

            "zone_risk":
            df.loc[index, "zone_risk"],

            "district":
            df.loc[index, "district"],

            "timestamp":
            df.loc[index, "timestamp"]

        })


outlier_df = pd.DataFrame(
    outlier_records
)


outlier_df.to_csv(
    OUTPUT_DIR /
    "outlier_records.csv",
    index=False
)


# ============================================================
# 15. HISTORICAL FALSE ALARM INVESTIGATION
# ============================================================

print("\n" + "=" * 60)
print("HISTORICAL FALSE ALARM INVESTIGATION")
print("=" * 60)


"""
IMPORTANT:

A true false alarm requires an actual historical outcome
showing that the severe warning did NOT occur.

The current dataset contains 'zone_risk' but does not contain
a separate actual_flood_outcome column.

Therefore, we DO NOT automatically delete records.

Instead, we identify suspicious Severe observations
for further investigation.
"""


# Create percentile thresholds

rainfall_low = (
    df["rainfall_mm"]
    .quantile(0.25)
)

river_low = (
    df["river_level_m"]
    .quantile(0.25)
)

calls_low = (
    df["emergency_calls"]
    .quantile(0.25)
)


rainfall_high = (
    df["rainfall_mm"]
    .quantile(0.75)
)

river_high = (
    df["river_level_m"]
    .quantile(0.75)
)

calls_high = (
    df["emergency_calls"]
    .quantile(0.75)
)


# Suspicious Severe records:
# Severe label but unusually low supporting indicators.

suspicious_false_alarms = df[
    (df["zone_risk"] == "Severe") &
    (
        (df["rainfall_mm"] <= rainfall_low) &
        (df["river_level_m"] <= river_low) &
        (df["emergency_calls"] <= calls_low)
    )
].copy()


print(
    "\nPotential suspicious Severe records:",
    len(suspicious_false_alarms)
)


if len(suspicious_false_alarms) > 0:

    print(
        "\nFirst 20 suspicious observations:"
    )

    print(
        suspicious_false_alarms[
            [
                "timestamp",
                "state",
                "district",
                "rainfall_mm",
                "river_level_m",
                "emergency_calls",
                "zone_risk"
            ]
        ].head(20)
    )

else:

    print(
        "\nNo suspicious Severe observations found "
        "using the current rule."
    )


# Save suspicious records

suspicious_false_alarms.to_csv(
    OUTPUT_DIR /
    "potential_false_alarms.csv",
    index=False
)


# ============================================================
# 16. FALSE ALARM SUPPORTING VISUALIZATION
# ============================================================

print(
    "\nGenerating false alarm investigation plot..."
)


plt.figure(figsize=(10, 6))


sns.scatterplot(
    data=df,
    x="rainfall_mm",
    y="river_level_m",
    hue="zone_risk",
    style="zone_risk",
    s=70
)


# Highlight suspicious records

if len(suspicious_false_alarms) > 0:

    plt.scatter(
        suspicious_false_alarms[
            "rainfall_mm"
        ],

        suspicious_false_alarms[
            "river_level_m"
        ],

        s=150,

        facecolors="none",

        edgecolors="black",

        linewidths=2,

        label="Potential False Alarm"
    )


plt.title(
    "Rainfall vs River Level - "
    "False Alarm Investigation"
)

plt.xlabel(
    "Rainfall (mm)"
)

plt.ylabel(
    "River Level (m)"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR /
    "false_alarm_investigation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. LEADING INDICATOR SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LEADING INDICATOR SUMMARY")
print("=" * 60)


low_mean = (
    df[
        df["zone_risk"] == "Low"
    ][indicators]
    .mean()
)


moderate_mean = (
    df[
        df["zone_risk"] == "Moderate"
    ][indicators]
    .mean()
)


severe_mean = (
    df[
        df["zone_risk"] == "Severe"
    ][indicators]
    .mean()
)


summary = pd.DataFrame({

    "Low": low_mean,

    "Moderate": moderate_mean,

    "Severe": severe_mean

})


print(summary)


summary.to_csv(
    OUTPUT_DIR /
    "leading_indicator_summary.csv"
)


# ============================================================
# 18. AUTOMATIC EDA FINDINGS
# ============================================================

print("\n" + "=" * 60)
print("EDA FINDINGS")
print("=" * 60)


for feature in indicators:

    low_value = (
        low_mean[feature]
    )

    severe_value = (
        severe_mean[feature]
    )


    if low_value != 0:

        percentage_change = (

            (
                severe_value -
                low_value
            )

            /

            abs(low_value)

        ) * 100


        print(

            f"{feature}: "

            f"Low={low_value:.2f}, "

            f"Severe={severe_value:.2f}, "

            f"Change={percentage_change:.2f}%"

        )


# ============================================================
# 19. CREATE EDA-CHECKED DATASET
# ============================================================

"""
We DO NOT automatically delete suspicious records because
a suspicious record is not automatically a confirmed false alarm.

The suspicious records are saved separately for investigation.

Therefore, the original cleaned dataset remains intact.
"""


eda_checked_df = df.copy()


# Remove helper column

eda_checked_df = (
    eda_checked_df.drop(
        columns=[
            "above_river_threshold"
        ],
        errors="ignore"
    )
)


# Convert timestamp back to readable format

eda_checked_df[
    "timestamp"
] = (
    eda_checked_df[
        "timestamp"
    ]
    .dt.strftime(
        "%d-%m-%Y %H:%M"
    )
)


# Save checked dataset

eda_checked_df.to_csv(
    OUTPUT_DIR /
    "eda_checked_dataset.csv",
    index=False
)


# ============================================================
# 20. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("EDA ENGINEER PROCESS COMPLETED")
print("=" * 60)


print("\nOutputs generated inside:")
print(OUTPUT_DIR)


print("\nCSV outputs:")

print(
    "1. eda_checked_dataset.csv"
)

print(
    "2. leading_indicator_summary.csv"
)

print(
    "3. potential_false_alarms.csv"
)

print(
    "4. outlier_records.csv"
)

print(
    "5. severe_risk_percentage_by_district.csv"
)

print(
    "6. river_threshold_risk_analysis.csv"
)


print("\nPNG outputs:")

print(
    "1. rainfall_mm_vs_risk.png"
)

print(
    "2. river_level_m_vs_risk.png"
)

print(
    "3. emergency_calls_vs_risk.png"
)

print(
    "4. road_closures_vs_risk.png"
)

print(
    "5. bridge_closures_vs_risk.png"
)

print(
    "6. flood_history_count_vs_risk.png"
)

print(
    "7. population_affected_vs_risk.png"
)

print(
    "8. water_level_change_m_vs_risk.png"
)

print(
    "9. rainfall_time_series.png"
)

print(
    "10. river_level_time_series.png"
)

print(
    "11. emergency_calls_time_series.png"
)

print(
    "12. severe_risk_by_district.png"
)

print(
    "13. leading_indicator_correlation.png"
)

print(
    "14. false_alarm_investigation.png"
)


print("\nEDA Engineer responsibility completed.")

print(
    "Dataset is ready for the ML Engineer."
)

print("=" * 60)