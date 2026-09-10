import pandas as pd
import numpy as np
import re
from pathlib import Path

INPUT_FILE = "Safety_SOP_Dataset_PROCESSED.csv"
OUTPUT_FILE = "Safety_SOP_Dataset_FINAL.csv"

df = pd.read_csv(INPUT_FILE)

print("Original Shape:", df.shape)
print("\nOriginal Columns:")
print(df.columns.tolist())

required_columns = [
    "hazard_type",
    "incident_type",
    "severity_level",
    "required_action",
    "response_procedure",
    "evacuation_instruction",
    "resource_required",
    "priority_level",
    "safety_precaution",
    "communication_code"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nMissing Columns:")
    print(missing_columns)
else:
    print("\nAll required columns are available")

def clean_text(value):
    if pd.isna(value):
        return ""
    
    value = str(value)
    value = re.sub(r"http\S+|www\S+", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip()
    
    return value

text_columns = [
    "hazard_type",
    "incident_type",
    "severity_level",
    "required_action",
    "response_procedure",
    "evacuation_instruction",
    "resource_required",
    "priority_level",
    "safety_precaution",
    "communication_code"
]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].apply(clean_text)

if "hazard_type" in df.columns:
    df["hazard_type"] = df["hazard_type"].str.lower()

if "incident_type" in df.columns:
    df["incident_type"] = df["incident_type"].str.lower()

if "severity_level" in df.columns:
    df["severity_level"] = df["severity_level"].str.title()

if "priority_level" in df.columns:
    df["priority_level"] = df["priority_level"].str.title()

before_duplicates = len(df)

if "record_id" in df.columns:
    df = df.drop_duplicates(subset=["record_id"])
else:
    df = df.drop_duplicates()

after_duplicates = len(df)

print("\nDuplicates Removed:", before_duplicates - after_duplicates)

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].replace("", "Unknown")

if "required_action" in df.columns:
    df["action_length"] = df["required_action"].str.len()

if "response_procedure" in df.columns:
    df["procedure_length"] = df["response_procedure"].str.len()

if "required_action" in df.columns:
    df["action_word_count"] = df["required_action"].str.split().str.len()

if "response_procedure" in df.columns:
    df["procedure_word_count"] = df["response_procedure"].str.split().str.len()

if "severity_level" in df.columns:
    severity_mapping = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }
    
    df["severity_score"] = (
        df["severity_level"]
        .map(severity_mapping)
        .fillna(0)
        .astype(int)
    )

if "priority_level" in df.columns:
    priority_mapping = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }
    
    df["priority_score"] = (
        df["priority_level"]
        .map(priority_mapping)
        .fillna(0)
        .astype(int)
    )

missing_values = df.isnull().sum()

print("\nMissing Value Report:")
print(missing_values[missing_values > 0])

duplicate_count = df.duplicated().sum()

print("\nRemaining Duplicate Rows:", duplicate_count)

empty_rows = df.isna().all(axis=1).sum()

print("Completely Empty Rows:", empty_rows)

df = df.dropna(how="all")

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nFinal Shape:", df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nDataset Saved Successfully:")
print(OUTPUT_FILE)

print("\nData Engineering Pipeline Completed")

print("\nSample Records:")
print(df.head())

print("\nSeverity Distribution:")
if "severity_level" in df.columns:
    print(df["severity_level"].value_counts())

print("\nHazard Distribution:")
if "hazard_type" in df.columns:
    print(df["hazard_type"].value_counts())

print("\nFinal Dataset Information:")
print(df.info())
