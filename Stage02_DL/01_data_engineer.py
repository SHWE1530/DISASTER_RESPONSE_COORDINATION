import os
import re
import sys
import glob
import json
import shutil
import zipfile
import warnings
from pathlib import Path
from io import BytesIO

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn Metrics & Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning Framework
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM,
    MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Add
)

# ==============================================================================
# 0. REPRODUCIBILITY & DIRECTORY SETUP
# ==============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path("/content/STAGE_02_DL")
BUILD_DIR = BASE_DIR / "FINAL_STAGE2_BUILD"
DATASET_DIR = Path("/content/STAGE_02_DATASET")
PACKAGE_DIR = Path("/content/STAGE_02_COMPLETE_DATASET")

PIPELINE_DIRS = {
    "raw_s1": BUILD_DIR / "satellite_data" / "sentinel1",
    "raw_bhuvan": BUILD_DIR / "satellite_data" / "bhuvan_maps",
    "flood_masks": BUILD_DIR / "satellite_data" / "flood_masks",
    "image_flood": BUILD_DIR / "image_data" / "flooded",
    "image_nonflood": BUILD_DIR / "image_data" / "non_flooded",
    "processed": BUILD_DIR / "processed_data",
    "models": BUILD_DIR / "models",
    "results": BUILD_DIR / "results",
    "metadata": BUILD_DIR / "metadata"
}

for folder in PIPELINE_DIRS.values():
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🚀 DATA ENGINEERING & DEEP LEARNING PIPELINE INITIALIZED")
print("=" * 80)


# ==============================================================================
# 1. DATA EXTRACTION & INGESTION (TABULAR SOURCES)
# ==============================================================================
def resolve_master_dataset():
    """Search for Master_Dataset.csv across runtime paths or trigger interactive upload."""
    search_paths = [
        "/content/Master_Dataset.csv",
        "/content/drive/MyDrive/Master_Dataset.csv",
        str(BASE_DIR / "Master_Dataset.csv"),
        str(PIPELINE_DIRS["metadata"] / "Master_Dataset.csv")
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return Path(path)
            
    matches = glob.glob("/content/**/Master_Dataset.csv", recursive=True)
    if matches:
        return Path(matches[0])
        
    print("\n⚠️ Master_Dataset.csv not found in runtime. Launching upload prompt...")
    from google.colab import files
    uploaded = files.upload()
    for name in uploaded.keys():
        if name.lower() == "master_dataset.csv":
            return Path("/content") / name
            
    raise FileNotFoundError("Master_Dataset.csv is required to proceed with temporal engineering.")

master_file = resolve_master_dataset()
shutil.copy2(master_file, PIPELINE_DIRS["metadata"] / "Master_Dataset.csv")
print(f"✅ Ingested Master Dataset from: {master_file}")


# ==============================================================================
# 2. DATA CLEANING & SCHEMATIZATION
# ==============================================================================
print("\n" + "=" * 80)
print("🧹 STAGE: DATA CLEANING & TEMPORAL VALIDATION")
print("=" * 80)

raw_df = pd.read_csv(master_file)

# Standardize schema names
raw_df.columns = [
    str(col).strip().lower().replace(" ", "_").replace("-", "_") 
    for col in raw_df.columns
]

# Temporal parsing & deduplication
if "timestamp" in raw_df.columns:
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    raw_df = raw_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

REQUIRED_NUMERICS = [
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

available_features = [col for col in REQUIRED_NUMERICS if col in raw_df.columns]

if "river_level_m" not in available_features:
    raise ValueError("Target feature 'river_level_m' missing from dataset schema.")

for col in available_features:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# Imputation strategy: linear interpolation followed by median fallback
raw_df[available_features] = raw_df[available_features].replace([np.inf, -np.inf], np.nan)
raw_df[available_features] = raw_df[available_features].interpolate(method="linear", limit_direction="both")
for col in available_features:
    raw_df[col] = raw_df[col].fillna(raw_df[col].median())

cleaned_df = raw_df.dropna(subset=["river_level_m"]).reset_index(drop=True)
cleaned_df.to_csv(PIPELINE_DIRS["processed"] / "Master_Dataset_Cleaned.csv", index=False)
print(f"✅ Cleaned records: {len(cleaned_df)} rows | Features: {len(available_features)}")


# ==============================================================================
# 3. TEMPORAL FEATURE ENGINEERING & DATA LEAKAGE PREVENTION
# ==============================================================================
print("\n" + "=" * 80)
print("⚙️ STAGE: ROLLING AGGREGATIONS & CHRONOLOGICAL PARTITIONING")
print("=" * 80)

# Domain-specific hydrologic features
cleaned_df["threshold_exceedance_m"] = (
    cleaned_df["river_level_m"] - cleaned_df.get("river_level_threshold_m", 0)
)
cleaned_df["rainfall_3h"] = cleaned_df["rainfall_mm"].rolling(3, min_periods=1).sum()
cleaned_df["rainfall_6h"] = cleaned_df["rainfall_mm"].rolling(6, min_periods=1).sum()
cleaned_df["water_level_3h_mean"] = cleaned_df["river_level_m"].rolling(3, min_periods=1).mean()
cleaned_df["water_level_6h_mean"] = cleaned_df["river_level_m"].rolling(6, min_periods=1).mean()

engineered_features = available_features + [
    "threshold_exceedance_m",
    "rainfall_3h",
    "rainfall_6h",
    "water_level_3h_mean",
    "water_level_6h_mean"
]

cleaned_df[engineered_features] = cleaned_df[engineered_features].interpolate().ffill().bfill()

# Chronological partition thresholds (No random shuffle to avoid leakage)
N_ROWS = len(cleaned_df)
TRAIN_IDX = int(N_ROWS * 0.70)
VAL_IDX = int(N_ROWS * 0.85)

train_split = cleaned_df.iloc[:TRAIN_IDX]

# Strict scaling: fit on train split only
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(train_split[engineered_features].values)
target_scaler.fit(train_split[["river_level_m"]].values)

X_scaled = feature_scaler.transform(cleaned_df[engineered_features].values)
y_scaled = target_scaler.transform(cleaned_df[["river_level_m"]].values).flatten()

# Multi-step Sequence Windowing
LOOKBACK = 24
FORECAST_HOURS = 3

X_seq, y_seq, seq_timestamps = [], [], []
for i in range(LOOKBACK, N_ROWS - FORECAST_HOURS + 1):
    X_seq.append(X_scaled[i - LOOKBACK:i])
    target_idx = i + FORECAST_HOURS - 1
    y_seq.append(y_scaled[target_idx])
    seq_timestamps.append(cleaned_df["timestamp"].iloc[target_idx])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
seq_timestamps = np.array(seq_timestamps)

TOTAL_SEQS = len(X_seq)
train_end = int(TOTAL_SEQS * 0.70)
val_end = int(TOTAL_SEQS * 0.85)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]
test_timestamps = seq_timestamps[val_end:]

print(f"✅ Sequences constructed: {TOTAL_SEQS} | Input: {LOOKBACK}h | Horizon: +{FORECAST_HOURS}h")
print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")


# ==============================================================================
# 4. TEMPORAL MODEL TRAINING (LSTM & TRANSFORMER)
# ==============================================================================
print("\n" + "=" * 80)
print("🧠 STAGE: MODEL COMPILATION & BENCHMARKING")
print("=" * 80)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# --- A. Stacked LSTM ---
tf.keras.backend.clear_session()
lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.20),
    LSTM(32),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

lstm_path = PIPELINE_DIRS["models"] / "water_level_lstm.keras"
lstm_model.save(lstm_path)

# --- B. Multi-Head Attention Transformer ---
tf.keras.backend.clear_session()
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
proj = Dense(64)(inputs)
attention = MultiHeadAttention(num_heads=4, key_dim=16)(proj, proj)
x = Add()([proj, attention])
x = LayerNormalization()(x)

ff = Dense(128, activation="relu")(x)
ff = Dropout(0.20)(ff)
ff = Dense(64)(ff)
x = Add()([x, ff])
x = LayerNormalization()(x)

x = GlobalAveragePooling1D()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.20)(x)
outputs = Dense(1)(x)

transformer_model = Model(inputs, outputs)
transformer_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history_transformer = transformer_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

transformer_path = PIPELINE_DIRS["models"] / "water_level_transformer.keras"
transformer_model.save(transformer_path)

# Evaluate on test set
actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
lstm_preds = target_scaler.inverse_transform(lstm_model.predict(X_test, verbose=0).reshape(-1, 1)).flatten()
transformer_preds = target_scaler.inverse_transform(transformer_model.predict(X_test, verbose=0).reshape(-1, 1)).flatten()

metrics_df = pd.DataFrame([
    {
        "Model": "LSTM",
        "MAE": mean_absolute_error(actual, lstm_preds),
        "RMSE": np.sqrt(mean_squared_error(actual, lstm_preds)),
        "R2": r2_score(actual, lstm_preds)
    },
    {
        "Model": "Transformer",
        "MAE": mean_absolute_error(actual, transformer_preds),
        "RMSE": np.sqrt(mean_squared_error(actual, transformer_preds)),
        "R2": r2_score(actual, transformer_preds)
    }
])

metrics_df.to_csv(PIPELINE_DIRS["results"] / "temporal_model_comparison.csv", index=False)
print("\n📊 Model Benchmark:")
print(metrics_df.round(4).to_string(index=False))

# Export Predictions
predictions_export = pd.DataFrame({
    "timestamp": test_timestamps,
    "actual_water_level_m": actual,
    "lstm_predicted_m": lstm_preds,
    "transformer_predicted_m": transformer_preds
})
predictions_export.to_csv(PIPELINE_DIRS["results"] / "water_level_predictions.csv", index=False)


# ==============================================================================
# 5. VISUALIZATION EXPORT
# ==============================================================================
limit = min(300, len(actual))
plt.figure(figsize=(14, 5))
plt.plot(actual[:limit], label="Ground Truth (Actual Level)", color="black", alpha=0.7)
plt.plot(lstm_preds[:limit], label="LSTM Prediction (+3h)", linestyle="--", color="blue")
plt.plot(transformer_preds[:limit], label="Transformer Prediction (+3h)", linestyle=":", color="red")
plt.title("River Water Level 3-Hour Forecast Comparison")
plt.xlabel("Timeline Sequence")
plt.ylabel("Water Level (m)")
plt.legend()
plt.tight_layout()
plt.savefig(PIPELINE_DIRS["results"] / "water_level_prediction_comparison.png", dpi=200)
plt.close()


# ==============================================================================
# 6. HUGGING FACE BENCHMARK INGESTION (VISUAL SATELLITE ENGINE)
# ==============================================================================
print("\n" + "=" * 80)
print("🛰️ STAGE: INGESTING SATELLITE BENCHMARK (SDSU MIDWEST 2019)")
print("=" * 80)

cnn_trained = False
cnn_accuracy = None

try:
    from datasets import load_dataset
    dataset_hf = load_dataset("youngsun05/SDSU_MidWest_Flood_2019")
    
    flood_count, nonflood_count = 0, 0
    MAX_PER_CLASS = 500

    for item in dataset_hf["train"]:
        if flood_count >= MAX_PER_CLASS and nonflood_count >= MAX_PER_CLASS:
            break

        img = item["image"].convert("RGB")
        mask = np.array(item["mask"].convert("L"))

        flood_ratio = np.sum(mask > 0) / mask.size

        # Rule: >1% mask coverage constitutes flooded sector
        if flood_ratio > 0.01 and flood_count < MAX_PER_CLASS:
            img.save(PIPELINE_DIRS["image_flood"] / f"flood_{flood_count:04d}.jpg")
            flood_count += 1
        elif flood_ratio <= 0.01 and nonflood_count < MAX_PER_CLASS:
            img.save(PIPELINE_DIRS["image_nonflood"] / f"nonflood_{nonflood_count:04d}.jpg")
            nonflood_count += 1

    print(f"✅ Ingested Imagery: {flood_count} Flooded, {nonflood_count} Non-Flooded")

    if flood_count >= 10 and nonflood_count >= 10:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.20,
            horizontal_flip=True,
            rotation_range=10,
            zoom_range=0.10
        )

        train_gen = datagen.flow_from_directory(
            str(BUILD_DIR / "image_data"),
            target_size=(128, 128),
            batch_size=16,
            class_mode="binary",
            subset="training",
            shuffle=True,
            seed=SEED
        )

        val_gen = datagen.flow_from_directory(
            str(BUILD_DIR / "image_data"),
            target_size=(128, 128),
            batch_size=16,
            class_mode="binary",
            subset="validation",
            shuffle=False,
            seed=SEED
        )

        cnn_model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid")
        ])

        cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        cnn_model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

        val_loss, cnn_accuracy = cnn_model.evaluate(val_gen, verbose=0)
        cnn_model.save(PIPELINE_DIRS["models"] / "flood_detection_cnn.keras")
        cnn_trained = True
        print(f"✅ CNN Training Completed | Val Accuracy: {cnn_accuracy:.4f}")

except Exception as e:
    print(f"⚠️ Visual dataset pipeline bypassed: {e}")


# ==============================================================================
# 7. METADATA REGISTRATION & ZIP PACKAGING
# ==============================================================================
print("\n" + "=" * 80)
print("📦 STAGE: COMPILING FINAL DATASET PACKAGE")
print("=" * 80)

# Create Master Catalog Structure
sub_dirs = [
    "01_SATELLITE_FLOOD_DATASET",
    "02_RAINFALL_DATASET",
    "03_RIVER_WATER_LEVEL_DATASET",
    "04_MASTER_DATASET",
    "05_METADATA"
]
for sub in sub_dirs:
    (PACKAGE_DIR / sub).mkdir(parents=True, exist_ok=True)

# Copy Core Artifacts
if os.path.exists(PIPELINE_DIRS["processed"] / "Master_Dataset_Cleaned.csv"):
    shutil.copy2(
        PIPELINE_DIRS["processed"] / "Master_Dataset_Cleaned.csv",
        PACKAGE_DIR / "04_MASTER_DATASET" / "Master_Dataset.csv"
    )

# Write Metadata Manifest
metadata_summary = {
    "temporal_models": {
        "status": "COMPLETED",
        "features": engineered_features,
        "lookback": LOOKBACK,
        "horizon": FORECAST_HOURS,
        "lstm_rmse": float(metrics_df.loc[metrics_df['Model']=='LSTM', 'RMSE'].values[0]),
        "transformer_rmse": float(metrics_df.loc[metrics_df['Model']=='Transformer', 'RMSE'].values[0])
    },
    "visual_model": {
        "status": "COMPLETED" if cnn_trained else "PENDING_OFFICIAL_MASKS",
        "cnn_accuracy": float(cnn_accuracy) if cnn_accuracy else None
    },
    "data_integrity": {
        "data_leakage_prevented": True,
        "chronological_split": True,
        "zero_fake_labels": True
    }
}

with open(PACKAGE_DIR / "05_METADATA" / "STAGE2_METADATA.json", "w") as f:
    json.dump(metadata_summary, f, indent=4)

# Create Output Archive
final_zip_target = "/content/STAGE_02_COMPLETE_DATASET"
if os.path.exists(f"{final_zip_target}.zip"):
    os.remove(f"{final_zip_target}.zip")

shutil.make_archive(final_zip_target, "zip", str(PACKAGE_DIR))
print(f"🎉 Pipeline Execution Complete!")
print(f"📁 Package Archive Exported: {final_zip_target}.zip")
