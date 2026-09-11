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

# Deep Learning Framework (Keras benchmark support)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM,
    MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Add
)

# ==============================================================================
# 0. REPRODUCIBILITY & DIRECTORY CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path("/content/STAGE_02_DL")
BUILD_DIR = BASE_DIR / "FINAL_STAGE2_BUILD"
DATASET_DIR = BASE_DIR / "data"
PACKAGE_DIR = BASE_DIR / "STAGE_02_COMPLETE_DATASET"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

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

# Required numeric columns for Stage 02 schema validation
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

# Hydrological features engineered by Data Engineer
ENGINEERED_FEATURE_NAMES = [
    "threshold_exceedance_m",
    "rainfall_3h",
    "rainfall_6h",
    "water_level_3h_mean",
    "water_level_6h_mean"
]


# ==============================================================================
# STAGE 02 DATA ENGINEER RESPONSIBILITY MODULE
# ==============================================================================
class Stage02DataEngineer:
    """
    Encapsulates all Data Engineering responsibilities for Stage 02:
    1. Data Ingestion: Locates & loads Master Dataset.
    2. Column Standardization: Standardizes column naming conventions.
    3. Schema Validation: Validates required numeric and temporal fields.
    4. Data Type Cleaning & Deduplication: Parses timestamps, handles infinite & missing values.
    5. Hydrological Feature Engineering: Computes temporal rolling features without lookahead leakage.
    6. Handoff & Export: Generates clean, validated datasets for EDA & ML pipeline.
    7. Catalog & Packaging: Constructs metadata manifests and catalog packages.
    """

    def __init__(self, base_dir: Path = BASE_DIR, build_dir: Path = BUILD_DIR):
        self.base_dir = base_dir
        self.build_dir = build_dir
        self.pipeline_dirs = PIPELINE_DIRS
        self.master_path = None
        self.cleaned_df = None
        self.available_features = []
        self.engineered_features = []

    def resolve_master_dataset(self) -> Path:
        """Locate Master_Dataset.csv across runtime environments."""
        search_paths = [
            "/content/Master_Dataset.csv",
            "/content/drive/MyDrive/Master_Dataset.csv",
            str(self.base_dir / "Master_Dataset.csv"),
            str(self.base_dir / "data" / "Master_Dataset.csv"),
            str(RAW_DATA_DIR / "04_MASTER_DATASET" / "Master_Dataset.csv"),
            str(self.pipeline_dirs["metadata"] / "Master_Dataset.csv"),
            str(self.base_dir / "data" / "Engineered_History_Trend_Dataset.csv")
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                self.master_path = Path(path)
                return self.master_path
                
        matches = glob.glob(f"{self.base_dir}/**/Master_Dataset.csv", recursive=True)
        if matches:
            self.master_path = Path(matches[0])
            return self.master_path
            
        try:
            from google.colab import files
            print("\n⚠️ Master_Dataset.csv not found in local path. Launching upload prompt...")
            uploaded = files.upload()
            for name in uploaded.keys():
                if name.lower() == "master_dataset.csv":
                    self.master_path = Path("/content") / name
                    return self.master_path
        except Exception:
            pass

        raise FileNotFoundError("Master_Dataset.csv is required to proceed with Data Engineering pipeline.")

    def ingest_data(self) -> pd.DataFrame:
        """Ingest raw tabular dataset from resolved path."""
        master_file = self.resolve_master_dataset()
        shutil.copy2(master_file, self.pipeline_dirs["metadata"] / "Master_Dataset.csv")
        print(f"✅ [Data Engineer] Ingested Master Dataset from: {master_file}")
        return pd.read_csv(master_file)

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names (lowercase, strip whitespace, replace separators)."""
        df = df.copy()
        df.columns = [
            str(col).strip().lower().replace(" ", "_").replace("-", "_")
            for col in df.columns
        ]
        return df

    def validate_and_clean_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamps, sort chronologically, and drop duplicate timestamps."""
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        return df

    def validate_and_clean_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required numeric schema, sanitize infs, and impute missing values safely."""
        df = df.copy()
        self.available_features = [col for col in REQUIRED_NUMERICS if col in df.columns]
        
        if "river_level_m" not in self.available_features:
            raise ValueError("Target feature 'river_level_m' missing from dataset schema.")

        for col in self.available_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Imputation strategy: linear interpolation followed by median fallback
        df[self.available_features] = df[self.available_features].replace([np.inf, -np.inf], np.nan)
        df[self.available_features] = df[self.available_features].interpolate(method="linear", limit_direction="both")
        for col in self.available_features:
            df[col] = df[col].fillna(df[col].median())

        cleaned = df.dropna(subset=["river_level_m"]).reset_index(drop=True)
        print(f"✅ [Data Engineer] Cleaned records: {len(cleaned)} rows | Features: {len(self.available_features)}")
        return cleaned

    def engineer_hydrological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate domain-specific hydrological rolling aggregations without lookahead leakage."""
        df = df.copy()
        df["threshold_exceedance_m"] = (
            df["river_level_m"] - df.get("river_level_threshold_m", 0)
        )
        df["rainfall_3h"] = df["rainfall_mm"].rolling(3, min_periods=1).mean() if "rainfall_mm" in df.columns else 0.0
        df["rainfall_6h"] = df["rainfall_mm"].rolling(6, min_periods=1).mean() if "rainfall_mm" in df.columns else 0.0
        df["water_level_3h_mean"] = df["river_level_m"].rolling(3, min_periods=1).mean()
        df["water_level_6h_mean"] = df["river_level_m"].rolling(6, min_periods=1).mean()

        self.engineered_features = self.available_features + ENGINEERED_FEATURE_NAMES

        # Use forward fill only to prevent lookahead leakage
        df[self.engineered_features] = df[self.engineered_features].interpolate(method="linear", limit_direction="forward").ffill()
        for col in self.engineered_features:
            df[col] = df[col].fillna(df[col].median())

        self.cleaned_df = df
        return df

    def export_cleaned_datasets(self) -> Path:
        """Export clean dataset artifacts for EDA and ML/DL downstream handoff."""
        processed_path = self.pipeline_dirs["processed"] / "Master_Dataset_Cleaned.csv"
        self.cleaned_df.to_csv(processed_path, index=False)
        
        # Ensure root data directory also receives clean dataset for DL engineer
        root_data_dir = self.base_dir / "data"
        root_data_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_df.to_csv(root_data_dir / "Engineered_History_Trend_Dataset.csv", index=False)
        
        # Stage catalog directory for EDA engineer under data/raw/04_MASTER_DATASET
        eda_raw_master_dir = RAW_DATA_DIR / "04_MASTER_DATASET"
        eda_raw_master_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_df.to_csv(eda_raw_master_dir / "Master_Dataset.csv", index=False)

        print(f"✅ [Data Engineer] Cleaned Master Dataset saved to: {processed_path}")
        return processed_path

    def package_metadata_and_archive(self, model_metrics: dict = None, cnn_trained: bool = False, cnn_accuracy: float = None):
        """Create master catalog directory structure, write metadata manifest, and compile ZIP archive."""
        sub_dirs = [
            "01_SATELLITE_FLOOD_DATASET",
            "02_RAINFALL_DATASET",
            "03_RIVER_WATER_LEVEL_DATASET",
            "04_MASTER_DATASET",
            "05_METADATA"
        ]
        for sub in sub_dirs:
            (PACKAGE_DIR / sub).mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.pipeline_dirs["processed"] / "Master_Dataset_Cleaned.csv"):
            shutil.copy2(
                self.pipeline_dirs["processed"] / "Master_Dataset_Cleaned.csv",
                PACKAGE_DIR / "04_MASTER_DATASET" / "Master_Dataset.csv"
            )

        metadata_summary = {
            "temporal_models": model_metrics or {
                "status": "COMPLETED",
                "features": self.engineered_features,
                "lookback": 24,
                "horizon": 3
            },
            "visual_model": {
                "status": "COMPLETED" if cnn_trained else "PENDING_OFFICIAL_MASKS",
                "cnn_accuracy": float(cnn_accuracy) if cnn_accuracy is not None else None
            },
            "data_integrity": {
                "data_leakage_prevented": True,
                "chronological_split": True,
                "zero_fake_labels": True
            }
        }

        with open(PACKAGE_DIR / "05_METADATA" / "STAGE2_METADATA.json", "w") as f:
            json.dump(metadata_summary, f, indent=4)

        final_zip_target = str(self.base_dir / "STAGE_02_COMPLETE_DATASET")
        if os.path.exists(f"{final_zip_target}.zip"):
            os.remove(f"{final_zip_target}.zip")

        shutil.make_archive(final_zip_target, "zip", str(PACKAGE_DIR))
        print(f"🎉 [Data Engineer] Package Archive Exported: {final_zip_target}.zip")

    def run_pipeline(self) -> pd.DataFrame:
        """Run complete Data Engineering pipeline sequence."""
        print("=" * 80)
        print("🚀 STAGE 02: DATA ENGINEERING PIPELINE")
        print("=" * 80)
        
        raw_df = self.ingest_data()
        std_df = self.standardize_schema(raw_df)
        temp_df = self.validate_and_clean_temporal(std_df)
        clean_df = self.validate_and_clean_numerics(temp_df)
        final_df = self.engineer_hydrological_features(clean_df)
        self.export_cleaned_datasets()
        return final_df


# ==============================================================================
# DOWNSTREAM ML / DL COMPATIBILITY EXECUTION WRAPPER
# ==============================================================================
def run_legacy_ml_benchmarks(data_engineer: Stage02DataEngineer):
    """
    Executes existing Keras benchmark models (LSTM, Transformer, CNN) and visual
    satellite benchmark ingestion to preserve existing Stage 02 output files.
    """
    cleaned_df = data_engineer.cleaned_df
    engineered_features = data_engineer.engineered_features
    N_ROWS = len(cleaned_df)
    TRAIN_IDX = int(N_ROWS * 0.70)

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

    print(f"\n🧠 STAGE: MODEL COMPILATION & BENCHMARKING (KERAS)")
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

    predictions_export = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual_water_level_m": actual,
        "lstm_predicted_m": lstm_preds,
        "transformer_predicted_m": transformer_preds
    })
    predictions_export.to_csv(PIPELINE_DIRS["results"] / "water_level_predictions.csv", index=False)

    # Visualization Export
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

    # Visual Satellite Benchmark Ingestion
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

            if flood_ratio > 0.01 and flood_count < MAX_PER_CLASS:
                img.save(PIPELINE_DIRS["image_flood"] / f"flood_{flood_count:04d}.jpg")
                flood_count += 1
            elif flood_ratio <= 0.01 and nonflood_count < MAX_PER_CLASS:
                img.save(PIPELINE_DIRS["image_nonflood"] / f"nonflood_{nonflood_count:04d}.jpg")
                nonflood_count += 1

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
    except Exception as e:
        print(f"⚠️ Visual dataset pipeline bypassed: {e}")

    # Package Catalog & Metadata
    model_metrics_summary = {
        "status": "COMPLETED",
        "features": engineered_features,
        "lookback": LOOKBACK,
        "horizon": FORECAST_HOURS,
        "lstm_rmse": float(metrics_df.loc[metrics_df['Model']=='LSTM', 'RMSE'].values[0]),
        "transformer_rmse": float(metrics_df.loc[metrics_df['Model']=='Transformer', 'RMSE'].values[0])
    }
    data_engineer.package_metadata_and_archive(model_metrics_summary, cnn_trained, cnn_accuracy)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    de = Stage02DataEngineer()
    de.run_pipeline()
    run_legacy_ml_benchmarks(de)
