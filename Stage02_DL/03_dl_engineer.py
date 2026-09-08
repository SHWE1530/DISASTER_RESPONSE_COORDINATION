"""Train and evaluate the Stage02 deep-learning models.

This module uses only data under ``Stage02_DL/data/raw``:

* ``Flood_Image_Dataset`` trains a binary flooded/unflooded CNN.
* ``03_RIVER_WATER_LEVEL_DATASET`` trains a chronological LSTM forecaster.
* ``04_MASTER_DATASET/Master_Dataset.csv`` trains the labelled zone-risk model.

Run from the repository root with::

    python Stage02_DL/03_dl_engineer.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	mean_absolute_error,
	mean_squared_error,
	precision_score,
	recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets as vision_datasets
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
ENGINEERED_HISTORY_PATH = BASE_DIR / "data" / "Engineered_History_Trend_Dataset.csv"
RANDOM_STATE = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LOOKBACK = 72
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = RANDOM_STATE) -> None:
	"""Make model training repeatable where the libraries permit it."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def ensure_directories() -> None:
	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def classification_metrics(y_true: Iterable, y_pred: Iterable, labels: list[str]) -> dict:
	"""Return stable classification metrics and a serializable report."""
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
		"recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		"labels": labels,
		"classification_report": classification_report(
			y_true, y_pred, labels=labels, output_dict=True, zero_division=0
		),
	}


def save_confusion_matrix(y_true, y_pred, labels: list[str], path: Path, title: str) -> None:
	"""Save a labelled confusion-matrix figure."""
	figure, axis = plt.subplots(figsize=(6, 5))
	sns.heatmap(
		confusion_matrix(y_true, y_pred, labels=labels),
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=labels,
		yticklabels=labels,
		ax=axis,
	)
	axis.set_xlabel("Predicted")
	axis.set_ylabel("Actual")
	axis.set_title(title)
	figure.tight_layout()
	figure.savefig(path, dpi=150)
	plt.close(figure)


class DisasterCNN(nn.Module):
	"""Small baseline CNN suitable for the raw flood images."""

	def __init__(self, classes: int = 2):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
		)
		self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, classes))

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.classifier(self.features(inputs))


def run_cnn_epoch(
	model,
	loader,
	optimizer=None,
	class_weights: torch.Tensor | None = None,
) -> tuple[float, list[int], list[int]]:
	"""Run one training or evaluation epoch."""
	training = optimizer is not None
	model.train(training)
	loss_function = nn.CrossEntropyLoss(
		weight=class_weights.to(DEVICE) if class_weights is not None else None
	)
	total_loss = 0.0
	y_true, y_pred = [], []
	for images, labels in loader:
		images, labels = images.to(DEVICE), labels.to(DEVICE)
		if training:
			optimizer.zero_grad()
		outputs = model(images)
		loss = loss_function(outputs, labels)
		if training:
			loss.backward()
			optimizer.step()
		total_loss += loss.item() * len(labels)
		y_true.extend(labels.cpu().tolist())
		y_pred.extend(outputs.argmax(1).cpu().tolist())
	return total_loss / len(loader.dataset), y_true, y_pred


def train_cnn() -> dict:
	"""Train, evaluate, and save the flooded/unflooded image classifier."""
	image_dir = BASE_DIR / "data" / "images"
	if not image_dir.exists():
		raise FileNotFoundError(f"CNN image directory not found: {image_dir}")
	normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	base_dataset = vision_datasets.ImageFolder(image_dir)
	labels = base_dataset.classes
	indices = np.arange(len(base_dataset))
	train_indices, remainder = train_test_split(
		indices, test_size=0.30, random_state=RANDOM_STATE,
		stratify=base_dataset.targets,
	)
	val_indices, test_indices = train_test_split(
		remainder, test_size=0.50, random_state=RANDOM_STATE,
		stratify=np.asarray(base_dataset.targets)[remainder],
	)
	train_targets = np.asarray(base_dataset.targets)[train_indices]
	class_counts = np.bincount(train_targets, minlength=len(labels)).astype(np.float32)
	
	# User explicitly wanted normalized inverse class counts for CrossEntropyLoss weight
	weights_inv = 1.0 / np.maximum(class_counts, 1)
	class_weights = weights_inv / weights_inv.sum()
	
	sample_weights = class_weights[train_targets]
	train_sampler = WeightedRandomSampler(
		torch.as_tensor(sample_weights, dtype=torch.double),
		num_samples=len(sample_weights),
		replacement=True,
	)
	
	train_dataset = vision_datasets.ImageFolder(
		image_dir, transform=transforms.Compose([
			transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(15),
			transforms.ColorJitter(brightness=0.2, contrast=0.2),
			transforms.ToTensor(),
			normalize,
		])
	)
	
	eval_dataset = vision_datasets.ImageFolder(
		image_dir, transform=transforms.Compose([
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize,
		])
	)
	loaders = {
		"train": DataLoader(Subset(train_dataset, train_indices), BATCH_SIZE, sampler=train_sampler),
		"val": DataLoader(Subset(eval_dataset, val_indices), BATCH_SIZE),
		"test": DataLoader(Subset(eval_dataset, test_indices), BATCH_SIZE),
	}
	model = DisasterCNN(len(labels)).to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
	best_state, best_val_f1 = None, -1.0
	patience = 7
	patience_counter = 0
	
	for epoch in range(50):
		train_loss, train_true, train_pred = run_cnn_epoch(
			model, loaders["train"], optimizer, torch.tensor(class_weights, dtype=torch.float32)
		)
		val_loss, val_true, val_pred = run_cnn_epoch(model, loaders["val"])
		val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
		
		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["train_accuracy"].append(float(accuracy_score(train_true, train_pred)))
		history["val_accuracy"].append(float(accuracy_score(val_true, val_pred)))
		
		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
			patience_counter = 0
		else:
			patience_counter += 1
			if patience_counter >= patience:
				print(f"CNN Early stopping triggered at epoch {epoch}")
				break
				
	if best_state is not None:
		model.load_state_dict(best_state)
	test_loss, test_true, test_pred = run_cnn_epoch(model, loaders["test"])
	metrics = classification_metrics(test_true, test_pred, list(range(len(labels))))
	metrics.update({
		"test_loss": test_loss,
		"classes": labels,
		"device": str(DEVICE),
		"class_weights": class_weights.tolist(),
		"train_class_distribution": {
			labels[index]: int(count) for index, count in enumerate(class_counts)
		},
	})
	torch.save({
		"state_dict": model.state_dict(),
		"classes": labels,
		"image_size": IMAGE_SIZE,
		"class_weights": class_weights.tolist(),
	}, MODEL_DIR / "disaster_cnn.pt")
	save_confusion_matrix(test_true, test_pred, list(range(len(labels))), OUTPUT_DIR / "cnn_confusion_matrix.png", "CNN test confusion matrix")
	
	figure, axes = plt.subplots(1, 2, figsize=(11, 4))
	axes[0].plot(history["train_loss"], label="train")
	axes[0].plot(history["val_loss"], label="validation")
	axes[0].set_title("CNN loss")
	axes[1].plot(history["train_accuracy"], label="train")
	axes[1].plot(history["val_accuracy"], label="validation")
	axes[1].set_title("CNN accuracy")
	for axis in axes:
		axis.legend()
	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "cnn_training_curves.png", dpi=150)
	plt.close(figure)
	return metrics


def load_river_series() -> pd.DataFrame:
	"""Load the engineered hourly history-trend water-level series and create additional features."""
	if not ENGINEERED_HISTORY_PATH.exists():
		raise FileNotFoundError(
			f"Engineered history-trend dataset not found: {ENGINEERED_HISTORY_PATH}"
		)
	frame = pd.read_csv(ENGINEERED_HISTORY_PATH, low_memory=False)
	frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
	frame["water_level"] = pd.to_numeric(frame["river_level_m"], errors="coerce")
	
	df = (
		frame[["timestamp", "water_level"]]
		.dropna()
		.sort_values("timestamp")
		.drop_duplicates("timestamp")
		.reset_index(drop=True)
	)
	
	df["rolling_mean_6h"] = df["water_level"].rolling(window=6, min_periods=1).mean()
	df["rolling_std_6h"] = df["water_level"].rolling(window=6, min_periods=1).std().fillna(0.0)
	df["diff_t_1"] = df["water_level"].diff().fillna(0.0)
	
	return df


class SequenceDataset(Dataset):
	"""Dataset for LSTM inputs shaped as (samples, timesteps, features)."""
	def __init__(self, sequences: np.ndarray, targets: np.ndarray):
		self.sequences = torch.tensor(sequences, dtype=torch.float32)
		self.targets = torch.tensor(targets, dtype=torch.float32)

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):
		return self.sequences[index], self.targets[index]


class WaterLevelLSTM(nn.Module):
	"""Univariate/Multivariate water-level forecaster."""
	def __init__(self, input_size: int = 4):
		super().__init__()
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
		self.output = nn.Linear(64, 1)

	def forward(self, inputs):
		outputs, _ = self.lstm(inputs)
		return self.output(outputs[:, -1, :]).squeeze(-1)


def make_sequences(features: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
	"""Create chronological one-step forecasting windows. Target is water_level (col 0)."""
	sequences = np.asarray([features[index:index + lookback] for index in range(len(features) - lookback)])
	targets = features[lookback:, 0]
	return sequences, targets


def train_lstm() -> dict:
	"""Train and evaluate the water-level LSTM with temporal features."""
	series = load_river_series()
	feature_cols = ["water_level", "rolling_mean_6h", "rolling_std_6h", "diff_t_1"]
	values = series[feature_cols].to_numpy(dtype=np.float32)
	
	train_end = int(len(values) * 0.70)
	val_end = int(len(values) * 0.85)
	
	scaler = StandardScaler().fit(values[:train_end])
	scaled = scaler.transform(values)
	
	train_x, train_y = make_sequences(scaled[:train_end], LOOKBACK)
	val_x, val_y = make_sequences(scaled[train_end - LOOKBACK:val_end], LOOKBACK)
	test_x, test_y = make_sequences(scaled[val_end - LOOKBACK:], LOOKBACK)
	
	train_loader = DataLoader(SequenceDataset(train_x, train_y), BATCH_SIZE, shuffle=False)
	val_dataset = SequenceDataset(val_x, val_y)
	
	model = WaterLevelLSTM(input_size=4).to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
	loss_function = nn.MSELoss()
	
	history = {"train_loss": [], "val_loss": []}
	best_state, best_val = None, float("inf")
	patience = 10
	patience_counter = 0
	
	for epoch in range(100):
		model.train()
		train_total = 0.0
		for batch_x, batch_y in train_loader:
			batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
			optimizer.zero_grad()
			loss = loss_function(model(batch_x), batch_y)
			loss.backward()
			optimizer.step()
			train_total += loss.item() * len(batch_y)
			
		model.eval()
		with torch.no_grad():
			val_preds = model(val_dataset.sequences.to(DEVICE))
			val_loss = loss_function(val_preds, val_dataset.targets.to(DEVICE)).item()
			val_rmse = np.sqrt(val_loss)
			
		scheduler.step(val_rmse)
			
		history["train_loss"].append(train_total / len(train_loader.dataset))
		history["val_loss"].append(val_loss)
		
		if val_rmse < best_val:
			best_val = val_rmse
			best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
			patience_counter = 0
		else:
			patience_counter += 1
			if patience_counter >= patience:
				print(f"LSTM Early stopping triggered at epoch {epoch}")
				break
				
	if best_state is not None:
		model.load_state_dict(best_state)
		
	model.eval()
	with torch.no_grad():
		pred_scaled = model(torch.tensor(test_x, dtype=torch.float32).to(DEVICE)).cpu().numpy()
		
	# Inverse transform requires all 4 columns, we only have predicted col 0
	# Create dummy array
	dummy = np.zeros((len(pred_scaled), 4))
	dummy[:, 0] = pred_scaled
	predicted = scaler.inverse_transform(dummy)[:, 0]
	
	dummy[:, 0] = test_y
	actual = scaler.inverse_transform(dummy)[:, 0]
	
	metrics = {
		"mae": float(mean_absolute_error(actual, predicted)),
		"mse": float(mean_squared_error(actual, predicted)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
		"lookback": LOOKBACK,
		"series_rows": int(len(series)),
		"device": str(DEVICE),
		"dataset": str(ENGINEERED_HISTORY_PATH.relative_to(BASE_DIR)).replace("\\", "/"),
	}
	
	torch.save({"state_dict": model.state_dict(), "lookback": LOOKBACK, "input_size": 4}, MODEL_DIR / "water_level_lstm.pt")
	joblib.dump(scaler, MODEL_DIR / "water_level_scaler.joblib")
	
	figure, axis = plt.subplots(figsize=(11, 4))
	axis.plot(actual[:500], label="actual")
	axis.plot(predicted[:500], label="predicted")
	axis.set_title("Water-level LSTM test predictions")
	axis.legend()
	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "lstm_actual_vs_predicted.png", dpi=150)
	plt.close(figure)
	return metrics


def train_severity_model() -> dict:
	"""Train the labelled three-class zone-risk model from the master dataset."""
	path = RAW_DIR / "04_MASTER_DATASET" / "Master_Dataset.csv"
	df = pd.read_csv(path)
	target = "zone_risk"
	if target not in df or df[target].isna().any():
		raise ValueError("Master dataset must contain complete zone_risk labels")
	df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
	df["hour"] = df["timestamp"].dt.hour
	df["month"] = df["timestamp"].dt.month
	df["river_level_margin_m"] = df["river_level_m"] - df["river_level_threshold_m"]
	df = df.drop(columns=["timestamp"])
	features = [column for column in df.columns if column != target]
	categorical = [column for column in ["state", "district"] if column in features]
	numeric = [column for column in features if column not in categorical]
	preprocessor = ColumnTransformer([
		("numeric", StandardScaler(), numeric),
		("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
	])
	pipeline = Pipeline([
		("preprocessor", preprocessor),
		("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
	])
	ordered = df.reset_index(drop=True)
	train_end, val_end = int(len(ordered) * 0.70), int(len(ordered) * 0.85)
	pipeline.fit(ordered.loc[:train_end - 1, features], ordered.loc[:train_end - 1, target])
	val_pred = pipeline.predict(ordered.loc[train_end:val_end - 1, features])
	test_pred = pipeline.predict(ordered.loc[val_end:, features])
	labels = sorted(ordered[target].unique().tolist())
	metrics = classification_metrics(ordered.loc[val_end:, target], test_pred, labels)
	metrics["validation"] = classification_metrics(ordered.loc[train_end:val_end - 1, target], val_pred, labels)
	joblib.dump(pipeline, MODEL_DIR / "severity_model.joblib")
	save_confusion_matrix(ordered.loc[val_end:, target], test_pred, labels, OUTPUT_DIR / "severity_confusion_matrix.png", "Zone-risk test confusion matrix")
	return metrics


def predict_image(image_path: str | Path) -> dict:
	"""Load the saved CNN and predict one new flood image."""
	checkpoint = torch.load(MODEL_DIR / "disaster_cnn.pt", map_location=DEVICE)
	model = DisasterCNN(len(checkpoint["classes"])).to(DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	transform = transforms.Compose([
		transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
		transforms.ToTensor(), normalize,
	])
	with Image.open(image_path).convert("RGB") as image:
		input_tensor = transform(image).unsqueeze(0).to(DEVICE)
	with torch.no_grad():
		probabilities = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
	index = int(probabilities.argmax())
	return {"label": checkpoint["classes"][index], "confidence": float(probabilities[index])}


def make_features_for_inference(levels_list: list[float]) -> np.ndarray:
	s = pd.Series(levels_list)
	df = pd.DataFrame({"water_level": s})
	df["rolling_mean_6h"] = df["water_level"].rolling(6, min_periods=1).mean()
	df["rolling_std_6h"] = df["water_level"].rolling(6, min_periods=1).std().fillna(0.0)
	df["diff_t_1"] = df["water_level"].diff().fillna(0.0)
	return df.to_numpy(dtype=np.float32)


def forecast_water_level(recent_levels: Iterable[float]) -> float:
	"""Forecast one next water-level value from the saved LSTM and scaler."""
	return forecast_water_levels(recent_levels, 1)[0]


def forecast_water_levels(recent_levels: Iterable[float], horizon: int = 6) -> list[float]:
	"""Recursively forecast multiple future water-level steps."""
	if horizon < 1 or horizon > 168:
		raise ValueError("horizon must be between 1 and 168 steps")
	
	checkpoint = torch.load(MODEL_DIR / "water_level_lstm.pt", map_location=DEVICE)
	lookback = int(checkpoint["lookback"])
	input_size = checkpoint.get("input_size", 4)
	
	values = list(recent_levels)
	if len(values) < lookback:
		raise ValueError(f"At least {lookback} water-level values are required")
	
	scaler = joblib.load(MODEL_DIR / "water_level_scaler.joblib")
	model = WaterLevelLSTM(input_size=input_size).to(DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	
	forecasts = []
	with torch.no_grad():
		for _ in range(horizon):
			feats = make_features_for_inference(values[-lookback:])
			scaled_window = scaler.transform(feats)[None, :, :]
			
			next_scaled_val = model(torch.tensor(scaled_window, dtype=torch.float32).to(DEVICE)).cpu().item()
			
			dummy = np.zeros((1, input_size))
			dummy[0, 0] = next_scaled_val
			next_val = float(scaler.inverse_transform(dummy)[0, 0])
			
			forecasts.append(next_val)
			values.append(next_val)
			
	return forecasts


def predict_severity(features: pd.DataFrame) -> np.ndarray:
	"""Predict zone-risk labels using the saved master-dataset model."""
	pipeline = joblib.load(MODEL_DIR / "severity_model.joblib")
	prepared = features.copy()
	if "timestamp" in prepared:
		timestamp = pd.to_datetime(prepared.pop("timestamp"), dayfirst=True, errors="coerce")
		prepared["hour"] = timestamp.dt.hour
		prepared["month"] = timestamp.dt.month
	if "river_level_margin_m" not in prepared and {"river_level_m", "river_level_threshold_m"}.issubset(prepared.columns):
		prepared["river_level_margin_m"] = prepared["river_level_m"] - prepared["river_level_threshold_m"]
	return pipeline.predict(prepared)


def main() -> None:
	"""Run all trainable Stage02 components and save one result manifest."""
	seed_everything()
	ensure_directories()
	results = {
		"cnn": train_cnn(),
		"lstm": train_lstm(),
		"severity": train_severity_model(),
	}
	(OUTPUT_DIR / "dl_training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
	print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
