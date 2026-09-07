"""Evaluation-only pipeline for the trained Stage02 models.

This module never calls a training function. It reconstructs the deterministic
test partitions used by ``03_dl_engineer.py``, loads the saved checkpoints,
runs inference, and writes evaluation artifacts below ``data/outputs``.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	classification_report,
	confusion_matrix,
	f1_score,
	mean_absolute_error,
	mean_squared_error,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torchvision import datasets as vision_datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
RANDOM_STATE = 42
BATCH_SIZE = 32


def load_training_module():
	"""Load model classes and preprocessing helpers without executing training."""
	spec = importlib.util.spec_from_file_location(
		"stage02_dl_engineer", BASE_DIR / "03_dl_engineer.py"
	)
	module = importlib.util.module_from_spec(spec)
	assert spec.loader is not None
	spec.loader.exec_module(module)
	return module


def model_size_mb(path: Path) -> float:
	"""Return an on-disk model size in megabytes."""
	return round(path.stat().st_size / (1024 * 1024), 4)


def save_matrix(y_true, y_pred, labels: list, path: Path, title: str) -> None:
	"""Save a confusion matrix for a classification evaluation."""
	figure, axis = plt.subplots(figsize=(6, 5))
	sns.heatmap(
		confusion_matrix(y_true, y_pred, labels=labels), annot=True, fmt="d",
		cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis,
	)
	axis.set_xlabel("Predicted")
	axis.set_ylabel("Actual")
	axis.set_title(title)
	figure.tight_layout()
	figure.savefig(path, dpi=150)
	plt.close(figure)


def classification_result(y_true, y_pred, labels: list[str]) -> dict:
	"""Create consistent classification metrics and per-class results."""
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
		"recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		"class_distribution": pd.Series(y_true).value_counts().to_dict(),
		"classification_report": classification_report(
			y_true, y_pred, labels=labels, output_dict=True, zero_division=0
		),
	}


def evaluate_cnn(dl) -> dict:
	"""Evaluate the saved CNN on its deterministic, unseen test images."""
	image_dir = RAW_DIR / "Flood_Image_Dataset"
	checkpoint = torch.load(MODEL_DIR / "disaster_cnn.pt", map_location=dl.DEVICE)
	normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	evaluation_transform = transforms.Compose([
		transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
		transforms.ToTensor(), normalize,
	])
	dataset = vision_datasets.ImageFolder(image_dir, transform=evaluation_transform)
	indices = np.arange(len(dataset))
	train_indices, remainder = train_test_split(
		indices, test_size=0.30, random_state=RANDOM_STATE,
		stratify=dataset.targets,
	)
	_, test_indices = train_test_split(
		remainder, test_size=0.50, random_state=RANDOM_STATE,
		stratify=np.asarray(dataset.targets)[remainder],
	)
	loader = DataLoader(Subset(dataset, test_indices), BATCH_SIZE, shuffle=False)
	model = dl.DisasterCNN(len(checkpoint["classes"])).to(dl.DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	all_true, all_pred, all_confidence, total_time = [], [], [], 0.0
	rows = []
	with torch.no_grad():
		for images, labels in loader:
			start = time.perf_counter()
			probabilities = torch.softmax(model(images.to(dl.DEVICE)), dim=1)
			total_time += time.perf_counter() - start
			predictions = probabilities.argmax(1).cpu().tolist()
			confidence = probabilities.max(1).values.cpu().tolist()
			all_true.extend(labels.tolist())
			all_pred.extend(predictions)
			all_confidence.extend(confidence)
	for index, actual, predicted, confidence in zip(test_indices, all_true, all_pred, all_confidence):
		path, _ = dataset.samples[index]
		rows.append({
			"file": str(Path(path).relative_to(BASE_DIR)).replace("\\", "/"),
			"actual": checkpoint["classes"][actual],
			"predicted": checkpoint["classes"][predicted],
			"confidence": confidence,
			"correct": actual == predicted,
		})
	predictions = pd.DataFrame(rows)
	predictions.to_csv(OUTPUT_DIR / "cnn_test_predictions.csv", index=False)
	labels = list(range(len(checkpoint["classes"])))
	metrics = classification_result(all_true, all_pred, labels)
	flooded_index = checkpoint["classes"].index("flooded")
	true_flooded = np.asarray(all_true) == flooded_index
	# AUC is reported for flooded as the positive class, not for accuracy.
	probability_flooded = 1 - np.asarray(all_confidence) if False else None
	# Re-run probabilities only for the positive-class AUC to retain scores.
	positive_scores = []
	with torch.no_grad():
		for images, _ in loader:
			positive_scores.extend(torch.softmax(model(images.to(dl.DEVICE)), dim=1)[:, flooded_index].cpu().tolist())
	if len(np.unique(true_flooded)) == 2:
		metrics["roc_auc_flooded"] = float(roc_auc_score(true_flooded, positive_scores))
		metrics["pr_auc_flooded"] = float(average_precision_score(true_flooded, positive_scores))
	metrics.update({
		"classes": checkpoint["classes"],
		"test_samples": len(test_indices),
		"inference_seconds": total_time,
		"average_latency_ms": total_time / len(test_indices) * 1000,
		"throughput_samples_per_second": len(test_indices) / total_time,
		"model_size_mb": model_size_mb(MODEL_DIR / "disaster_cnn.pt"),
		"training_history_available": False,
	})
	save_matrix(all_true, all_pred, labels, OUTPUT_DIR / "evaluation_cnn_confusion_matrix.png", "CNN unseen-test confusion matrix")
	return metrics


def evaluate_lstm(dl) -> dict:
	"""Evaluate the saved LSTM on the chronological unseen test partition."""
	series = dl.load_river_series()
	values = series["water_level"].to_numpy(dtype=np.float32)
	train_end, val_end = int(len(values) * 0.70), int(len(values) * 0.85)
	scaler = joblib.load(MODEL_DIR / "water_level_scaler.joblib")
	scaled = scaler.transform(values[:, None]).ravel()
	_, test_y = dl.make_sequences(scaled[val_end - 24:], 24)
	test_x, _ = dl.make_sequences(scaled[val_end - 24:], 24)
	checkpoint = torch.load(MODEL_DIR / "water_level_lstm.pt", map_location=dl.DEVICE)
	model = dl.WaterLevelLSTM().to(dl.DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	inputs = torch.tensor(test_x, dtype=torch.float32).to(dl.DEVICE)
	start = time.perf_counter()
	with torch.no_grad():
		predicted_scaled = model(inputs).cpu().numpy()
	inference_seconds = time.perf_counter() - start
	predicted = scaler.inverse_transform(predicted_scaled[:, None]).ravel()
	actual = scaler.inverse_transform(test_y[:, None]).ravel()
	metrics = {
		"mae": float(mean_absolute_error(actual, predicted)),
		"mse": float(mean_squared_error(actual, predicted)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
		"test_samples": len(actual),
		"inference_seconds": inference_seconds,
		"average_latency_ms": inference_seconds / len(actual) * 1000,
		"throughput_samples_per_second": len(actual) / inference_seconds,
		"model_size_mb": model_size_mb(MODEL_DIR / "water_level_lstm.pt"),
		"lookback": int(checkpoint["lookback"]),
		"dataset": str(dl.ENGINEERED_HISTORY_PATH.relative_to(dl.BASE_DIR)).replace("\\", "/"),
		"series_rows": int(len(series)),
		"training_history_available": False,
	}
	pd.DataFrame({"actual": actual, "predicted": predicted, "absolute_error": np.abs(actual - predicted)}).to_csv(
		OUTPUT_DIR / "lstm_test_predictions.csv", index=False
	)
	figure, axis = plt.subplots(figsize=(11, 4))
	axis.plot(actual[:500], label="actual")
	axis.plot(predicted[:500], label="predicted")
	axis.set_title("LSTM unseen-test predictions")
	axis.legend()
	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "evaluation_lstm_actual_vs_predicted.png", dpi=150)
	plt.close(figure)
	return metrics


def evaluate_robustness(dl) -> dict:
	"""Measure reasonable perturbations without changing the official test scores."""
	series = dl.load_river_series()
	values = series["water_level"].to_numpy(dtype=np.float32)
	val_end = int(len(values) * 0.85)
	scaler = joblib.load(MODEL_DIR / "water_level_scaler.joblib")
	scaled = scaler.transform(values[:, None]).ravel()
	test_x, test_y = dl.make_sequences(scaled[val_end - 24:], 24)
	checkpoint = torch.load(MODEL_DIR / "water_level_lstm.pt", map_location=dl.DEVICE)
	model = dl.WaterLevelLSTM().to(dl.DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	perturbed = test_x + np.random.default_rng(RANDOM_STATE).normal(0, 0.05, test_x.shape)
	with torch.no_grad():
		prediction = model(torch.tensor(perturbed, dtype=torch.float32).to(dl.DEVICE)).cpu().numpy()
	actual = scaler.inverse_transform(test_y[:, None]).ravel()
	predicted = scaler.inverse_transform(prediction[:, None]).ravel()
	return {
		"perturbation": "Gaussian noise, scaled-space standard deviation 0.05",
		"mae": float(mean_absolute_error(actual, predicted)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
	}


def build_report(cnn: dict, lstm: dict, robustness: dict) -> None:
	"""Write an interpretable Markdown evaluation report."""
	cnn_recall = cnn["classification_report"]["0"]["recall"]
	report = f"""# Stage02 Deep Learning Evaluation Report

## Workflow

Raw test data -> saved model -> inference -> predictions -> metrics -> error analysis -> recommendation.

The CNN test partition contains {cnn['test_samples']} unseen images. The LSTM test partition contains
{lstm['test_samples']} chronological one-step sequences. The deterministic 70/15/15 split is reconstructed
from the raw data with seed 42; split manifests were not saved by the DL training script, so exact replay
depends on the raw files and folder ordering remaining unchanged.

## CNN

| Metric | Result |
| --- | ---: |
| Accuracy | {cnn['accuracy']:.4f} |
| Macro precision | {cnn['precision_macro']:.4f} |
| Macro recall | {cnn['recall_macro']:.4f} |
| Macro F1 | {cnn['f1_macro']:.4f} |
| Flooded ROC-AUC | {cnn.get('roc_auc_flooded', float('nan')):.4f} |
| Flooded PR-AUC | {cnn.get('pr_auc_flooded', float('nan')):.4f} |
| Average latency (ms/sample) | {cnn['average_latency_ms']:.4f} |

The confusion matrix and `cnn_test_predictions.csv` show the error pattern. The flooded class recall is
{cnn_recall:.4f}; accuracy alone is therefore misleading because the test set is imbalanced.

## LSTM

| Metric | Result |
| --- | ---: |
| MAE | {lstm['mae']:.4f} |
| MSE | {lstm['mse']:.4f} |
| RMSE | {lstm['rmse']:.4f} |
| Average latency (ms/sample) | {lstm['average_latency_ms']:.4f} |

LSTM errors are stored in `lstm_test_predictions.csv`; larger absolute errors identify difficult periods.

## Robustness

The LSTM perturbation test used a small scaled-space Gaussian noise standard deviation of 0.05.
Its perturbed MAE was {robustness['mae']:.4f} and RMSE was {robustness['rmse']:.4f}. These are sensitivity
measurements, not replacements for the official test metrics.

## Generalization and model comparison

The saved checkpoints do not contain epoch-by-epoch histories, so a numeric training-versus-validation
generalization gap cannot be recomputed without retraining. Existing training curves are preserved in
`cnn_training_curves.png`; the evaluation pipeline does not alter the models.

CNN and LSTM solve different tasks and must not be ranked by accuracy against regression error. The CNN
produces a visual flood class, while the LSTM forecasts a continuous water level. Both outputs are useful
signals for a later response component, but this evaluation provides no evidence that one replaces the other.

## Artifacts

- `evaluation_cnn_confusion_matrix.png`
- `evaluation_lstm_actual_vs_predicted.png`
- `cnn_test_predictions.csv`
- `lstm_test_predictions.csv`
- `evaluation_metrics.json`
- `evaluation_comparison.csv`
"""
	(OUTPUT_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
	"""Run evaluation only and write the final report."""
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	dl = load_training_module()
	cnn = evaluate_cnn(dl)
	lstm = evaluate_lstm(dl)
	robustness = evaluate_robustness(dl)
	comparison = pd.DataFrame([
		{"metric": "accuracy", "cnn": cnn["accuracy"], "lstm": None, "applicable": "CNN classification only"},
		{"metric": "macro_f1", "cnn": cnn["f1_macro"], "lstm": None, "applicable": "CNN classification only"},
		{"metric": "mae", "cnn": None, "lstm": lstm["mae"], "applicable": "LSTM regression only"},
		{"metric": "rmse", "cnn": None, "lstm": lstm["rmse"], "applicable": "LSTM regression only"},
		{"metric": "average_latency_ms", "cnn": cnn["average_latency_ms"], "lstm": lstm["average_latency_ms"], "applicable": "Both, different workloads"},
		{"metric": "model_size_mb", "cnn": cnn["model_size_mb"], "lstm": lstm["model_size_mb"], "applicable": "Both"},
	])
	comparison.to_csv(OUTPUT_DIR / "evaluation_comparison.csv", index=False)
	results = {"cnn": cnn, "lstm": lstm, "robustness": robustness}
	(OUTPUT_DIR / "evaluation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
	build_report(cnn, lstm, robustness)
	print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
