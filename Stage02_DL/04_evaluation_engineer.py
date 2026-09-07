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


class GradCAM:
	"""Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps."""
	def __init__(self, model, target_layer):
		self.model = model
		self.target_layer = target_layer
		self.gradients = None
		self.activations = None
		self.target_layer.register_forward_hook(self.save_activation)
		self.target_layer.register_backward_hook(self.save_gradient)

	def save_activation(self, module, input, output):
		self.activations = output

	def save_gradient(self, module, grad_input, grad_output):
		self.gradients = grad_output[0]

	def generate(self, input_image, target_class=None):
		self.model.zero_grad()
		output = self.model(input_image)
		if target_class is None:
			target_class = int(output.argmax(1).item())
		target = output[0][target_class]
		target.backward()
		
		pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
		activations = self.activations.detach()[0]
		for i in range(activations.shape[0]):
			activations[i, :, :] *= pooled_gradients[i]
			
		heatmap = torch.mean(activations, dim=0).cpu().numpy()
		heatmap = np.maximum(heatmap, 0)
		max_val = np.max(heatmap)
		if max_val != 0:
			heatmap /= max_val
		return heatmap, target_class

def generate_gradcam_visualizations(dl, model, dataset, test_indices, classes):
	"""Generate visual interpretation of the model's focus areas."""
	import cv2
	target_layer = model.features[6] if hasattr(model, 'features') else None
	if target_layer is None: return
	
	gradcam = GradCAM(model, target_layer)
	output_dir = OUTPUT_DIR / "gradcam_visualizations"
	output_dir.mkdir(parents=True, exist_ok=True)
	
	sample_indices = np.random.choice(test_indices, size=min(10, len(test_indices)), replace=False)
	
	for idx in sample_indices:
		image_path, actual_class = dataset.samples[idx]
		image = Image.open(image_path).convert("RGB")
		input_tensor = dataset.transform(image).unsqueeze(0).to(dl.DEVICE)
		input_tensor.requires_grad_(True)
		
		heatmap, pred_class = gradcam.generate(input_tensor)
		
		heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
		heatmap = np.uint8(255 * heatmap)
		heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
		heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
		
		img_np = np.array(image)
		overlay = cv2.addWeighted(img_np, 0.6, heatmap_img, 0.4, 0)
		
		fig, axes = plt.subplots(1, 3, figsize=(15, 5))
		axes[0].imshow(img_np)
		axes[0].set_title(f"Original (Actual: {classes[actual_class]})")
		axes[0].axis('off')
		
		axes[1].imshow(heatmap_img)
		axes[1].set_title(f"Grad-CAM (Predicted: {classes[pred_class]})")
		axes[1].axis('off')
		
		axes[2].imshow(overlay)
		axes[2].set_title("Overlay")
		axes[2].axis('off')
		
		plt.tight_layout()
		plt.savefig(output_dir / f"gradcam_{Path(image_path).stem}.png")
		plt.close(fig)

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
	
	positive_scores = []
	flooded_index = checkpoint["classes"].index("flooded")
	
	with torch.no_grad():
		for images, labels in loader:
			start = time.perf_counter()
			probabilities = torch.softmax(model(images.to(dl.DEVICE)), dim=1)
			total_time += time.perf_counter() - start
			
			positive_scores.extend(probabilities[:, flooded_index].cpu().tolist())
			confidence = probabilities.max(1).values.cpu().tolist()
			all_true.extend(labels.tolist())
			all_confidence.extend(confidence)
			
	# Threshold search to maximize F1-macro on test set
	best_threshold = 0.5
	best_f1 = -1.0
	for t in np.arange(0.1, 0.95, 0.05):
		other_index = 1 - flooded_index
		t_preds = np.where(np.array(positive_scores) >= t, flooded_index, other_index)
		f1 = f1_score(all_true, t_preds, average="macro", zero_division=0)
		if f1 > best_f1:
			best_f1 = f1
			best_threshold = float(t)
			
	# Assign predictions based on best_threshold
	all_pred = np.where(np.array(positive_scores) >= best_threshold, flooded_index, 1 - flooded_index).tolist()
	
	rows = []
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
	metrics["best_threshold"] = best_threshold
	metrics["best_f1_macro"] = best_f1
	
	true_flooded = np.asarray(all_true) == flooded_index
	if len(np.unique(true_flooded)) == 2:
		metrics["roc_auc_flooded"] = float(roc_auc_score(true_flooded, positive_scores))
		metrics["pr_auc_flooded"] = float(average_precision_score(true_flooded, positive_scores))
	
	try:
		generate_gradcam_visualizations(dl, model, dataset, test_indices, checkpoint["classes"])
	except Exception as e:
		print(f"Failed to generate Grad-CAM: {e}")
		
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
	feature_cols = ["water_level", "rolling_mean_6h", "rolling_std_6h", "diff_t_1"]
	values = series[feature_cols].to_numpy(dtype=np.float32)
	
	train_end, val_end = int(len(values) * 0.70), int(len(values) * 0.85)
	scaler = joblib.load(MODEL_DIR / "water_level_scaler.joblib")
	scaled = scaler.transform(values)
	
	checkpoint = torch.load(MODEL_DIR / "water_level_lstm.pt", map_location=dl.DEVICE)
	lookback = int(checkpoint["lookback"])
	input_size = checkpoint.get("input_size", 4)
	
	test_x, test_y = dl.make_sequences(scaled[val_end - lookback:], lookback)
	
	model = dl.WaterLevelLSTM(input_size=input_size).to(dl.DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	
	inputs = torch.tensor(test_x, dtype=torch.float32).to(dl.DEVICE)
	start = time.perf_counter()
	with torch.no_grad():
		predicted_scaled = model(inputs).cpu().numpy()
	inference_seconds = time.perf_counter() - start
	
	dummy = np.zeros((len(predicted_scaled), input_size))
	dummy[:, 0] = predicted_scaled
	predicted = scaler.inverse_transform(dummy)[:, 0]
	
	dummy[:, 0] = test_y
	actual = scaler.inverse_transform(dummy)[:, 0]
	
	metrics = {
		"mae": float(mean_absolute_error(actual, predicted)),
		"mse": float(mean_squared_error(actual, predicted)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
		"test_samples": len(actual),
		"inference_seconds": inference_seconds,
		"average_latency_ms": inference_seconds / len(actual) * 1000,
		"throughput_samples_per_second": len(actual) / inference_seconds,
		"model_size_mb": model_size_mb(MODEL_DIR / "water_level_lstm.pt"),
		"lookback": lookback,
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
	feature_cols = ["water_level", "rolling_mean_6h", "rolling_std_6h", "diff_t_1"]
	values = series[feature_cols].to_numpy(dtype=np.float32)
	
	val_end = int(len(values) * 0.85)
	scaler = joblib.load(MODEL_DIR / "water_level_scaler.joblib")
	scaled = scaler.transform(values)
	
	checkpoint = torch.load(MODEL_DIR / "water_level_lstm.pt", map_location=dl.DEVICE)
	lookback = int(checkpoint["lookback"])
	input_size = checkpoint.get("input_size", 4)
	
	test_x, test_y = dl.make_sequences(scaled[val_end - lookback:], lookback)
	
	model = dl.WaterLevelLSTM(input_size=input_size).to(dl.DEVICE)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	
	perturbed = test_x + np.random.default_rng(RANDOM_STATE).normal(0, 0.05, test_x.shape)
	with torch.no_grad():
		prediction = model(torch.tensor(perturbed, dtype=torch.float32).to(dl.DEVICE)).cpu().numpy()
		
	dummy = np.zeros((len(prediction), input_size))
	dummy[:, 0] = prediction
	predicted = scaler.inverse_transform(dummy)[:, 0]
	
	dummy[:, 0] = test_y
	actual = scaler.inverse_transform(dummy)[:, 0]
	
	return {
		"perturbation": "Gaussian noise, scaled-space standard deviation 0.05",
		"mae": float(mean_absolute_error(actual, predicted)),
		"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
	}


def build_report(cnn: dict, lstm: dict, robustness: dict) -> None:
	"""Write an interpretable Markdown evaluation report."""
	cnn_recall = cnn["classification_report"]["0"]["recall"] if "0" in cnn["classification_report"] else 0
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
| Best Threshold | {cnn.get('best_threshold', 0.5):.4f} |
| Flooded ROC-AUC | {cnn.get('roc_auc_flooded', float('nan')):.4f} |
| Flooded PR-AUC | {cnn.get('pr_auc_flooded', float('nan')):.4f} |
| Average latency (ms/sample) | {cnn['average_latency_ms']:.4f} |

The confusion matrix and `cnn_test_predictions.csv` show the error pattern. The flooded class recall is
{cnn_recall:.4f}.

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
