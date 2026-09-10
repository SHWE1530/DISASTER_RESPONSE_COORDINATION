"""Train and evaluate the Stage02 deep-learning models.

This module uses only data under ``Stage02_DL/data/raw``:

* ``data/images`` trains a binary flooded/unflooded CNN (ResNet18 transfer learning).
* ``data/Engineered_History_Trend_Dataset.csv`` trains a chronological water-level LSTM.

A third model (a logistic-regression zone-risk classifier over Master_Dataset.csv)
was removed: it duplicated the Stage 01 ML task on the same labels, was never
called by the dashboard or by any other stage, and existed only as dead weight.
Zone-risk classification lives in Stage 01.

Run from the repository root with::

    python Stage02_DL/03_dl_engineer.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets as vision_datasets
from torchvision import models as vision_models
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
ENGINEERED_HISTORY_PATH = BASE_DIR / "data" / "Engineered_History_Trend_Dataset.csv"
RANDOM_STATE = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
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
	"""CNN classifier supporting baseline custom architecture, BatchNorm custom architecture, and ResNet18 transfer learning."""

	def __init__(self, classes: int = 2, use_resnet: bool = True, use_batchnorm: bool = True, pretrained: bool = True):
		super().__init__()
		self.classes = classes
		self.use_resnet = use_resnet
		self.use_batchnorm = use_batchnorm
		self.pretrained = pretrained
		self._build_architecture()

	def _build_architecture(self):
		if self.use_resnet:
			if hasattr(self, "features"):
				delattr(self, "features")
			if hasattr(self, "classifier"):
				delattr(self, "classifier")
			if self.pretrained:
				try:
					backbone = vision_models.resnet18(weights=vision_models.ResNet18_Weights.DEFAULT)
				except Exception:
					backbone = vision_models.resnet18(weights=None)
			else:
				backbone = vision_models.resnet18(weights=None)
			num_ftrs = backbone.fc.in_features
			backbone.fc = nn.Sequential(
				nn.Dropout(0.3),
				nn.Linear(num_ftrs, self.classes)
			)
			self.backbone = backbone
		else:
			if hasattr(self, "backbone"):
				delattr(self, "backbone")
			if self.use_batchnorm:
				self.features = nn.Sequential(
					nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
					nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
					nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
				)
			else:
				self.features = nn.Sequential(
					nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
					nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
					nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
				)
			self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, self.classes))

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		if self.use_resnet:
			return self.backbone(inputs)
		return self.classifier(self.features(inputs))

	def load_state_dict(self, state_dict: dict, strict: bool = True):
		is_resnet_state = any(k.startswith("backbone") for k in state_dict.keys())
		has_bn = any("running_mean" in k or "running_var" in k for k in state_dict.keys())
		
		if is_resnet_state != self.use_resnet or (not is_resnet_state and has_bn != self.use_batchnorm):
			self.use_resnet = is_resnet_state
			self.use_batchnorm = has_bn
			self._build_architecture()

		return super().load_state_dict(state_dict, strict=strict)


def run_cnn_epoch(
	model,
	loader,
	optimizer=None,
	class_weights: torch.Tensor | None = None,
) -> tuple[float, list[int], list[float]]:
	"""Run one training or evaluation epoch, returning loss, true targets, and raw positive (flooded) probabilities."""
	training = optimizer is not None
	model.train(training)
	loss_function = nn.CrossEntropyLoss(
		weight=class_weights.to(DEVICE) if class_weights is not None else None
	)
	total_loss = 0.0
	y_true, y_prob_flooded = [], []
	
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
		probs = torch.softmax(outputs, dim=1)
		
		y_true.extend(labels.cpu().tolist())
		y_prob_flooded.extend(probs[:, 0].cpu().tolist()) # class 0 is 'flooded'
		
	return total_loss / len(loader.dataset), y_true, y_prob_flooded


def train_cnn() -> dict:
	"""Train, benchmark controlled experiments, evaluate, and save the flooded/unflooded image classifier."""
	image_dir = BASE_DIR / "data" / "images"
	if not image_dir.exists():
		raise FileNotFoundError(f"CNN image directory not found: {image_dir}")
		
	normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	base_dataset = vision_datasets.ImageFolder(image_dir)
	labels = base_dataset.classes # ['flooded', 'unflooded']
	flooded_idx = labels.index("flooded")
	unflooded_idx = labels.index("unflooded")
	
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
	val_targets = np.asarray(base_dataset.targets)[val_indices]
	test_targets = np.asarray(base_dataset.targets)[test_indices]

	# Persist the split by FILE PATH, not just by index.
	# ImageFolder orders samples by the filesystem, so re-deriving the split from
	# a seed on another machine can silently produce a different "held-out" set.
	# The evaluation engineer replays this manifest instead of re-splitting.
	split_manifest = {
		"seed": RANDOM_STATE,
		"strategy": "stratified 70/15/15 via train_test_split on ImageFolder order",
		"classes": labels,
		"counts": {
			"train": int(len(train_indices)),
			"val": int(len(val_indices)),
			"test": int(len(test_indices)),
		},
		"splits": {
			name: [
				str(Path(base_dataset.samples[index][0]).relative_to(BASE_DIR)).replace("\\", "/")
				for index in split_indices
			]
			for name, split_indices in (
				("train", train_indices), ("val", val_indices), ("test", test_indices)
			)
		},
	}
	(OUTPUT_DIR / "cnn_split_manifest.json").write_text(
		json.dumps(split_manifest, indent=2), encoding="utf-8"
	)
	print(f"  Split manifest saved: {OUTPUT_DIR / 'cnn_split_manifest.json'}")
	
	class_counts = np.bincount(train_targets, minlength=len(labels)).astype(np.float32)
	print(f"\n[CNN DATA DISTRIBUTION]")
	print(f"  Total Images: {len(base_dataset)}")
	print(f"  Train Count : {len(train_indices)} (Flooded: {class_counts[flooded_idx]}, Unflooded: {class_counts[unflooded_idx]})")
	print(f"  Val Count   : {len(val_indices)} (Flooded: {np.sum(val_targets==flooded_idx)}, Unflooded: {np.sum(val_targets==unflooded_idx)})")
	print(f"  Test Count  : {len(test_indices)} (Flooded: {np.sum(test_targets==flooded_idx)}, Unflooded: {np.sum(test_targets==unflooded_idx)})")
	
	# Class weighting calculation for CrossEntropyLoss
	# pos_weight calculation: inverse frequency
	weights_inv = 1.0 / np.maximum(class_counts, 1.0)
	class_weights_tensor = torch.tensor(weights_inv / weights_inv.sum(), dtype=torch.float32)

	# Realistic CCTV/drone augmentation for training
	train_transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(10),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
		transforms.ToTensor(),
		normalize,
	])

	eval_transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		normalize,
	])

	train_dataset = vision_datasets.ImageFolder(image_dir, transform=train_transform)
	eval_dataset = vision_datasets.ImageFolder(image_dir, transform=eval_transform)

	loaders = {
		"train": DataLoader(Subset(train_dataset, train_indices), BATCH_SIZE, shuffle=True),
		"val": DataLoader(Subset(eval_dataset, val_indices), BATCH_SIZE, shuffle=False),
		"test": DataLoader(Subset(eval_dataset, test_indices), BATCH_SIZE, shuffle=False),
	}

	# Define 3 Controlled Experiments:
	# Exp A: Baseline Custom CNN (No BatchNorm, Double Balancing, argmax threshold)
	# Exp B: Corrected Custom CNN with BatchNorm + Weighted Loss + Val Thresholding
	# Exp C: ResNet18 Transfer Learning + Weighted Loss + Val Thresholding (Strongest Config)
	
	experiments = {
		"Baseline_3Layer_CNN": {"use_resnet": False, "use_bn": False, "use_double_sampler": True, "lr": 1e-3},
		"BatchNorm_Custom_CNN": {"use_resnet": False, "use_bn": True, "use_double_sampler": False, "lr": 1e-3},
		"ResNet18_Transfer": {"use_resnet": True, "use_bn": True, "use_double_sampler": False, "lr": 3e-4},
	}
	
	exp_results = {}
	best_overall_model_state = None
	best_overall_threshold = 0.5
	best_overall_score = -1.0
	best_exp_name = "ResNet18_Transfer"
	best_history = None

	print("\n" + "=" * 70)
	print("[CNN CONTROLLED EXPERIMENTS BENCHMARKING]")
	print("=" * 70)

	for exp_name, cfg in experiments.items():
		seed_everything()
		
		# Build loader for experiment
		if cfg["use_double_sampler"]:
			sample_weights = class_weights_tensor.numpy()[train_targets]
			train_sampler = WeightedRandomSampler(
				torch.as_tensor(sample_weights, dtype=torch.double),
				num_samples=len(sample_weights),
				replacement=True
			)
			exp_train_loader = DataLoader(Subset(train_dataset, train_indices), BATCH_SIZE, sampler=train_sampler)
		else:
			exp_train_loader = DataLoader(Subset(train_dataset, train_indices), BATCH_SIZE, shuffle=True)

		model = DisasterCNN(len(labels), use_resnet=cfg["use_resnet"], use_batchnorm=cfg["use_bn"]).to(DEVICE)
		optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

		history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
		best_state, best_val_f1 = None, -1.0
		patience, patience_counter = 12, 0

		for epoch in range(EPOCHS):
			# Use class_weights loss if not using double sampler or for weighted loss
			loss_w = class_weights_tensor if not cfg["use_double_sampler"] else None
			train_loss, train_true, train_prob = run_cnn_epoch(model, exp_train_loader, optimizer, loss_w)
			val_loss, val_true, val_prob = run_cnn_epoch(model, loaders["val"], None, None)

			train_preds = [flooded_idx if p >= 0.5 else unflooded_idx for p in train_prob]
			val_preds = [flooded_idx if p >= 0.5 else unflooded_idx for p in val_prob]

			val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

			history["train_loss"].append(train_loss)
			history["val_loss"].append(val_loss)
			history["train_accuracy"].append(float(accuracy_score(train_true, train_preds)))
			history["val_accuracy"].append(float(accuracy_score(val_true, val_preds)))

			if val_f1 > best_val_f1:
				best_val_f1 = val_f1
				best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= patience:
					break

		if best_state is not None:
			model.load_state_dict(best_state)

		# Validation Decision Threshold Tuning for P(flooded)
		val_loss, val_true, val_prob = run_cnn_epoch(model, loaders["val"], None, None)
		best_thresh = 0.5
		best_val_flooded_f1 = -1.0

		# Grid search threshold on validation set
		for t in np.arange(0.15, 0.85, 0.05):
			t_preds = [flooded_idx if p >= t else unflooded_idx for p in val_prob]
			# Flooded class is index 0
			f_rec = recall_score(val_true, t_preds, pos_label=flooded_idx, zero_division=0)
			f_f1 = f1_score(val_true, t_preds, pos_label=flooded_idx, zero_division=0)
			if f_f1 > best_val_flooded_f1:
				best_val_flooded_f1 = f_f1
				best_thresh = float(t)

		# VALIDATION metrics at the selected threshold. Architecture selection is
		# driven exclusively by these: an earlier revision computed the selection
		# score from held-out TEST metrics, which leaks the test set into the
		# model-choice decision and invalidates it as an independent estimate.
		val_eval_thresh = 0.5 if cfg["use_double_sampler"] else best_thresh
		val_preds_at_thresh = [flooded_idx if p >= val_eval_thresh else unflooded_idx for p in val_prob]
		val_flooded_rec = recall_score(val_true, val_preds_at_thresh, pos_label=flooded_idx, zero_division=0)
		val_flooded_f1 = f1_score(val_true, val_preds_at_thresh, pos_label=flooded_idx, zero_division=0)
		binary_val_true = (np.array(val_true) == flooded_idx).astype(int)
		val_pr_auc = (
			float(average_precision_score(binary_val_true, np.array(val_prob)))
			if len(np.unique(binary_val_true)) == 2
			else 0.0
		)

		# Evaluate on HELD-OUT TEST SET
		test_loss, test_true, test_prob = run_cnn_epoch(model, loaders["test"], None, None)
		
		if cfg["use_double_sampler"]:
			test_preds = [flooded_idx if p >= 0.5 else unflooded_idx for p in test_prob]
			eval_thresh = 0.5
		else:
			test_preds = [flooded_idx if p >= best_thresh else unflooded_idx for p in test_prob]
			eval_thresh = best_thresh

		# Compute held-out test metrics
		acc = accuracy_score(test_true, test_preds)
		macro_prec = precision_score(test_true, test_preds, average="macro", zero_division=0)
		macro_rec = recall_score(test_true, test_preds, average="macro", zero_division=0)
		macro_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
		
		flooded_prec = precision_score(test_true, test_preds, pos_label=flooded_idx, zero_division=0)
		flooded_rec = recall_score(test_true, test_preds, pos_label=flooded_idx, zero_division=0)
		flooded_f1 = f1_score(test_true, test_preds, pos_label=flooded_idx, zero_division=0)
		
		# ROC-AUC & PR-AUC (for flooded class = 0, target=0 is positive)
		binary_test_true = (np.array(test_true) == flooded_idx).astype(int)
		binary_test_prob = np.array(test_prob) # P(flooded)
		roc_auc = float(roc_auc_score(binary_test_true, binary_test_prob))
		pr_auc = float(average_precision_score(binary_test_true, binary_test_prob))
		cm = confusion_matrix(test_true, test_preds, labels=[0, 1])

		exp_results[exp_name] = {
			"threshold": float(eval_thresh),
			"validation_flooded_recall": float(val_flooded_rec),
			"validation_flooded_f1": float(val_flooded_f1),
			"validation_pr_auc": float(val_pr_auc),
			"accuracy": float(acc),
			"macro_precision": float(macro_prec),
			"macro_recall": float(macro_rec),
			"macro_f1": float(macro_f1),
			"flooded_precision": float(flooded_prec),
			"flooded_recall": float(flooded_rec),
			"flooded_f1": float(flooded_f1),
			"roc_auc": float(roc_auc),
			"pr_auc": float(pr_auc),
			"confusion_matrix": cm.tolist(),
		}

		print(f"\n--- Experiment: {exp_name} ---")
		print(f"  Decision Threshold: {eval_thresh:.2f}")
		print(f"  [VAL ] FLOODED Recall: {val_flooded_rec:.4f}  |  FLOODED F1: {val_flooded_f1:.4f}  |  PR-AUC: {val_pr_auc:.4f}   <- drives selection")
		print(f"  [TEST] Accuracy      : {acc:.4f}  |  Macro F1: {macro_f1:.4f}")
		print(f"  [TEST] FLOODED Recall: {flooded_rec:.4f}  |  FLOODED Precision: {flooded_prec:.4f}  |  FLOODED F1: {flooded_f1:.4f}")
		print(f"  [TEST] ROC-AUC       : {roc_auc:.4f}  |  PR-AUC   : {pr_auc:.4f}")
		print(f"  [TEST] Confusion     : TP(Flooded)={cm[0,0]}, FN={cm[0,1]}, FP={cm[1,0]}, TN(Unflooded)={cm[1,1]}")

		# Selection score computed on VALIDATION only.
		# Primary Flooded Recall, Secondary Flooded F1 & PR-AUC.
		selection_score = (val_flooded_rec * 0.5) + (val_flooded_f1 * 0.3) + (val_pr_auc * 0.2)
		if selection_score > best_overall_score:
			best_overall_score = selection_score
			best_exp_name = exp_name
			best_overall_model_state = best_state
			best_overall_threshold = eval_thresh
			best_history = history

	print("\n" + "=" * 70)
	print(f"SELECTED OPTIMAL CNN CONFIGURATION: {best_exp_name}")
	print("=" * 70)

	best_model_config = experiments[best_exp_name]
	final_cnn_model = DisasterCNN(
		len(labels),
		use_resnet=best_model_config["use_resnet"],
		use_batchnorm=best_model_config["use_bn"]
	).to(DEVICE)
	
	if best_overall_model_state is not None:
		final_cnn_model.load_state_dict(best_overall_model_state)

	# Evaluate Final Model on Test Set
	test_loss, test_true, test_prob = run_cnn_epoch(final_cnn_model, loaders["test"], None, None)
	final_test_preds = [flooded_idx if p >= best_overall_threshold else unflooded_idx for p in test_prob]

	final_metrics = classification_metrics(test_true, final_test_preds, list(range(len(labels))))
	
	binary_test_true = (np.array(test_true) == flooded_idx).astype(int)
	binary_test_prob = np.array(test_prob)

	final_metrics.update({
		"test_loss": test_loss,
		"classes": labels,
		"device": str(DEVICE),
		"selected_experiment": best_exp_name,
		"best_threshold": float(best_overall_threshold),
		"use_resnet": best_model_config["use_resnet"],
		"use_batchnorm": best_model_config["use_bn"],
		"flooded_precision": float(precision_score(test_true, final_test_preds, pos_label=flooded_idx, zero_division=0)),
		"flooded_recall": float(recall_score(test_true, final_test_preds, pos_label=flooded_idx, zero_division=0)),
		"flooded_f1": float(f1_score(test_true, final_test_preds, pos_label=flooded_idx, zero_division=0)),
		"roc_auc_flooded": float(roc_auc_score(binary_test_true, binary_test_prob)),
		"pr_auc_flooded": float(average_precision_score(binary_test_true, binary_test_prob)),
		"class_weights": class_weights_tensor.tolist(),
		"train_class_distribution": {
			labels[index]: int(count) for index, count in enumerate(class_counts)
		},
		"experiments_benchmark": exp_results
	})

	# Save PyTorch Model Checkpoint
	torch.save({
		"state_dict": final_cnn_model.state_dict(),
		"classes": labels,
		"image_size": IMAGE_SIZE,
		"best_threshold": float(best_overall_threshold),
		"use_resnet": best_model_config["use_resnet"],
		"use_batchnorm": best_model_config["use_bn"],
		"class_weights": class_weights_tensor.tolist(),
	}, MODEL_DIR / "disaster_cnn.pt")
	
	save_confusion_matrix(test_true, final_test_preds, list(range(len(labels))), OUTPUT_DIR / "cnn_confusion_matrix.png", f"CNN Test Confusion Matrix ({best_exp_name})")

	if best_history is not None:
		figure, axes = plt.subplots(1, 2, figsize=(11, 4))
		axes[0].plot(best_history["train_loss"], label="train")
		axes[0].plot(best_history["val_loss"], label="validation")
		axes[0].set_title(f"CNN Loss ({best_exp_name})")
		axes[1].plot(best_history["train_accuracy"], label="train")
		axes[1].plot(best_history["val_accuracy"], label="validation")
		axes[1].set_title(f"CNN Accuracy ({best_exp_name})")
		for axis in axes:
			axis.legend()
		figure.tight_layout()
		figure.savefig(OUTPUT_DIR / "cnn_training_curves.png", dpi=150)
		plt.close(figure)

	return final_metrics


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
	def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
		self.output = nn.Linear(hidden_size, 1)

	def forward(self, inputs):
		# This used to silently rebuild self.lstm on a feature-count mismatch,
		# which discards every trained weight and returns the output of a freshly
		# initialised random network. A forecast produced that way is
		# indistinguishable from a real one at the call site. Fail loudly instead.
		if inputs.size(-1) != self.input_size:
			raise ValueError(
				f"WaterLevelLSTM expects {self.input_size} features per timestep, "
				f"received {inputs.size(-1)}. Rebuild the model with the correct "
				"input_size instead of feeding a mismatched tensor."
			)
		outputs, _ = self.lstm(inputs)
		return self.output(outputs[:, -1, :]).squeeze(-1)


def make_sequences(features: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
	"""Create chronological one-step forecasting windows. Target is water_level (col 0)."""
	features = np.asarray(features)
	if features.ndim == 1:
		features = features.reshape(-1, 1)
	sequences = np.asarray([features[index:index + lookback] for index in range(len(features) - lookback)])
	targets = features[lookback:, 0]
	return sequences, targets


def evaluate_recursive_forecast(
	series: pd.DataFrame,
	scaler: StandardScaler,
	model: nn.Module,
	horizons: tuple[int, ...] = (1, 3, 6),
	max_origins: int = 250,
) -> dict:
	"""Measure error at each step of the recursive forecast the API actually serves.

	Rolls the model forward step by step from origins inside the chronological
	test partition, exactly as forecast_water_levels() does at inference, and
	compares each horizon against the observed series.
	"""
	feature_cols = ["water_level", "rolling_mean_6h", "rolling_std_6h", "diff_t_1"]
	values = series[feature_cols].to_numpy(dtype=np.float32)
	levels = series["water_level"].to_numpy(dtype=np.float64)
	val_end = int(len(values) * 0.85)
	max_horizon = max(horizons)

	origins = [
		origin for origin in range(val_end, len(levels) - max_horizon)
		if origin >= LOOKBACK
	]
	if not origins:
		return {}
	# Evenly sample origins so this stays fast without biasing toward one period.
	if len(origins) > max_origins:
		step = len(origins) / max_origins
		origins = [origins[int(index * step)] for index in range(max_origins)]

	errors: dict[int, list[float]] = {horizon: [] for horizon in horizons}
	model.eval()
	with torch.no_grad():
		for origin in origins:
			window = list(levels[origin - LOOKBACK:origin])
			for step in range(1, max_horizon + 1):
				feats = make_features_for_inference(window[-LOOKBACK:])
				scaled = scaler.transform(feats)[None, :, :]
				scaled_prediction = model(
					torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
				).cpu().item()
				dummy = np.zeros((1, len(feature_cols)))
				dummy[0, 0] = scaled_prediction
				prediction = float(scaler.inverse_transform(dummy)[0, 0])
				window.append(prediction)
				if step in errors:
					errors[step].append(prediction - levels[origin + step - 1])

	results = {}
	for horizon in horizons:
		residuals = np.asarray(errors[horizon], dtype=np.float64)
		if residuals.size == 0:
			continue
		results[f"step_{horizon}"] = {
			"mae": float(np.mean(np.abs(residuals))),
			"rmse": float(np.sqrt(np.mean(residuals ** 2))),
			"samples": int(residuals.size),
		}
	return results


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
	
	# Multi-step recursive forecast evaluation.
	#
	# The API and the dashboard chart both serve a 6-step recursive forecast, but
	# only 1-step MAE was ever measured. Recursive forecasting feeds each
	# prediction back in as input, so error compounds -- the honest thing is to
	# report the horizon actually served.
	multi_step_metrics = evaluate_recursive_forecast(series, scaler, model, horizons=(1, 3, 6))
	metrics["recursive_forecast"] = multi_step_metrics
	print("\n  Recursive multi-step forecast error (test partition):")
	for horizon_key, horizon_metrics in multi_step_metrics.items():
		print(
			f"    {horizon_key:>9}: MAE={horizon_metrics['mae']:.4f}  "
			f"RMSE={horizon_metrics['rmse']:.4f}  (n={horizon_metrics['samples']})"
		)

	metrics["training_distribution"] = {
		"water_level_mean_m": float(scaler.mean_[0]),
		"water_level_std_m": float(scaler.scale_[0]),
		"water_level_min_m": float(series["water_level"].min()),
		"water_level_max_m": float(series["water_level"].max()),
		"note": (
			"Inputs beyond +/-4 sigma of the training mean are extrapolation. "
			"forecast_water_levels() flags them via water_level_distribution_check()."
		),
	}

	split_manifest = {
		"seed": RANDOM_STATE,
		"strategy": "chronological 70/15/15, no shuffling",
		"lookback": LOOKBACK,
		"series_rows": int(len(values)),
		"train_rows": int(train_end),
		"val_rows": int(val_end - train_end),
		"test_rows": int(len(values) - val_end),
		"train_end_index": int(train_end),
		"val_end_index": int(val_end),
		"first_timestamp": str(series["timestamp"].iloc[0]),
		"train_end_timestamp": str(series["timestamp"].iloc[train_end - 1]),
		"val_end_timestamp": str(series["timestamp"].iloc[val_end - 1]),
		"last_timestamp": str(series["timestamp"].iloc[-1]),
	}
	(OUTPUT_DIR / "lstm_split_manifest.json").write_text(
		json.dumps(split_manifest, indent=2), encoding="utf-8"
	)

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


_CNN_CACHE: dict[str, Any] = {}
_LSTM_CACHE: dict[str, Any] = {}


def _load_cnn_bundle() -> dict:
	"""Load the CNN checkpoint once and reuse it across requests.

	predict_image() previously re-read the 44 MB checkpoint and rebuilt ResNet18
	on every single call (~230-450 ms per request).
	"""
	if not _CNN_CACHE:
		# weights_only=True: torch.load unpickles arbitrary objects by default,
		# which makes loading a checkpoint equivalent to executing it.
		checkpoint = torch.load(
			MODEL_DIR / "disaster_cnn.pt", map_location=DEVICE, weights_only=True
		)
		model = DisasterCNN(
			len(checkpoint["classes"]),
			use_resnet=checkpoint.get("use_resnet", True),
			use_batchnorm=checkpoint.get("use_batchnorm", True),
			pretrained=False
		).to(DEVICE)
		model.load_state_dict(checkpoint["state_dict"])
		model.eval()
		normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		_CNN_CACHE["checkpoint"] = checkpoint
		_CNN_CACHE["model"] = model
		_CNN_CACHE["transform"] = transforms.Compose([
			transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
			transforms.ToTensor(),
			normalize,
		])
	return _CNN_CACHE


def _load_lstm_bundle() -> dict:
	"""Load the LSTM checkpoint and scaler once and reuse them across requests."""
	if not _LSTM_CACHE:
		checkpoint = torch.load(
			MODEL_DIR / "water_level_lstm.pt", map_location=DEVICE, weights_only=True
		)
		input_size = int(checkpoint.get("input_size", 4))
		model = WaterLevelLSTM(input_size=input_size).to(DEVICE)
		model.load_state_dict(checkpoint["state_dict"])
		model.eval()
		_LSTM_CACHE["checkpoint"] = checkpoint
		_LSTM_CACHE["model"] = model
		_LSTM_CACHE["scaler"] = joblib.load(MODEL_DIR / "water_level_scaler.joblib")
		_LSTM_CACHE["lookback"] = int(checkpoint["lookback"])
		_LSTM_CACHE["input_size"] = input_size
	return _LSTM_CACHE


def predict_image(image_path: str | Path) -> dict:
	"""Load the saved CNN and predict one new flood image."""
	bundle = _load_cnn_bundle()
	checkpoint = bundle["checkpoint"]
	model = bundle["model"]
	transform = bundle["transform"]
	best_threshold = checkpoint.get("best_threshold", 0.5)

	with Image.open(image_path).convert("RGB") as image:
		input_tensor = transform(image).unsqueeze(0).to(DEVICE)
	with torch.no_grad():
		probabilities = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
		
	flooded_index = checkpoint["classes"].index("flooded")
	unflooded_index = checkpoint["classes"].index("unflooded")
	flooded_prob = probabilities[flooded_index]
	
	if flooded_prob >= best_threshold:
		predicted_label = "flooded"
		conf = float(flooded_prob)
	else:
		predicted_label = "unflooded"
		conf = float(probabilities[unflooded_index])
		
	return {"label": predicted_label, "confidence": conf, "flooded_probability": float(flooded_prob)}


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


# Any input window whose water level sits further than this many standard
# deviations from the training mean is outside the range the LSTM ever saw.
# Beyond it the network's output is extrapolation, not forecasting.
OOD_SIGMA_THRESHOLD = 4.0


def water_level_distribution_check(values: Iterable[float]) -> dict:
	"""Flag input windows that fall outside the LSTM's training distribution.

	The scaler was fitted on the training partition, so its mean_/scale_ for the
	water_level column IS the training distribution. Reusing it here avoids
	storing a second copy of the same statistics.

	This exists because the model silently produced physically impossible output
	for out-of-range inputs: fed a flat 10.0 m series (training mean 3.12 m,
	std 1.56 m, i.e. +4.4 sigma) it returned an oscillating
	8.50 / 9.28 / 6.63 / 8.45 / 6.04 / 7.27 instead of a flat ~10.0 m.
	"""
	bundle = _load_lstm_bundle()
	scaler = bundle["scaler"]
	train_mean = float(scaler.mean_[0])
	train_std = float(scaler.scale_[0])

	array = np.asarray(list(values), dtype="float64")
	z_scores = np.abs((array - train_mean) / max(train_std, 1e-9))
	max_z = float(z_scores.max()) if array.size else 0.0

	return {
		"in_distribution": bool(max_z <= OOD_SIGMA_THRESHOLD),
		"max_sigma_from_training_mean": round(max_z, 2),
		"training_mean_m": round(train_mean, 3),
		"training_std_m": round(train_std, 3),
		"sigma_threshold": OOD_SIGMA_THRESHOLD,
	}


def forecast_water_levels(recent_levels: Iterable[float], horizon: int = 6) -> list[float]:
	"""Recursively forecast multiple future water-level steps."""
	if horizon < 1 or horizon > 168:
		raise ValueError("horizon must be between 1 and 168 steps")

	bundle = _load_lstm_bundle()
	lookback = bundle["lookback"]
	input_size = bundle["input_size"]
	scaler = bundle["scaler"]
	model = bundle["model"]

	values = [float(value) for value in recent_levels]
	if len(values) < lookback:
		raise ValueError(f"At least {lookback} water-level values are required")

	# NaN/inf used to propagate silently and return a list of NaN forecasts.
	if not np.isfinite(np.asarray(values, dtype="float64")).all():
		raise ValueError("water_levels must contain only finite numeric values")

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


def main() -> None:
	"""Run all trainable Stage02 components and save one result manifest."""
	seed_everything()
	ensure_directories()
	results = {
		"cnn": train_cnn(),
		"lstm": train_lstm(),
	}
	(OUTPUT_DIR / "dl_training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
	print("\n" + "=" * 70)
	print("STAGE 02 DEEP LEARNING TRAINING COMPLETED SUCCESSFULLY")
	print("=" * 70)
	print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
