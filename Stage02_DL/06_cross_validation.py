"""Stratified k-fold cross-validation for the Stage 02 CNN.

WHY THIS EXISTS
---------------
The single 70/15/15 split leaves a test set of 75 images containing only 15
flooded positives. A flooded recall of 0.9333 on 15 positives means "14 of 15":
one image moves the metric by 6.7 points, and the 95% confidence interval spans
roughly 68%-99.8%. Quoting that number as if it were precise is the single
weakest claim in the whole project.

Cross-validation is the standard answer for small datasets. Every one of the 500
images is used for evaluation exactly once across the folds, so the estimate is
built on all 100 flooded positives rather than 15. The fold-to-fold standard
deviation is itself the honest measure of how much the single-split number
should be trusted.

This does NOT replace the held-out evaluation in 04_evaluation_engineer.py --
that one still measures the exact deployed checkpoint. This measures the
ARCHITECTURE AND TRAINING PROCEDURE, which is the thing a confidence interval
can legitimately be placed around.

Usage:
    python Stage02_DL/06_cross_validation.py               # 5 folds
    python Stage02_DL/06_cross_validation.py --folds 3     # faster
    python Stage02_DL/06_cross_validation.py --epochs 15   # shorter runs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as vision_datasets
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
RANDOM_STATE = 42


def _load_dl_module():
    spec = importlib.util.spec_from_file_location(
        "stage02_dl_engineer", BASE_DIR / "03_dl_engineer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dl = _load_dl_module()


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for any metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(scores, 100 * alpha / 2)),
        float(np.percentile(scores, 100 * (1 - alpha / 2))),
    )


def run_fold(
    image_dir: Path,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
    flooded_idx: int,
    epochs: int,
    batch_size: int,
) -> dict:
    """Train the selected architecture on one fold and score the held-out part."""
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((dl.IMAGE_SIZE, dl.IMAGE_SIZE)),
        transforms.RandomResizedCrop(dl.IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((dl.IMAGE_SIZE, dl.IMAGE_SIZE)),
        transforms.ToTensor(), normalize,
    ])
    train_dataset = vision_datasets.ImageFolder(image_dir, transform=train_transform)
    eval_dataset = vision_datasets.ImageFolder(image_dir, transform=eval_transform)

    # Carve a validation slice out of THIS fold's training data only, for early
    # stopping and threshold tuning. The fold's test partition is never touched
    # until the final scoring below.
    inner_train_idx, inner_val_idx = train_test_split(
        train_idx, test_size=0.1765, random_state=RANDOM_STATE,
        stratify=targets[train_idx],
    )

    loaders = {
        "train": DataLoader(Subset(train_dataset, inner_train_idx), batch_size, shuffle=True),
        "val": DataLoader(Subset(eval_dataset, inner_val_idx), batch_size, shuffle=False),
        "test": DataLoader(Subset(eval_dataset, test_idx), batch_size, shuffle=False),
    }

    class_counts = np.bincount(targets[inner_train_idx], minlength=len(labels)).astype(np.float32)
    weights_inv = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = torch.tensor(weights_inv / weights_inv.sum(), dtype=torch.float32)

    dl.seed_everything()
    model = dl.DisasterCNN(len(labels), use_resnet=True, use_batchnorm=True).to(dl.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_state, best_val_f1, patience_counter = None, -1.0, 0
    for _epoch in range(epochs):
        dl.run_cnn_epoch(model, loaders["train"], optimizer, class_weights)
        _loss, val_true, val_prob = dl.run_cnn_epoch(model, loaders["val"], None, None)
        val_pred = [flooded_idx if p >= 0.5 else 1 - flooded_idx for p in val_prob]
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold tuned on the inner validation slice, never on the fold's test set.
    _loss, val_true, val_prob = dl.run_cnn_epoch(model, loaders["val"], None, None)
    best_threshold, best_flooded_f1 = 0.5, -1.0
    for t in np.arange(0.15, 0.85, 0.05):
        preds = [flooded_idx if p >= t else 1 - flooded_idx for p in val_prob]
        score = f1_score(val_true, preds, pos_label=flooded_idx, zero_division=0)
        if score > best_flooded_f1:
            best_flooded_f1, best_threshold = score, float(t)

    _loss, test_true, test_prob = dl.run_cnn_epoch(model, loaders["test"], None, None)
    test_pred = [flooded_idx if p >= best_threshold else 1 - flooded_idx for p in test_prob]

    binary_true = (np.asarray(test_true) == flooded_idx).astype(int)
    return {
        "threshold": best_threshold,
        "accuracy": float(accuracy_score(test_true, test_pred)),
        "macro_f1": float(f1_score(test_true, test_pred, average="macro", zero_division=0)),
        "flooded_recall": float(recall_score(test_true, test_pred, pos_label=flooded_idx, zero_division=0)),
        "flooded_precision": float(precision_score(test_true, test_pred, pos_label=flooded_idx, zero_division=0)),
        "flooded_f1": float(f1_score(test_true, test_pred, pos_label=flooded_idx, zero_division=0)),
        "roc_auc": float(roc_auc_score(binary_true, test_prob)) if len(np.unique(binary_true)) == 2 else None,
        "pr_auc": float(average_precision_score(binary_true, test_prob)) if len(np.unique(binary_true)) == 2 else None,
        "test_size": len(test_true),
        "flooded_positives": int(binary_true.sum()),
        "_true": [int(v) for v in test_true],
        "_pred": [int(v) for v in test_pred],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified k-fold CV for the Stage 02 CNN")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    image_dir = BASE_DIR / "data" / "images"
    base = vision_datasets.ImageFolder(image_dir)
    labels = base.classes
    flooded_idx = labels.index("flooded")
    targets = np.asarray(base.targets)
    indices = np.arange(len(base))

    print("=" * 70)
    print(f"STAGE 02 CNN - {args.folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    print(f"Images: {len(base)} | classes: {labels} | "
          f"flooded: {int((targets == flooded_idx).sum())}")
    print(f"Architecture: ResNet18 transfer (the configuration selected on validation)")
    print(f"Max epochs/fold: {args.epochs}, early stopping patience 8\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    fold_results, pooled_true, pooled_pred = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, targets), start=1):
        print(f"[Fold {fold}/{args.folds}] training on {len(train_idx)}, "
              f"testing on {len(test_idx)} ...")
        result = run_fold(image_dir, train_idx, test_idx, targets, labels,
                          flooded_idx, args.epochs, args.batch_size)
        pooled_true.extend(result.pop("_true"))
        pooled_pred.extend(result.pop("_pred"))
        fold_results.append(result)
        print(f"  acc={result['accuracy']:.4f}  macroF1={result['macro_f1']:.4f}  "
              f"floodedRecall={result['flooded_recall']:.4f}  "
              f"(n={result['test_size']}, positives={result['flooded_positives']})")

    def summarise(key: str) -> dict:
        values = [f[key] for f in fold_results if f[key] is not None]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary = {key: summarise(key) for key in
               ["accuracy", "macro_f1", "flooded_recall", "flooded_precision",
                "flooded_f1", "roc_auc", "pr_auc"]}

    # Pooled out-of-fold predictions: every image predicted exactly once, by a
    # model that never saw it in training. Bootstrap CIs are computed over this.
    pooled_true_arr = np.asarray(pooled_true)
    pooled_pred_arr = np.asarray(pooled_pred)

    pooled = {
        "n": int(len(pooled_true_arr)),
        "flooded_positives": int((pooled_true_arr == flooded_idx).sum()),
        "accuracy": float(accuracy_score(pooled_true_arr, pooled_pred_arr)),
        "macro_f1": float(f1_score(pooled_true_arr, pooled_pred_arr, average="macro", zero_division=0)),
        "flooded_recall": float(recall_score(pooled_true_arr, pooled_pred_arr, pos_label=flooded_idx, zero_division=0)),
        "flooded_precision": float(precision_score(pooled_true_arr, pooled_pred_arr, pos_label=flooded_idx, zero_division=0)),
    }
    pooled["confidence_intervals_95"] = {
        "accuracy": bootstrap_ci(pooled_true_arr, pooled_pred_arr, accuracy_score),
        "macro_f1": bootstrap_ci(
            pooled_true_arr, pooled_pred_arr,
            lambda a, b: f1_score(a, b, average="macro", zero_division=0)),
        "flooded_recall": bootstrap_ci(
            pooled_true_arr, pooled_pred_arr,
            lambda a, b: recall_score(a, b, pos_label=flooded_idx, zero_division=0)),
        "flooded_precision": bootstrap_ci(
            pooled_true_arr, pooled_pred_arr,
            lambda a, b: precision_score(a, b, pos_label=flooded_idx, zero_division=0)),
    }

    print("\n" + "=" * 70)
    print("CROSS-VALIDATED RESULTS (mean +/- std across folds)")
    print("=" * 70)
    for key, stats in summary.items():
        print(f"  {key:20} {stats['mean']:.4f} +/- {stats['std']:.4f}   "
              f"[min {stats['min']:.4f}, max {stats['max']:.4f}]")

    print("\nPOOLED OUT-OF-FOLD (every image scored once, by a model that never saw it)")
    print(f"  n = {pooled['n']}, flooded positives = {pooled['flooded_positives']}")
    for key in ["accuracy", "macro_f1", "flooded_recall", "flooded_precision"]:
        low, high = pooled["confidence_intervals_95"][key]
        print(f"  {key:20} {pooled[key]:.4f}   95% CI [{low:.4f}, {high:.4f}]")

    report = {
        "folds": args.folds,
        "epochs_max": args.epochs,
        "architecture": "ResNet18 transfer learning",
        "total_images": len(base),
        "per_fold": fold_results,
        "summary_across_folds": summary,
        "pooled_out_of_fold": pooled,
        "interpretation": (
            "The single 70/15/15 held-out split evaluates only 75 images with 15 "
            "flooded positives, so its metrics carry wide uncertainty. These "
            "cross-validated figures use every image exactly once for evaluation. "
            "The fold-to-fold standard deviation and the bootstrap confidence "
            "intervals are the honest precision of the estimate; quote the "
            "interval, not the point estimate alone."
        ),
    }
    path = OUTPUT_DIR / "cnn_cross_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
