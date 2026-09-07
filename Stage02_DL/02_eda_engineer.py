"""Exploratory data analysis for the raw Stage02 deep-learning datasets.

The script reads every tabular file below Stage02_DL/data/raw and inventories
the flood images. Outputs are written below Stage02_DL/data/outputs.
"""

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
	from PIL import Image
except ImportError:
	Image = None

try:
	import torch
	from torchvision import transforms
except ImportError:
	torch = None
	transforms = None


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "data" / "raw"
OUTPUT_DIR = SCRIPT_DIR / "data" / "outputs"
SOURCE_DIRECTORIES = {
	"satellite": "01_SATELLITE_FLOOD_DATASET",
	"rainfall": "02_RAINFALL_DATASET",
	"river_water_level": "03_RIVER_WATER_LEVEL_DATASET",
	"master": "04_MASTER_DATASET",
	"metadata": "05_METADATA",
	"flood_images": "Flood_Image_Dataset",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CNN_CLASSES = ["flooded", "unflooded"]


def safe_name(path: Path) -> str:
	"""Return a filesystem-safe name for a dataset."""
	name = path.stem.lower()
	return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def read_raw_table(path: Path) -> pd.DataFrame:
	"""Read one complete raw tabular file for exploratory analysis."""
	suffix = path.suffix.lower()
	if suffix == ".csv":
		return pd.read_csv(path, low_memory=False)
	if suffix == ".json":
		return pd.read_json(path)
	if suffix == ".arrow":
		return pd.read_feather(path)
	raise ValueError(f"Unsupported tabular format: {path}")


def read_satellite_dataset(dataset_dir: Path) -> pd.DataFrame:
	"""Load the Hugging Face satellite dataset and summarize its features."""
	try:
		from datasets import load_from_disk
	except ImportError as error:
		raise ImportError(
			"Install the 'datasets' package to read the satellite dataset"
		) from error

	dataset = load_from_disk(str(dataset_dir))
	return pd.DataFrame(
		{
			column: [str(value) for value in dataset[column]]
			for column in dataset.column_names
		}
	)


def dataset_summary(path: Path, frame: pd.DataFrame) -> dict:
	"""Build a JSON-serializable summary for one raw dataset."""
	numeric = frame.select_dtypes(include="number")
	return {
		"file": str(path.relative_to(SCRIPT_DIR)).replace("\\", "/"),
		"rows_analyzed": int(len(frame)),
		"columns": int(frame.shape[1]),
		"column_names": [str(column) for column in frame.columns],
		"missing_values": {
			str(column): int(value)
			for column, value in frame.isna().sum().items()
			if value
		},
		"duplicate_rows": int(frame.duplicated().sum()),
		"numeric_summary": json.loads(
			numeric.describe().round(4).to_json()
		)
		if not numeric.empty
		else {},
	}


def save_numeric_plot(name: str, frame: pd.DataFrame) -> None:
	"""Save a compact histogram and correlation chart for numeric columns."""
	numeric = frame.select_dtypes(include="number")
	if numeric.empty:
		return

	plot_name = safe_name(Path(name))
	numeric.hist(figsize=(14, 9), bins=30)
	plt.suptitle(f"Numeric distributions: {name}")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / f"{plot_name}_distributions.png", dpi=150)
	plt.close("all")

	if numeric.shape[1] > 1:
		correlation = numeric.corr()
		figure, axis = plt.subplots(figsize=(10, 8))
		image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
		axis.set_xticks(range(len(correlation.columns)))
		axis.set_yticks(range(len(correlation.columns)))
		axis.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
		axis.set_yticklabels(correlation.columns, fontsize=7)
		figure.colorbar(image, ax=axis, label="Correlation")
		axis.set_title(f"Numeric correlation: {name}")
		figure.tight_layout()
		figure.savefig(OUTPUT_DIR / f"{plot_name}_correlation.png", dpi=150)
		plt.close(figure)


def image_summary(source_name: str, image_files: list[Path]) -> dict:
	"""Summarize image count, file sizes, and dimensions for a raw image source."""
	widths = []
	heights = []
	file_sizes = []
	for image_file in image_files:
		file_sizes.append(image_file.stat().st_size)
		if Image is not None:
			try:
				with Image.open(image_file) as image:
					width, height = image.size
				widths.append(width)
				heights.append(height)
			except (OSError, ValueError):
				continue

	return {
		"source": source_name,
		"file_count": len(image_files),
		"total_size_mb": round(sum(file_sizes) / (1024 * 1024), 3),
		"dimensions": {
			"width_min": min(widths) if widths else None,
			"width_max": max(widths) if widths else None,
			"height_min": min(heights) if heights else None,
			"height_max": max(heights) if heights else None,
		},
		"pillow_available": Image is not None,
	}


def generate_saliency_audit(image_files: list[Path]) -> dict:
	"""Create gradient saliency overlays for a balanced CNN image audit.

	The dark-pixel overlap is only a screening statistic. It cannot prove that
	the model focuses on shadows because the folder-image dataset has no
	pixel-level water/shadow annotations.
	"""
	if torch is None or transforms is None or Image is None:
		return {"status": "skipped", "reason": "torch, torchvision, and Pillow are required"}
	checkpoint_path = SCRIPT_DIR / "data" / "models" / "disaster_cnn.pt"
	if not checkpoint_path.exists():
		return {"status": "skipped", "reason": f"CNN checkpoint not found: {checkpoint_path}"}

	try:
		import importlib.util
		spec = importlib.util.spec_from_file_location(
			"stage02_dl_for_eda", SCRIPT_DIR / "03_dl_engineer.py"
		)
		module = importlib.util.module_from_spec(spec)
		assert spec.loader is not None
		spec.loader.exec_module(module)
	except Exception as error:
		return {"status": "skipped", "reason": f"Unable to load CNN architecture: {error}"}

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model = module.DisasterCNN(len(checkpoint["classes"])).to(device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	transform = transforms.Compose([
		transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
		transforms.ToTensor(), normalize,
	])
	selected = []
	for label in CNN_CLASSES:
		selected.extend(
			path for path in image_files
			if label in path.parts
		) 
	selected = selected[:3] + [path for path in selected if "unflooded" in path.parts][:3]
	selected = list(dict.fromkeys(selected))[:6]
	if not selected:
		return {"status": "skipped", "reason": "No class-folder images found"}

	output_dir = OUTPUT_DIR / "saliency_maps"
	output_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	for image_path in selected:
		with Image.open(image_path) as source_image:
			image = source_image.convert("RGB")
			input_tensor = transform(image).unsqueeze(0).to(device)
			input_tensor.requires_grad_(True)
			model.zero_grad(set_to_none=True)
			logits = model(input_tensor)
			predicted_index = int(logits.argmax(1).item())
			logits[0, predicted_index].backward()
			saliency = input_tensor.grad.detach().abs().max(dim=1).values[0].cpu().numpy()
			saliency = (saliency - saliency.min()) / (np.ptp(saliency) + 1e-8)
			base = np.asarray(image.resize((saliency.shape[1], saliency.shape[0])), dtype=np.float32) / 255
			dark_pixels = base.mean(axis=2) < 0.20
			top_pixels = saliency >= np.percentile(saliency, 90)
			dark_overlap = float((top_pixels & dark_pixels).sum() / max(top_pixels.sum(), 1))
			figure, axes = plt.subplots(1, 3, figsize=(12, 4))
			axes[0].imshow(base)
			axes[0].set_title("Input")
			axes[1].imshow(saliency, cmap="magma")
			axes[1].set_title("Gradient saliency")
			axes[2].imshow(base)
			axes[2].imshow(saliency, cmap="jet", alpha=0.45)
			axes[2].set_title("Saliency overlay")
			for axis in axes:
				axis.axis("off")
			figure.suptitle(f"Actual: {image_path.parent.name} | Predicted: {checkpoint['classes'][predicted_index]}")
			figure.tight_layout()
			output_path = output_dir / f"{safe_name(image_path)}_saliency.png"
			figure.savefig(output_path, dpi=150)
			plt.close(figure)
		rows.append({
			"image": str(image_path.relative_to(SCRIPT_DIR)).replace("\\", "/"),
			"actual_class": image_path.parent.name,
			"predicted_class": checkpoint["classes"][predicted_index],
			"top_10_percent_saliency_dark_pixel_overlap": dark_overlap,
			"overlay": str(output_path.relative_to(SCRIPT_DIR)).replace("\\", "/"),
		})

	pd.DataFrame(rows).to_csv(OUTPUT_DIR / "saliency_audit.csv", index=False)
	average_overlap = float(
		pd.DataFrame(rows)["top_10_percent_saliency_dark_pixel_overlap"].mean()
	)
	flagged = int(
		(
			pd.DataFrame(rows)["top_10_percent_saliency_dark_pixel_overlap"]
			> 0.30
		).sum()
	)
	review = f"""# CNN Saliency Review

## Purpose

This EDA step checks whether the trained CNN's strongest gradient-saliency
regions appear to rely on dark image regions. It is a screening step for the
failure mode of learning dark shadows instead of floodwater.

## Samples

- Images reviewed: {len(rows)}
- Flooded examples: {sum(row['actual_class'] == 'flooded' for row in rows)}
- Unflooded examples: {sum(row['actual_class'] == 'unflooded' for row in rows)}
- Mean dark-pixel overlap in the top 10% saliency: {average_overlap:.4f}
- Samples above the 0.30 screening threshold: {flagged}

## Interpretation

The overlays in `data/outputs/saliency_maps/` must be visually reviewed for
water-shaped regions, edges, roads, buildings, and shadows. The dark-pixel
overlap is not a water mask and cannot prove semantic floodwater focus.
Because this dataset has no pixel-level water/shadow annotations, this audit
must report evidence rather than claim verification.

The CNN's flooded-class recall and prediction errors must be considered with
this review. A low dark-pixel overlap does not guarantee reliable floodwater
recognition, and a high overlap is a warning that shadow-related shortcuts may
be present.
"""
	(OUTPUT_DIR / "saliency_review.md").write_text(review, encoding="utf-8")
	return {
		"status": "generated",
		"images_reviewed": len(rows),
		"mean_dark_pixel_overlap": average_overlap,
		"samples_above_dark_overlap_threshold": flagged,
		"maps_directory": str(output_dir.relative_to(SCRIPT_DIR)).replace("\\", "/"),
		"audit_csv": "data/outputs/saliency_audit.csv",
		"review_report": "data/outputs/saliency_review.md",
		"interpretation": "Review overlays manually; dark-pixel overlap is a screening statistic, not a water/shadow label.",
	}


def source_name_for(path: Path) -> str:
	"""Return the configured raw source name for a file path."""
	for source_name, directory_name in SOURCE_DIRECTORIES.items():
		if directory_name in path.parts:
			return source_name
	return "other"


def main() -> None:
	"""Analyze all raw Stage02 tabular data and flood images."""
	if not RAW_DIR.exists():
		raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	tabular_files = sorted(
		path
		for path in RAW_DIR.rglob("*")
		if path.is_file() and path.suffix.lower() in {".csv", ".json", ".arrow"}
	)
	image_files = sorted(
		path
		for path in RAW_DIR.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	)
	if not tabular_files and not image_files:
		raise FileNotFoundError(f"No raw datasets found below: {RAW_DIR}")

	summaries = []
	satellite_dir = RAW_DIR / SOURCE_DIRECTORIES["satellite"] / "dataset" / "train"
	if satellite_dir.exists() and (satellite_dir / "state.json").exists():
		try:
			frame = read_satellite_dataset(satellite_dir)
			summary = dataset_summary(
				satellite_dir / "data-00000-of-00001.arrow", frame
			)
			summary["source"] = "satellite"
			summaries.append(summary)
			print(f"Analyzed satellite dataset: {frame.shape}")
		except Exception as error:
			print(f"Skipped satellite dataset {satellite_dir}: {error}")

	for tabular_file in tabular_files:
		try:
			if satellite_dir in tabular_file.parents:
				continue
			frame = read_raw_table(tabular_file)
			summary = dataset_summary(tabular_file, frame)
			summary["source"] = source_name_for(tabular_file)
			summaries.append(summary)
			save_numeric_plot(
				f"{summary['source']}_{tabular_file.stem}", frame
			)
			print(
				f"Analyzed {summary['source']}: "
				f"{tabular_file.name} {frame.shape}"
			)
		except (OSError, ValueError, ImportError, pd.errors.ParserError,
			UnicodeDecodeError) as error:
			print(f"Skipped {tabular_file}: {error}")
		except Exception as error:
			print(f"Skipped unreadable raw file {tabular_file}: {error}")

	image_inventory = []
	if image_files:
		image_inventory.append(image_summary("flood_images", image_files))
		print(f"Inventoried flood_images: {len(image_files)} images")
		saliency_audit = generate_saliency_audit(image_files)
		print(f"Saliency audit: {saliency_audit['status']}")
	else:
		saliency_audit = {"status": "skipped", "reason": "No flood images found"}

	summary_path = OUTPUT_DIR / "raw_eda_summary.json"
	summary_path.write_text(
		json.dumps(
			{
				"tabular_datasets": summaries,
				"image_datasets": image_inventory,
				"saliency_audit": saliency_audit,
			},
			indent=2,
		),
		encoding="utf-8",
	)
	pd.DataFrame(
		[
			{
				"file": item["file"],
				"rows_analyzed": item["rows_analyzed"],
				"columns": item["columns"],
				"duplicate_rows": item["duplicate_rows"],
				"missing_values": sum(item["missing_values"].values()),
			}
			for item in summaries
		]
	).to_csv(OUTPUT_DIR / "raw_eda_dataset_inventory.csv", index=False)
	if image_inventory:
		pd.DataFrame(image_inventory).to_csv(
			OUTPUT_DIR / "raw_flood_image_inventory.csv", index=False
		)
	print(f"Saved EDA outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
