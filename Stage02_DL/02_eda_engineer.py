"""Exploratory data analysis for the raw Stage02 deep-learning datasets.

The script reads every tabular file below Stage02_DL/data/raw and inventories
the flood images. Outputs are written below Stage02_DL/data/outputs.
"""

from pathlib import Path
import json
import hashlib
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


def _image_features(image: "Image.Image") -> dict:
	"""Extract explainable quality, color, edge, and water-proxy features."""
	array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
	gray = array.mean(axis=2)
	red, green, blue = array.transpose(2, 0, 1)
	brightness = float(gray.mean())
	contrast = float(gray.std())
	gradient_y, gradient_x = np.gradient(gray)
	sharpness = float((gradient_x ** 2 + gradient_y ** 2).mean())
	saturation = array.max(axis=2) - array.min(axis=2)
	dark = gray < 0.20
	low_saturation = saturation < 0.15
	water_proxy = (blue > red * 1.05) & (blue >= green * 0.95) & (gray > 0.15)
	return {
		"width": int(array.shape[1]),
		"height": int(array.shape[0]),
		"aspect_ratio": round(float(array.shape[1] / max(array.shape[0], 1)), 4),
		"brightness_mean": round(brightness, 6),
		"contrast_std": round(contrast, 6),
		"sharpness_gradient_energy": round(sharpness, 6),
		"dark_pixel_ratio": round(float(dark.mean()), 6),
		"shadow_proxy_ratio": round(float((dark & low_saturation).mean()), 6),
		"wet_asphalt_proxy_ratio": round(float((dark & low_saturation & (blue >= red * 0.9)).mean()), 6),
		"water_color_proxy_ratio": round(float(water_proxy.mean()), 6),
		"edge_density": round(float(((np.abs(gradient_x) + np.abs(gradient_y)) > 0.12).mean()), 6),
	}


def analyze_images(image_files: list[Path]) -> dict:
	"""Run image quality, duplicate, balance, and engineered-feature audits."""
	if Image is None:
		return {"status": "skipped", "reason": "Pillow is required"}
	rows = []
	hashes = {}
	perceptual_hashes = {}
	water_mask_dir = OUTPUT_DIR / "water_masks"
	water_mask_dir.mkdir(parents=True, exist_ok=True)
	for image_path in image_files:
		try:
			file_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
			with Image.open(image_path) as image:
				features = _image_features(image)
				gray_small = np.asarray(image.convert("L").resize((16, 16)), dtype=np.float32)
				average_hash = "".join((gray_small >= gray_small.mean()).astype(np.uint8).flatten().astype(str))
				rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
				red, green, blue = rgb.transpose(2, 0, 1)
				water_mask = ((blue > red * 1.05) & (blue >= green * 0.95) & (rgb.mean(axis=2) > 0.15)).astype(np.uint8) * 255
				Image.fromarray(water_mask).save(water_mask_dir / f"{safe_name(image_path)}_water_mask.png")
			row = {
				"file": str(image_path.relative_to(SCRIPT_DIR)).replace("\\", "/"),
				"class": next((label for label in CNN_CLASSES if label in image_path.parts), "unknown"),
				"capture_source": "cctv" if "cctv" in str(image_path).lower() else ("drone" if "drone" in str(image_path).lower() else "folder_image"),
				"sha256": file_hash,
				"average_hash": average_hash,
				"file_size_bytes": image_path.stat().st_size,
				**features,
			}
			rows.append(row)
			hashes.setdefault(file_hash, []).append(row["file"])
			perceptual_hashes.setdefault(average_hash, []).append(row["file"])
		except (OSError, ValueError) as error:
			rows.append({"file": str(image_path), "error": str(error)})
	frame = pd.DataFrame(rows)
	frame.to_csv(OUTPUT_DIR / "image_quality_features.csv", index=False)
	if frame.empty:
		return {"status": "generated", "images_analyzed": 0}
	duplicates = [files for files in hashes.values() if len(files) > 1]
	perceptual_duplicates = [files for files in perceptual_hashes.values() if len(files) > 1]
	pd.DataFrame(
		[{"duplicate_group": index, "file": file} for index, files in enumerate(duplicates, 1) for file in files]
	).to_csv(OUTPUT_DIR / "image_duplicate_groups.csv", index=False)
	pd.DataFrame(
		[{"duplicate_group": index, "file": file} for index, files in enumerate(perceptual_duplicates, 1) for file in files]
	).to_csv(OUTPUT_DIR / "image_perceptual_duplicate_groups.csv", index=False)
	class_counts = frame["class"].value_counts().rename_axis("class").reset_index(name="count")
	class_counts["percentage"] = (class_counts["count"] / max(len(frame), 1) * 100).round(3)
	class_counts.to_csv(OUTPUT_DIR / "image_class_balance.csv", index=False)
	quality = frame.dropna(subset=["brightness_mean"])
	if not quality.empty:
		quality.describe().round(6).to_csv(OUTPUT_DIR / "image_quality_summary.csv")
	return {
		"status": "generated",
		"images_analyzed": int(len(frame)),
		"duplicate_groups": len(duplicates),
		"perceptual_duplicate_groups": len(perceptual_duplicates),
		"class_balance": class_counts.to_dict("records"),
		"outputs": [
			"data/outputs/image_quality_features.csv",
			"data/outputs/image_quality_summary.csv",
			"data/outputs/image_class_balance.csv",
			"data/outputs/image_duplicate_groups.csv",
			"data/outputs/image_perceptual_duplicate_groups.csv",
			"data/outputs/water_masks/",
		],
	}


def analyze_temporal_tables(table_frames: dict[Path, pd.DataFrame]) -> dict:
	"""Create trend and seasonality summaries for timestamped numeric tables."""
	outputs = []
	for path, frame in table_frames.items():
		datetime_column = next(
			(column for column in frame.columns if any(token in str(column).lower() for token in ("time", "date", "timestamp"))),
			None,
		)
		if datetime_column is None:
			continue
		parsed = pd.to_datetime(frame[datetime_column], errors="coerce")
		numeric = frame.select_dtypes(include="number").columns.tolist()
		if parsed.notna().sum() < 3 or not numeric:
			continue
		working = frame.loc[parsed.notna(), numeric].copy()
		working["_time"] = parsed[parsed.notna()].values
		working = working.sort_values("_time")
		name = safe_name(Path(path).with_suffix(""))
		plot_columns = numeric[:4]
		figure, axes = plt.subplots(len(plot_columns), 1, figsize=(12, max(3, 2.5 * len(plot_columns))), squeeze=False)
		for axis, column in zip(axes[:, 0], plot_columns):
			axis.plot(working["_time"], working[column], linewidth=0.7)
			axis.set_title(f"Trend: {column}")
			axis.set_ylabel(column)
		figure.tight_layout()
		figure.savefig(OUTPUT_DIR / f"{name}_temporal_trends.png", dpi=150)
		plt.close(figure)
		seasonal = working.assign(month=working["_time"].dt.month).groupby("month")[plot_columns].mean()
		seasonal.to_csv(OUTPUT_DIR / f"{name}_seasonality.csv")
		outputs.append({"file": str(path.relative_to(SCRIPT_DIR)).replace("\\", "/"), "datetime_column": str(datetime_column), "numeric_columns": plot_columns})
	return {"status": "generated", "tables_analyzed": len(outputs), "tables": outputs}


def analyze_satellite_masks() -> dict:
	"""Summarize embedded satellite masks and image brightness when available."""
	satellite_dir = RAW_DIR / SOURCE_DIRECTORIES["satellite"] / "dataset" / "train"
	if not (satellite_dir / "state.json").exists():
		return {"status": "skipped", "reason": "Satellite dataset not available"}
	try:
		from datasets import load_from_disk
		dataset = load_from_disk(str(satellite_dir))
		rows = []
		for index in range(len(dataset)):
			image = np.asarray(dataset[index]["image"].convert("RGB"), dtype=np.float32) / 255.0
			mask = np.asarray(dataset[index]["mask"])
			mask = mask[..., 0] if mask.ndim == 3 else mask
			rows.append({"index": index, "file_name": str(dataset[index].get("file_name", index)), "mask_coverage": float((mask > 0).mean()), "image_brightness": float(image.mean())})
		frame = pd.DataFrame(rows)
		frame.to_csv(OUTPUT_DIR / "satellite_change_analysis.csv", index=False)
		return {"status": "generated", "samples_analyzed": len(frame), "output": "data/outputs/satellite_change_analysis.csv", "note": "Mask coverage is a flood-area proxy; true temporal change requires dated repeat imagery."}
	except Exception as error:
		return {"status": "skipped", "reason": str(error)}


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
	table_frames = {}
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
			table_frames[tabular_file] = frame
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
		image_analysis = analyze_images(image_files)
		print(f"Image analysis: {image_analysis['status']}")
		saliency_audit = generate_saliency_audit(image_files)
		print(f"Saliency audit: {saliency_audit['status']}")
	else:
		image_analysis = {"status": "skipped", "reason": "No flood images found"}
		saliency_audit = {"status": "skipped", "reason": "No flood images found"}
	temporal_analysis = analyze_temporal_tables(table_frames)
	satellite_analysis = analyze_satellite_masks()
	print(f"Temporal analysis: {temporal_analysis['status']}")
	print(f"Satellite mask analysis: {satellite_analysis['status']}")

	summary_path = OUTPUT_DIR / "raw_eda_summary.json"
	summary_path.write_text(
		json.dumps(
			{
				"tabular_datasets": summaries,
				"image_datasets": image_inventory,
				"image_analysis": image_analysis,
				"temporal_analysis": temporal_analysis,
				"satellite_analysis": satellite_analysis,
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
