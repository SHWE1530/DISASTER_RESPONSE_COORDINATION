"""EDA for vocabulary and phrasing patterns in the Stage03 NLP datasets.

The analysis compares urgent and routine reports across the dispatcher log,
social feeds, and safety SOP corpus. Results are written to data/outputs.
"""

from collections import Counter
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"
OUTPUT_DIR = SCRIPT_DIR / "data" / "outputs"

DATASETS = {
	"dispatcher": "Dispatcher_Log_Master_60000_Processed.csv",
	"safety_sop": "Safety_SOP_NLP_Dataset_PROCESSED.csv",
	"social_feeds": "Social_Feeds_India_Processed.csv",
}

STOPWORDS = {
	"a", "about", "after", "all", "also", "an", "and", "are", "around",
	"as", "at", "be", "been", "being", "by", "can", "for", "from", "had",
	"has", "have", "in", "into", "is", "it", "its", "may", "more", "near",
	"of", "on", "or", "our", "reported", "send", "that", "the", "their",
	"there", "these", "this", "to", "was", "we", "were", "will", "with",
	"you", "your",
}
URGENT_TERMS = {
	"urgent", "urgently", "immediate", "immediately", "emergency", "critical",
	"trapped", "evacuation", "evacuate", "rescue", "distress", "danger",
	"warning", "required", "need", "needed", "injuries", "injury",
}
TOKEN_PATTERN = re.compile(r"[a-z][a-z']+")


def clean_text(value: object) -> str:
	"""Normalize a cell to lowercase text suitable for token analysis."""
	text = "" if pd.isna(value) else str(value).lower()
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
	"""Return filtered word tokens while retaining useful domain vocabulary."""
	return [
		token for token in TOKEN_PATTERN.findall(clean_text(text))
		if token not in STOPWORDS and len(token) > 1
	]


def classify_urgency(row: pd.Series, text: str, dataset: str) -> str:
	"""Create a comparable urgent/routine label for each source schema."""
	if dataset == "dispatcher":
		severity = clean_text(row.get("severity", ""))
		return "urgent" if severity in {"high", "critical"} else "routine"
	if dataset == "social_feeds":
		level = clean_text(row.get("urgency_level", ""))
		return "urgent" if level in {"high", "critical"} else "routine"
	matches = sum(term in text.split() for term in URGENT_TERMS)
	return "urgent" if matches else "routine"


def load_reports(dataset: str, filename: str) -> pd.DataFrame:
	"""Load one processed CSV and map its text to a common analysis schema."""
	path = PROCESSED_DIR / filename
	if not path.exists():
		raise FileNotFoundError(f"Processed dataset not found: {path}")
	frame = pd.read_csv(path, low_memory=False)
	text_column = {
		"dispatcher": "text",
		"safety_sop": "text_clean",
		"social_feeds": "clean_text",
	}[dataset]
	if text_column not in frame.columns:
		raise ValueError(f"{path.name} is missing required column '{text_column}'")
	result = pd.DataFrame({"text": frame[text_column].map(clean_text)})
	result["dataset"] = dataset
	result["urgency"] = [
		classify_urgency(row, text, dataset)
		for (_, row), text in zip(frame.iterrows(), result["text"])
	]
	result["tokens"] = result["text"].map(tokens)
	result["token_count"] = result["tokens"].map(len)
	result["unique_token_count"] = result["tokens"].map(lambda values: len(set(values)))
	return result


def count_ngrams(token_rows: pd.Series, n: int) -> Counter:
	"""Count contiguous n-word phrases across a series of token lists."""
	counts = Counter()
	for row_tokens in token_rows:
		counts.update(" ".join(row_tokens[index:index + n])
					  for index in range(len(row_tokens) - n + 1))
	return counts


def counter_frame(counts: Counter, total: int, top_n: int) -> pd.DataFrame:
	"""Convert counts into a stable table with counts and normalized rates."""
	rows = [
		{"term": term, "count": count, "per_1000_tokens": round(count / max(total, 1) * 1000, 4)}
		for term, count in counts.most_common(top_n)
	]
	return pd.DataFrame(rows, columns=["term", "count", "per_1000_tokens"])


def compare_vocabulary(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
	"""Rank terms by frequency difference between urgent and routine reports."""
	groups = {}
	for label in ("urgent", "routine"):
		counts = Counter(token for row in frame.loc[frame["urgency"] == label, "tokens"] for token in row)
		groups[label] = counts
	urgent_total = sum(groups["urgent"].values())
	routine_total = sum(groups["routine"].values())
	terms = set(groups["urgent"]) | set(groups["routine"])
	rows = []
	for term in terms:
		urgent_rate = groups["urgent"][term] / max(urgent_total, 1) * 1000
		routine_rate = groups["routine"][term] / max(routine_total, 1) * 1000
		rows.append({
			"term": term,
			"urgent_count": groups["urgent"][term],
			"routine_count": groups["routine"][term],
			"urgent_per_1000_tokens": round(urgent_rate, 4),
			"routine_per_1000_tokens": round(routine_rate, 4),
			"rate_difference": round(urgent_rate - routine_rate, 4),
			"log_rate_ratio": round(math.log((urgent_rate + 0.1) / (routine_rate + 0.1)), 4),
		})
	result = pd.DataFrame(rows)
	return result.sort_values(
		["rate_difference", "term"], ascending=[False, True]
	).head(top_n)


def analyze_source(frame: pd.DataFrame, dataset: str, top_n: int) -> dict:
	"""Write vocabulary and phrasing outputs for one dataset."""
	source_dir = OUTPUT_DIR / dataset
	source_dir.mkdir(parents=True, exist_ok=True)
	outputs = []
	for urgency in ("urgent", "routine"):
		subset = frame[frame["urgency"] == urgency]
		all_tokens = [token for row in subset["tokens"] for token in row]
		prefix = f"{dataset}_{urgency}"
		counter_frame(Counter(all_tokens), len(all_tokens), top_n).to_csv(
			source_dir / f"{prefix}_vocabulary.csv", index=False)
		outputs.append(f"{dataset}/{prefix}_vocabulary.csv")
		for n, name in ((2, "bigrams"), (3, "trigrams")):
			counter_frame(count_ngrams(subset["tokens"], n), len(all_tokens), top_n).rename(
				columns={"term": "phrase"}
			).to_csv(source_dir / f"{prefix}_{name}.csv", index=False)
			outputs.append(f"{dataset}/{prefix}_{name}.csv")
	compare_vocabulary(frame, top_n).to_csv(source_dir / "urgent_vs_routine_vocabulary.csv", index=False)
	outputs.append(f"{dataset}/urgent_vs_routine_vocabulary.csv")
	summary = {
		"dataset": dataset,
		"reports": int(len(frame)),
		"urgent_reports": int((frame["urgency"] == "urgent").sum()),
		"routine_reports": int((frame["urgency"] == "routine").sum()),
		"mean_tokens": round(float(frame["token_count"].mean()), 4),
		"mean_unique_tokens": round(float(frame["unique_token_count"].mean()), 4),
		"outputs": outputs,
	}
	return summary


def save_plots(all_frames: pd.DataFrame, top_n: int) -> list[str]:
	"""Create compact comparison plots for report length and vocabulary size."""
	plot_paths = []
	for metric, title, filename in (
		("token_count", "Report length by urgency", "report_length_by_urgency.png"),
		("unique_token_count", "Vocabulary size by urgency", "vocabulary_size_by_urgency.png"),
	):
		figure, axis = plt.subplots(figsize=(10, 6))
		all_frames.boxplot(column=metric, by=["dataset", "urgency"], ax=axis, grid=False)
		axis.set_title(title)
		axis.set_xlabel("Dataset and urgency")
		axis.set_ylabel(metric.replace("_", " ").title())
		figure.suptitle("")
		figure.tight_layout()
		figure.savefig(OUTPUT_DIR / filename, dpi=150)
		plt.close(figure)
		plot_paths.append(f"{filename}")
	return plot_paths


def run(top_n: int = 30) -> dict:
	"""Run the complete NLP EDA workflow and return its manifest."""
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	frames = [load_reports(dataset, filename) for dataset, filename in DATASETS.items()]
	combined = pd.concat(frames, ignore_index=True)
	summaries = [analyze_source(frame, dataset, top_n) for frame, dataset in zip(frames, DATASETS)]
	combined.to_json(OUTPUT_DIR / "normalized_report_metadata.json", orient="records")
	plot_paths = save_plots(combined, top_n)
	manifest = {
		"analysis": "urgent versus routine vocabulary and phrasing",
		"top_n": top_n,
		"total_reports": int(len(combined)),
		"sources": summaries,
		"plots": plot_paths,
	}
	(OUTPUT_DIR / "eda_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	return manifest


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--top-n", type=int, default=30, help="Terms and phrases to retain per table")
	args = parser.parse_args()
	if args.top_n < 1:
		parser.error("--top-n must be at least 1")
	manifest = run(args.top_n)
	print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
	main()
