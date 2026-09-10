"""Production NLP Pipeline for Stage 03 Disaster Response Coordination.

This module implements:
1. 4-Class Urgency Level Text Classifier (LOW, MEDIUM, HIGH, CRITICAL).
2. Multi-Class Hazard Type Text Classifier.
3. Token-Level BIO Named-Entity Recognizer (NER) for LOCATION, RESOURCE, and HEADCOUNT.
4. Hybrid Pattern & Gazetteer Fallback Engine.
5. Misinterpretation & Ambiguity Audit Logging.
6. Public Inference API (predict_urgency, predict_hazard, extract_entities, analyze_text, process_batch).

Classifier: Logistic Regression classifier with predict_proba probability estimates.
NER Model: Lightweight token-level BIO classifier with contextual token features.

Primary inputs:
- Stage03_NLP/data/processed/Dispatcher_Log_Master_60000_Processed.csv
- Stage03_NLP/data/processed/Social_Feeds_India_Processed.csv
- Stage03_NLP/data/processed/Safety_SOP_NLP_Dataset_PROCESSED.csv
- Stage03_NLP/data/outputs/dispatcher_ner_bio.jsonl
- Stage03_NLP/data/outputs/social_feeds_ner_bio.jsonl
- Stage03_NLP/data/outputs/safety_sop_ner_bio.jsonl

Model artifacts stored under Stage03_NLP/data/models/
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

if TRANSFORMERS_AVAILABLE:
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.encodings.input_ids)



BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
MODEL_DIR = BASE_DIR / "data" / "models"
SEED = 42

# Capping Dispatcher records to MAX_DISPATCHER_NER_RECORDS (10,000) prevents 60,000
# synthetic dispatcher sequences from dominating 3,158 social feed sequences.
MAX_DISPATCHER_NER_RECORDS = 10000

URGENCY_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

URGENCY_MAP = {
	"low": "LOW",
	"medium": "MEDIUM",
	"high": "HIGH",
	"critical": "CRITICAL",
}

TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b")
NUMBER_WORD_MAP = {
	"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
	"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
	"eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
	"thirty": 30, "fifty": 50, "hundred": 100,
}

_LOADED_ARTIFACTS: dict[str, Any] = {}


def ensure_directories() -> None:
	"""Ensure essential output and model directories exist."""
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	MODEL_DIR.mkdir(parents=True, exist_ok=True)


def atomic_joblib_dump(obj: Any, path: Path) -> None:
	"""Write a joblib artifact atomically: serialise to a temp file, then rename.

	joblib.dump() truncates the destination before serialising. When a dump
	failed part-way (e.g. an unpicklable object), it left the previously working
	model as a 2-byte stub -- so a failed retrain destroyed the deployed model.
	os.replace() is atomic on the same filesystem, so the destination either
	holds the old artifact or the complete new one, never a partial write.
	"""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temp_path = path.with_suffix(path.suffix + ".tmp")
	try:
		joblib.dump(obj, temp_path)
		os.replace(temp_path, path)
	finally:
		if temp_path.exists():
			temp_path.unlink(missing_ok=True)


def atomic_write_text(text: str, path: Path) -> None:
	"""Write a text artifact atomically (same rationale as atomic_joblib_dump)."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temp_path = path.with_suffix(path.suffix + ".tmp")
	try:
		temp_path.write_text(text, encoding="utf-8")
		os.replace(temp_path, path)
	finally:
		if temp_path.exists():
			temp_path.unlink(missing_ok=True)


def clean_text_basic(text: Any) -> str:
	"""Basic text cleaning preserving word casing and numeric structure for NER."""
	if text is None or pd.isna(text):
		return ""
	cleaned = str(text).strip()
	cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
	cleaned = re.sub(r"@\w+", " ", cleaned)
	cleaned = re.sub(r"#", " ", cleaned)
	cleaned = re.sub(r"\s+", " ", cleaned).strip()
	return cleaned


def clean_text_for_classification(text: Any) -> str:
	"""Normalized text cleaning suitable for TF-IDF classification."""
	raw = clean_text_basic(text).lower()
	raw = re.sub(r"[^a-zA-Z0-9\s]", " ", raw)
	return re.sub(r"\s+", " ", raw).strip()


def extract_token_features(tokens: list[str], index: int) -> dict[str, Any]:
	"""Extract word-level and contextual features for fast token-level sequence tagging."""
	word = tokens[index]
	w_lower = word.lower()
	feats = {
		"w": w_lower,
		"len": len(word),
		"title": word.istitle(),
		"upper": word.isupper(),
		"digit": word.isdigit(),
		"pref2": w_lower[:2],
		"suff2": w_lower[-2:],
	}
	if index > 0:
		feats["-1:w"] = tokens[index - 1].lower()
		feats["-1:title"] = tokens[index - 1].istitle()
	if index < len(tokens) - 1:
		feats["+1:w"] = tokens[index + 1].lower()
		feats["+1:title"] = tokens[index + 1].istitle()
	return feats


class TokenLevelNERModel:
	"""Lightweight token-level BIO classifier with contextual token features."""

	def __init__(self) -> None:
		self.vectorizer = DictVectorizer(sparse=True)
		self.classifier = LogisticRegression(
			max_iter=200,
			class_weight="balanced",
			random_state=SEED,
			solver="lbfgs",
		)
		self.classes_: list[str] = []

	def fit(self, sequences_tokens: list[list[str]], sequences_tags: list[list[str]]) -> TokenLevelNERModel:
		X_feats, y_tags = [], []
		for tokens, tags in zip(sequences_tokens, sequences_tags):
			for index in range(len(tokens)):
				X_feats.append(extract_token_features(tokens, index))
				y_tags.append(tags[index])

		X_vec = self.vectorizer.fit_transform(X_feats)
		self.classifier.fit(X_vec, y_tags)
		self.classes_ = list(self.classifier.classes_)
		return self

	def predict_sequence(self, tokens: list[str]) -> list[str]:
		if not tokens:
			return []
		feats = [extract_token_features(tokens, index) for index in range(len(tokens))]
		X_vec = self.vectorizer.transform(feats)
		return list(self.classifier.predict(X_vec))

	def to_components(self) -> dict[str, Any]:
		"""Serialize as plain sklearn objects rather than as an instance of this class.

		Pickling the instance itself embedded a reference to whatever module name
		this file happened to be loaded under. Because the file is loaded through
		importlib with a synthetic name ("stage03_nlp_engineer", "__main__", or a
		pytest-assigned name depending on the caller), unpickling failed with
		"Can't pickle <class 'stage03_nlp_engineer.TokenLevelNERModel'>" and the
		module worked around it by mutating sys.modules at load time.
		Round-tripping the fitted components avoids the problem entirely.
		"""
		return {
			"format": "token_level_ner_components_v1",
			"vectorizer": self.vectorizer,
			"classifier": self.classifier,
			"classes": list(self.classes_),
		}

	@classmethod
	def from_components(cls, payload: dict[str, Any]) -> TokenLevelNERModel:
		"""Rebuild from to_components() output, tolerating legacy whole-object pickles."""
		if isinstance(payload, cls):
			return payload
		if not isinstance(payload, dict) or "vectorizer" not in payload:
			raise ValueError(
				"Unrecognised NER artifact format. Retrain Stage 03 with "
				"`python Stage03_NLP/03_nlp_engineer.py`."
			)
		model = cls()
		model.vectorizer = payload["vectorizer"]
		model.classifier = payload["classifier"]
		model.classes_ = list(payload.get("classes", []))
		return model


def load_classification_datasets() -> pd.DataFrame:
	"""Load and harmonize Dispatcher Log and Social Feeds datasets for classification."""
	disp_path = PROCESSED_DIR / "Dispatcher_Log_Master_60000_Processed.csv"
	social_path = PROCESSED_DIR / "Social_Feeds_India_Processed.csv"

	records = []
	if disp_path.exists():
		df_disp = pd.read_csv(disp_path, low_memory=False)
		for _, row in df_disp.iterrows():
			raw_text = clean_text_basic(row.get("text"))
			urg_raw = str(row.get("severity", "")).strip().lower()
			urgency = URGENCY_MAP.get(urg_raw, None)
			hazard = str(row.get("hazard_type", "")).strip()
			if urgency and hazard:
				records.append({
					"source": "dispatcher",
					"text_raw": raw_text,
					"text_clean": clean_text_for_classification(raw_text),
					"urgency": urgency,
					"hazard": hazard,
					# Dispatcher hazard comes from the structured call_type field of
					# the dispatch record, so the keyword lookup is the same value.
					"hazard_baseline": hazard,
				})

	if social_path.exists():
		df_social = pd.read_csv(social_path, low_memory=False)
		for _, row in df_social.iterrows():
			raw_text = clean_text_basic(row.get("text") or row.get("clean_text"))
			urg_raw = str(row.get("urgency_level", "")).strip().lower()
			urgency = URGENCY_MAP.get(urg_raw, None)
			hazard = str(row.get("hazard_type", "")).strip()
			if urgency and hazard:
				records.append({
					"source": "social_feeds",
					"text_raw": raw_text,
					"text_clean": clean_text_for_classification(raw_text),
					"urgency": urgency,
					"hazard": hazard,
					# Keyword-lookup prediction, carried so the evaluation can
					# report what the trained model adds over a plain lookup.
					"hazard_baseline": str(
						row.get("hazard_keyword_baseline") or hazard
					).strip(),
				})

	combined = pd.DataFrame(records)
	if combined.empty:
		raise RuntimeError("No classification records could be loaded from processed datasets.")
	return combined


def load_bio_ner_datasets() -> tuple[list[list[str]], list[list[str]], dict[str, Any]]:
	"""Load official BIO JSONL annotations and validate token-tag alignment.
	
	Note: Capping Dispatcher records to MAX_DISPATCHER_NER_RECORDS (10,000) reduces
	source imbalance so synthetic templates do not overwhelm real social feeds.
	"""
	bio_files = [
		("dispatcher", OUTPUT_DIR / "dispatcher_ner_bio.jsonl", MAX_DISPATCHER_NER_RECORDS),
		("social_feeds", OUTPUT_DIR / "social_feeds_ner_bio.jsonl", None),
		("safety_sop", OUTPUT_DIR / "safety_sop_ner_bio.jsonl", None),
	]

	all_tokens: list[list[str]] = []
	all_tags: list[list[str]] = []
	stats = {"total_records": 0, "malformed_records": 0, "used_records": 0, "sources": {}}

	target_tags = {"B-LOCATION", "I-LOCATION", "B-RESOURCE", "I-RESOURCE", "B-HEADCOUNT", "I-HEADCOUNT", "O"}

	for source_name, filepath, max_records in bio_files:
		source_count, source_used = 0, 0
		if not filepath.exists():
			continue

		with open(filepath, "r", encoding="utf-8") as f:
			for line_num, line in enumerate(f, 1):
				source_count += 1
				stats["total_records"] += 1
				if max_records and source_used >= max_records:
					continue
				try:
					rec = json.loads(line)
					tokens = rec.get("tokens", [])
					tags = rec.get("ner_tags", [])
					if not tokens or len(tokens) != len(tags):
						stats["malformed_records"] += 1
						continue
					
					if source_name == "safety_sop":
						if not any(t in target_tags and t != "O" for t in tags):
							continue
							
					all_tokens.append(tokens)
					all_tags.append(tags)
					source_used += 1
					stats["used_records"] += 1
				except Exception:
					stats["malformed_records"] += 1

		stats["sources"][source_name] = {"total": source_count, "used": source_used}

	if not all_tokens:
		raise RuntimeError("No valid BIO NER records loaded from output JSONL files.")
	return all_tokens, all_tags, stats


def evaluate_ner_entities(
	model: TokenLevelNERModel,
	sequences_tokens: list[list[str]],
	sequences_tags: list[list[str]],
) -> dict[str, Any]:
	"""Entity-level precision / recall / F1 in addition to token accuracy.

	Token accuracy is a misleading headline for NER: the tag distribution is
	dominated by "O", so a model that predicts "O" everywhere already scores
	very highly. An entity is counted correct only when its type and its full
	span both match exactly.
	"""

	def spans(tags: list[str]) -> set[tuple[str, int, int]]:
		found: set[tuple[str, int, int]] = set()
		current_type: str | None = None
		start = 0
		for index, tag in enumerate(list(tags) + ["O"]):
			if tag.startswith("B-") or tag == "O" or (
				tag.startswith("I-") and current_type != tag[2:]
			):
				if current_type is not None:
					found.add((current_type, start, index))
					current_type = None
			if tag.startswith("B-"):
				current_type = tag[2:]
				start = index
			elif tag.startswith("I-") and current_type is None:
				# Treat a stray I- as the start of an entity rather than dropping it.
				current_type = tag[2:]
				start = index
		return found

	per_type: dict[str, dict[str, int]] = {}
	correct_tokens = total_tokens = 0

	for tokens, true_tags in zip(sequences_tokens, sequences_tags):
		predicted_tags = model.predict_sequence(tokens)
		for predicted, truth in zip(predicted_tags, true_tags):
			correct_tokens += int(predicted == truth)
			total_tokens += 1

		true_spans = spans(list(true_tags))
		predicted_spans = spans(list(predicted_tags))
		for entity_type in {span[0] for span in true_spans | predicted_spans}:
			bucket = per_type.setdefault(
				entity_type, {"tp": 0, "fp": 0, "fn": 0}
			)
			truth_of_type = {s for s in true_spans if s[0] == entity_type}
			predicted_of_type = {s for s in predicted_spans if s[0] == entity_type}
			bucket["tp"] += len(truth_of_type & predicted_of_type)
			bucket["fp"] += len(predicted_of_type - truth_of_type)
			bucket["fn"] += len(truth_of_type - predicted_of_type)

	def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
		precision = tp / (tp + fp) if (tp + fp) else 0.0
		recall = tp / (tp + fn) if (tp + fn) else 0.0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
		return {
			"precision": round(precision, 4),
			"recall": round(recall, 4),
			"f1": round(f1, 4),
			"support": tp + fn,
		}

	by_type = {name: prf(**counts) for name, counts in sorted(per_type.items())}
	total_tp = sum(counts["tp"] for counts in per_type.values())
	total_fp = sum(counts["fp"] for counts in per_type.values())
	total_fn = sum(counts["fn"] for counts in per_type.values())

	macro_f1 = (
		round(float(np.mean([entry["f1"] for entry in by_type.values()])), 4)
		if by_type else 0.0
	)

	return {
		"token_accuracy": round(correct_tokens / max(total_tokens, 1), 4),
		"tokens_evaluated": total_tokens,
		"entity_level": {
			"by_type": by_type,
			"micro": prf(total_tp, total_fp, total_fn),
			"macro_f1": macro_f1,
		},
		"note": (
			"Entity-level scores require an exact type and span match. Token "
			"accuracy is reported alongside them but is inflated by the dominant "
			"'O' tag and should not be quoted as the headline NER metric."
		),
	}


def train_urgency_tfidf(train_df, val_df, test_df) -> dict[str, Any]:
	"""Candidate A: TF-IDF + multinomial Logistic Regression."""
	vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=50000)
	x_train = vectorizer.fit_transform(train_df["text_clean"])
	x_val = vectorizer.transform(val_df["text_clean"])
	x_test = vectorizer.transform(test_df["text_clean"])

	model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
	model.fit(x_train, train_df["urgency"])

	val_pred = model.predict(x_val)
	test_pred = model.predict(x_test)
	return {
		"name": "tfidf_logreg",
		"model": model,
		"vectorizer": vectorizer,
		"val_macro_f1": float(f1_score(val_df["urgency"], val_pred, average="macro", zero_division=0)),
		"val_accuracy": float(accuracy_score(val_df["urgency"], val_pred)),
		"test_macro_f1": float(f1_score(test_df["urgency"], test_pred, average="macro", zero_division=0)),
		"test_accuracy": float(accuracy_score(test_df["urgency"], test_pred)),
		"test_report": classification_report(
			test_df["urgency"], test_pred, output_dict=True, zero_division=0
		),
	}


def train_urgency_transformer(train_df, val_df, test_df, train_cap: int, epochs: int) -> dict[str, Any] | None:
	"""Candidate B: fine-tuned DistilBERT.

	The previous implementation fine-tuned on 100 rows for 1 epoch, never
	evaluated the result, hardcoded 0.98 as its accuracy, and then never loaded
	the model at inference (the branch that would have used it tested for a
	tokenizer key that _get_artifacts never set). This trains on a real budget,
	measures real validation and test scores, and returns them for selection.
	"""
	if not TRANSFORMERS_AVAILABLE:
		return None

	label_map = {label: index for index, label in enumerate(URGENCY_CLASSES)}
	inverse_label_map = {index: label for label, index in label_map.items()}

	sample = train_df.sample(n=min(len(train_df), train_cap), random_state=SEED)
	val_sample = val_df.sample(n=min(len(val_df), 2000), random_state=SEED)
	test_sample = test_df

	tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
	model = AutoModelForSequenceClassification.from_pretrained(
		"distilbert-base-uncased", num_labels=len(URGENCY_CLASSES)
	)

	train_dataset = TextDataset(
		sample["text_clean"].tolist(),
		[label_map[label] for label in sample["urgency"]],
		tokenizer,
	)

	arguments = TrainingArguments(
		output_dir=str(MODEL_DIR / "transformer_urgency"),
		num_train_epochs=epochs,
		per_device_train_batch_size=16,
		learning_rate=5e-5,
		use_cpu=not (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()),
		save_strategy="no",
		report_to="none",
		logging_steps=100,
		seed=SEED,
	)
	trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset)
	trainer.train()
	model.eval()

	def predict(texts: list[str]) -> list[str]:
		predictions: list[str] = []
		batch_size = 64
		for start in range(0, len(texts), batch_size):
			batch = texts[start:start + batch_size]
			encoded = tokenizer(
				batch, return_tensors="pt", truncation=True, padding=True, max_length=128
			)
			with torch.no_grad():
				logits = model(**encoded).logits
			predictions.extend(
				inverse_label_map[int(index)] for index in torch.argmax(logits, dim=-1)
			)
		return predictions

	val_pred = predict(val_sample["text_clean"].tolist())
	test_pred = predict(test_sample["text_clean"].tolist())

	return {
		"name": "distilbert",
		"model": model,
		"tokenizer": tokenizer,
		"train_samples_used": int(len(sample)),
		"epochs": epochs,
		"val_macro_f1": float(f1_score(val_sample["urgency"], val_pred, average="macro", zero_division=0)),
		"val_accuracy": float(accuracy_score(val_sample["urgency"], val_pred)),
		"test_macro_f1": float(f1_score(test_sample["urgency"], test_pred, average="macro", zero_division=0)),
		"test_accuracy": float(accuracy_score(test_sample["urgency"], test_pred)),
		"test_report": classification_report(
			test_sample["urgency"], test_pred, output_dict=True, zero_division=0
		),
	}


def train_models(
	transformer_train_cap: int = 8000,
	transformer_epochs: int = 2,
	skip_transformer: bool = False,
) -> dict[str, Any]:
	"""Train the urgency, hazard and NER models and write a real metrics manifest.

	Urgency is a benchmark between TF-IDF+LogReg and fine-tuned DistilBERT,
	selected on VALIDATION macro F1. Whichever wins is the artifact that
	inference loads -- recorded in urgency_model_meta.json so the serving path
	cannot silently diverge from the trained model.
	"""
	seed_everything()
	ensure_directories()

	print("\n" + "=" * 70)
	print("STAGE 03 NLP MODEL TRAINING & EVALUATION")
	print("=" * 70)

	df_class = load_classification_datasets()
	print("\n[CLASSIFICATION DATASET]")
	print(f"  Total records: {len(df_class):,}")
	print(f"  Urgency counts:\n{df_class['urgency'].value_counts().to_string()}")
	print(f"  Hazard counts:\n{df_class['hazard'].value_counts().to_string()}")

	# Stratified 70/15/15 train/val/test split
	train_df, rem_df = train_test_split(
		df_class, test_size=0.30, random_state=SEED, stratify=df_class["urgency"]
	)
	val_df, test_df = train_test_split(
		rem_df, test_size=0.50, random_state=SEED, stratify=rem_df["urgency"]
	)
	print(f"\n  Split -> train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

	# ------------------------------------------------------------------
	# URGENCY: benchmark two candidates, select on VALIDATION macro F1
	# ------------------------------------------------------------------
	print("\n[URGENCY CLASSIFIER BENCHMARK]")
	candidates: list[dict[str, Any]] = []

	print("  Training candidate A: TF-IDF + Logistic Regression ...")
	tfidf_candidate = train_urgency_tfidf(train_df, val_df, test_df)
	candidates.append(tfidf_candidate)
	print(
		f"    val macro F1 = {tfidf_candidate['val_macro_f1']:.4f} | "
		f"val acc = {tfidf_candidate['val_accuracy']:.4f}"
	)

	transformer_candidate = None
	if TRANSFORMERS_AVAILABLE and not skip_transformer:
		print(
			f"  Training candidate B: DistilBERT fine-tune "
			f"(cap {transformer_train_cap:,} rows, {transformer_epochs} epochs) ..."
		)
		transformer_candidate = train_urgency_transformer(
			train_df, val_df, test_df, transformer_train_cap, transformer_epochs
		)
		if transformer_candidate is not None:
			candidates.append(transformer_candidate)
			print(
				f"    val macro F1 = {transformer_candidate['val_macro_f1']:.4f} | "
				f"val acc = {transformer_candidate['val_accuracy']:.4f}"
			)
	elif skip_transformer:
		print("  Candidate B skipped (skip_transformer=True).")
	else:
		print("  Candidate B unavailable (transformers not installed).")

	# SELECTION ON VALIDATION ONLY. Test scores are reported, never selected on.
	best = max(candidates, key=lambda entry: entry["val_macro_f1"])
	print(f"\n  SELECTED URGENCY MODEL: {best['name']} (val macro F1 {best['val_macro_f1']:.4f})")

	urgency_meta: dict[str, Any] = {
		"selected_backend": "transformer" if best["name"] == "distilbert" else "tfidf",
		"selected_model": best["name"],
		"selected_on": "validation macro F1",
		"classes": URGENCY_CLASSES,
		"candidates": {
			candidate["name"]: {
				key: value
				for key, value in candidate.items()
				if key not in {"model", "vectorizer", "tokenizer", "test_report"}
			}
			for candidate in candidates
		},
	}

	if best["name"] == "distilbert":
		best["model"].save_pretrained(MODEL_DIR / "transformer_urgency")
		best["tokenizer"].save_pretrained(MODEL_DIR / "transformer_urgency")
	# The TF-IDF pair is always persisted so a fallback backend exists even when
	# the transformer wins, and so the app still runs without torch installed.
	atomic_joblib_dump(tfidf_candidate["model"], MODEL_DIR / "urgency_classifier.joblib")
	atomic_joblib_dump(tfidf_candidate["vectorizer"], MODEL_DIR / "urgency_tfidf.joblib")

	# ------------------------------------------------------------------
	# HAZARD
	# ------------------------------------------------------------------
	print("\n[TRAINING HAZARD CLASSIFIER]")
	haz_vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), max_features=10000)
	x_train_haz = haz_vectorizer.fit_transform(train_df["text_clean"])
	x_val_haz = haz_vectorizer.transform(val_df["text_clean"])
	x_test_haz = haz_vectorizer.transform(test_df["text_clean"])

	haz_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
	haz_model.fit(x_train_haz, train_df["hazard"])

	val_haz_pred = haz_model.predict(x_val_haz)
	test_haz_pred = haz_model.predict(x_test_haz)
	val_haz_acc = float(accuracy_score(val_df["hazard"], val_haz_pred))
	test_haz_acc = float(accuracy_score(test_df["hazard"], test_haz_pred))
	test_haz_macro_f1 = float(
		f1_score(test_df["hazard"], test_haz_pred, average="macro", zero_division=0)
	)

	# Keyword-lookup baseline. The hazard label used to BE this lookup applied to
	# the same text, which is why every class scored exactly 1.0000. Reporting
	# the baseline alongside the model shows what the model actually adds.
	baseline_available = "hazard_baseline" in test_df.columns
	baseline_accuracy = (
		float(accuracy_score(test_df["hazard"], test_df["hazard_baseline"]))
		if baseline_available else None
	)

	print(f"  Hazard Val Accuracy      : {val_haz_acc:.4f}")
	print(f"  Hazard Test Accuracy     : {test_haz_acc:.4f}")
	print(f"  Hazard Test Macro F1     : {test_haz_macro_f1:.4f}")
	if baseline_accuracy is not None:
		print(f"  Keyword baseline accuracy: {baseline_accuracy:.4f}")

	atomic_joblib_dump(haz_model, MODEL_DIR / "hazard_classifier.joblib")
	atomic_joblib_dump(haz_vectorizer, MODEL_DIR / "hazard_tfidf.joblib")

	# ------------------------------------------------------------------
	# NER
	# ------------------------------------------------------------------
	print("\n[TRAINING TOKEN-LEVEL BIO NER MODEL]")
	bio_tokens, bio_tags, bio_stats = load_bio_ner_datasets()
	print(
		f"  BIO Datasets loaded: {bio_stats['used_records']:,} sequences used "
		f"(Malformed/Skipped: {bio_stats['malformed_records']})"
	)

	indices = np.arange(len(bio_tokens))
	train_idx, rem_idx = train_test_split(indices, test_size=0.30, random_state=SEED)
	val_idx, test_idx = train_test_split(rem_idx, test_size=0.50, random_state=SEED)

	train_tokens = [bio_tokens[i] for i in train_idx]
	train_tags = [bio_tags[i] for i in train_idx]
	test_tokens = [bio_tokens[i] for i in test_idx]
	test_tags = [bio_tags[i] for i in test_idx]

	ner_model = TokenLevelNERModel()
	ner_model.fit(train_tokens, train_tags)

	ner_metrics = evaluate_ner_entities(ner_model, test_tokens, test_tags)
	print(f"  NER Token Test Accuracy : {ner_metrics['token_accuracy']:.4f}")
	print(f"  NER Entity Micro F1     : {ner_metrics['entity_level']['micro']['f1']:.4f}")
	print(f"  NER Entity Macro F1     : {ner_metrics['entity_level']['macro_f1']:.4f}")
	for entity_type, scores in ner_metrics["entity_level"]["by_type"].items():
		print(
			f"    {entity_type:<10} P={scores['precision']:.4f} "
			f"R={scores['recall']:.4f} F1={scores['f1']:.4f} (n={scores['support']})"
		)

	atomic_joblib_dump(ner_model.to_components(), MODEL_DIR / "ner_model.joblib")

	urgency_meta["ner_artifact_format"] = "token_level_ner_components_v1"
	atomic_write_text(json.dumps(urgency_meta, indent=2), MODEL_DIR / "urgency_model_meta.json")

	# ------------------------------------------------------------------
	# MANIFEST -- every number below is measured, none are hardcoded
	# ------------------------------------------------------------------
	metrics = {
		"classification": {
			"total_samples": len(df_class),
			"urgency_train_samples": len(train_df),
			"urgency_val_samples": len(val_df),
			"urgency_test_samples": len(test_df),
			"hazard_train_samples": len(train_df),
			"hazard_val_samples": len(val_df),
			"hazard_test_samples": len(test_df),
			"urgency_selected_model": best["name"],
			"urgency_selection_metric": "validation macro F1",
			"urgency_accuracy_val": best["val_accuracy"],
			"urgency_accuracy_test": best["test_accuracy"],
			"urgency_macro_f1_val": best["val_macro_f1"],
			"urgency_macro_f1_test": best["test_macro_f1"],
			"urgency_test_report": best["test_report"],
			"urgency_candidates": urgency_meta["candidates"],
			"hazard_accuracy_val": val_haz_acc,
			"hazard_accuracy_test": test_haz_acc,
			"hazard_macro_f1_test": test_haz_macro_f1,
			"hazard_keyword_baseline_accuracy": baseline_accuracy,
			"urgency_classes": list(URGENCY_CLASSES),
			"hazard_classes": list(haz_model.classes_),
		},
		"ner": {
			"max_dispatcher_ner_records": MAX_DISPATCHER_NER_RECORDS,
			"total_sequences": len(bio_tokens),
			"train_sequences": len(train_tokens),
			"val_sequences": len(val_idx),
			"test_sequences": len(test_tokens),
			"token_accuracy_test": ner_metrics["token_accuracy"],
			"entity_level": ner_metrics["entity_level"],
			"bio_tags": ner_model.classes_,
			"bio_dataset_stats": bio_stats,
			"metric_note": ner_metrics["note"],
		},
		"primary_metric": (
			"Macro F1. The urgency classes are imbalanced (CRITICAL is the "
			"smallest and the most operationally costly to miss), so accuracy "
			"overstates performance and is reported only as a secondary figure."
		),
	}

	atomic_write_text(
		json.dumps(metrics, indent=2), OUTPUT_DIR / "nlp_evaluation_metrics.json"
	)
	print("\n[ARTIFACTS SAVED SUCCESSFULLY]")
	return metrics


def seed_everything(seed: int = SEED) -> None:
	"""Set random seed for repeatability."""
	np.random.seed(seed)


TRAIN_COMMAND_HINT = (
	"Run `python Stage03_NLP/03_nlp_engineer.py` to train Stage 03 before serving. "
	"Training is never triggered automatically from an inference call."
)


def _load_urgency_backend(meta: dict[str, Any], artifacts: dict[str, Any]) -> None:
	"""Load whichever urgency backend training actually selected.

	The transformer branch in analyze_text() used to be dead code: it required
	an "urgency_tokenizer" key that nothing ever set, so a fine-tuned DistilBERT
	could be trained and saved and still never serve a single prediction. The
	backend is now read from urgency_model_meta.json, written by train_models().
	"""
	backend = meta.get("selected_backend", "tfidf")
	transformer_dir = MODEL_DIR / "transformer_urgency"

	if backend == "transformer":
		# The fine-tuned transformer is ~1 GB and is deliberately gitignored, so a
		# fresh clone has the metadata but not the weights. Degrade to the TF-IDF
		# pair (always persisted by train_models) with a loud warning rather than
		# taking the whole stage offline.
		unavailable_reason: str | None = None
		if not TRANSFORMERS_AVAILABLE:
			unavailable_reason = "`transformers`/`torch` are not installed"
		elif not (transformer_dir / "config.json").exists():
			unavailable_reason = f"no model weights found in {transformer_dir}"

		if unavailable_reason is None:
			model = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
			model.eval()
			artifacts["urgency_backend"] = "transformer"
			artifacts["urgency_model"] = model
			artifacts["urgency_tokenizer"] = AutoTokenizer.from_pretrained(transformer_dir)
			return

		fallback_available = (
			(MODEL_DIR / "urgency_classifier.joblib").exists()
			and (MODEL_DIR / "urgency_tfidf.joblib").exists()
		)
		if not fallback_available:
			raise FileNotFoundError(
				f"Transformer urgency backend selected but {unavailable_reason}, and no "
				f"TF-IDF fallback exists in {MODEL_DIR}. " + TRAIN_COMMAND_HINT
			)
		print(
			f"[Stage03] WARNING: transformer urgency backend selected but "
			f"{unavailable_reason}. Falling back to the TF-IDF classifier, which "
			f"scores lower (see urgency_model_meta.json for both candidates). "
			f"Retrain to restore the selected model."
		)
		artifacts["urgency_backend_requested"] = "transformer"
		artifacts["urgency_backend_fallback_reason"] = unavailable_reason

	artifacts["urgency_backend"] = "tfidf"
	artifacts["urgency_model"] = joblib.load(MODEL_DIR / "urgency_classifier.joblib")
	artifacts["urgency_vectorizer"] = joblib.load(MODEL_DIR / "urgency_tfidf.joblib")


def _get_artifacts() -> dict[str, Any]:
	"""Load model artifacts once for inference. NEVER trains.

	This previously called train_models() when an artifact was missing or failed
	to unpickle. Because the Flask adapter calls into here, a single HTTP request
	could kick off a full 63k-record training run plus a DistilBERT fine-tune
	inside the request handler -- and when that run failed part-way it truncated
	the deployed NER model to 2 bytes. Missing artifacts are now a loud, fast
	error telling the operator to run training explicitly.
	"""
	global _LOADED_ARTIFACTS
	if _LOADED_ARTIFACTS:
		return _LOADED_ARTIFACTS

	required = {
		"hazard_classifier.joblib": MODEL_DIR / "hazard_classifier.joblib",
		"hazard_tfidf.joblib": MODEL_DIR / "hazard_tfidf.joblib",
		"ner_model.joblib": MODEL_DIR / "ner_model.joblib",
	}
	missing = [name for name, path in required.items() if not path.exists()]
	if missing:
		raise FileNotFoundError(
			f"Stage 03 model artifacts missing from {MODEL_DIR}: {sorted(missing)}. "
			+ TRAIN_COMMAND_HINT
		)

	meta_path = MODEL_DIR / "urgency_model_meta.json"
	if meta_path.exists():
		meta = json.loads(meta_path.read_text(encoding="utf-8"))
	else:
		# Artifacts predating the metadata file: fall back to TF-IDF if present.
		meta = {"selected_backend": "tfidf"}
	if meta.get("selected_backend") != "transformer":
		for name in ("urgency_classifier.joblib", "urgency_tfidf.joblib"):
			if not (MODEL_DIR / name).exists():
				raise FileNotFoundError(
					f"Stage 03 urgency artifact missing: {MODEL_DIR / name}. "
					+ TRAIN_COMMAND_HINT
				)

	artifacts: dict[str, Any] = {"urgency_meta": meta}
	_load_urgency_backend(meta, artifacts)

	artifacts["hazard_model"] = joblib.load(MODEL_DIR / "hazard_classifier.joblib")
	artifacts["hazard_vectorizer"] = joblib.load(MODEL_DIR / "hazard_tfidf.joblib")

	try:
		ner_payload = joblib.load(MODEL_DIR / "ner_model.joblib")
	except Exception as exc:
		# Artifacts written before the components format embedded a reference to
		# the loading module's synthetic name and cannot be unpickled here.
		raise RuntimeError(
			f"Could not load the NER artifact ({type(exc).__name__}: {exc}). "
			"This usually means it predates the components serialisation format. "
			+ TRAIN_COMMAND_HINT
		) from exc
	artifacts["ner_model"] = TokenLevelNERModel.from_components(ner_payload)

	_LOADED_ARTIFACTS = artifacts
	return _LOADED_ARTIFACTS


def predict_urgency(text: str) -> str | None:
	"""Predict 4-class urgency level (LOW, MEDIUM, HIGH, CRITICAL). Returns None for empty input."""
	res = analyze_text(text)
	return res["urgency_level"]


def predict_hazard(text: str) -> str | None:
	"""Predict multi-class hazard type (Flood, Landslide, Cyclone, etc.). Returns None for empty input."""
	res = analyze_text(text)
	return res["hazard_type"]


def _validate_headcount(val: int, text: str) -> bool:
	"""Validate if a candidate headcount number has required people-related contextual indicators."""
	str_val = str(val)
	# Reject timestamp patterns like 14:30 or 08:00
	if re.search(r"\b" + str_val + r":\d{2}\b", text) or re.search(r"\b\d{1,2}:" + str_val + r"\b", text):
		return False
	# Reject route / highway designations like Highway 44 or Route 66
	if re.search(r"\b(?:highway|route|nh|state highway|national highway|st|no|number|gate|platform|pier)\s+" + str_val + r"\b", text, re.IGNORECASE):
		return False
	# Require explicit people-related contextual indicators
	indicators_pattern = (
		r"\b" + str_val + r"\s*(?:people|person|persons|residents|families|individuals|victims|affected|trapped|injured|rescued|evacuated|in need|stranded|missing|patients|citizens|casualties)\b"
	)
	context_pattern = (
		r"\b(?:affected|trapped|injured|rescued|evacuated|stranded|missing)\s*(?:for|of)?\s*" + str_val + r"\b"
	)
	return bool(re.search(indicators_pattern, text, re.IGNORECASE) or re.search(context_pattern, text, re.IGNORECASE))


def _extract_headcount_fallback(text: str) -> int | None:
	"""Regex & pattern headcount fallback extractor requiring explicit people indicators."""
	pattern1 = re.compile(
		r"\b(\d{1,4})\s*(?:people|person|persons|residents|families|individuals|victims|affected|trapped|injured|rescued|evacuated|in need)\b",
		re.IGNORECASE,
	)
	match1 = pattern1.search(text)
	if match1:
		try:
			val = int(match1.group(1))
			if _validate_headcount(val, text):
				return val
		except ValueError:
			pass

	pattern2 = re.compile(
		r"\b(?:affected|trapped|injured|rescued|evacuated)\s*(?:for|of)?\s*(\d{1,4})\s*(?:people|person|persons|residents|families|individuals|victims)?\b",
		re.IGNORECASE,
	)
	match2 = pattern2.search(text)
	if match2:
		try:
			val = int(match2.group(1))
			if _validate_headcount(val, text):
				return val
		except ValueError:
			pass

	# Search written numbers requiring people-related context
	for word, val in NUMBER_WORD_MAP.items():
		if re.search(r"\b" + word + r"\b\s*(?:people|person|persons|residents|families|individuals|victims|affected|trapped|injured|rescued|evacuated|in need)", text, re.IGNORECASE):
			return val

	return None


def extract_entities(text: str) -> dict[str, Any]:
	"""Extract location, resource_needed, and headcount entities using trained NER + fallback."""
	artifacts = _get_artifacts()
	ner_model: TokenLevelNERModel = artifacts["ner_model"]

	raw_text = clean_text_basic(text)
	tokens = TOKEN_PATTERN.findall(raw_text)

	locations: list[str] = []
	resources: list[str] = []
	headcounts: list[int] = []

	if tokens:
		tags = ner_model.predict_sequence(tokens)

		current_loc, current_res = [], []
		for tok, tag in zip(tokens, tags):
			if tag == "B-LOCATION":
				if current_loc:
					locations.append(" ".join(current_loc))
					current_loc = []
				current_loc.append(tok)
			elif tag == "I-LOCATION":
				if current_loc:
					current_loc.append(tok)
			else:
				if current_loc:
					locations.append(" ".join(current_loc))
					current_loc = []

			if tag == "B-RESOURCE":
				if current_res:
					resources.append(" ".join(current_res))
					current_res = []
				current_res.append(tok)
			elif tag == "I-RESOURCE":
				if current_res:
					current_res.append(tok)
			else:
				if current_res:
					resources.append(" ".join(current_res))
					current_res = []

			if tag == "B-HEADCOUNT":
				if tok.isdigit():
					headcounts.append(int(tok))

		if current_loc:
			locations.append(" ".join(current_loc))
		if current_res:
			resources.append(" ".join(current_res))

	# Deduplicate extracted lists preserving order
	unique_locations = list(dict.fromkeys(locations))
	unique_resources = list(dict.fromkeys(resources))

	# Reconstruct multi-word location phrases with location suffix nouns if supported in raw_text
	loc_suffix_pattern = r"(?:bridge|river|road|street|nagar|junction|colony|park|flyover|expressway|highway|hospital|station|lake|dam|cross|lane|puram|vihar|marg|chowk|area|district|village|city|town)"
	expanded_locations = []
	for loc in unique_locations:
		match = re.search(r"\b(" + re.escape(loc) + r"\s+" + loc_suffix_pattern + r")\b", raw_text, re.IGNORECASE)
		if match:
			expanded_locations.append(match.group(1))
		else:
			expanded_locations.append(loc)
	unique_locations = expanded_locations

	# Filter out generic hazard words mistakenly tagged as locations (e.g., 'flood').
	generic_location_terms = {
		"flood", "flooding", "cyclone", "rain", "rainfall", "storm", "landslide",
		"earthquake", "tsunami", "fire", "wildfire", "drowning", "waterlogging",
		"disaster", "heavy rain", "flash flood", "wind", "damage",
		"rescue", "resuce", "team", "police", "ambulance", "water", "people", "person", "help", "emergency"
	}
	connectors = {"to", "the", "and", "of", "for", "after", "due", "with", "in", "near", "around", "at", "from", "on", "send", "me", "please", "help", "need"}
	unique_locations = [
		loc for loc in unique_locations
		if str(loc).strip().lower() not in generic_location_terms and not str(loc).strip().isdigit()
	]

	# Apply Controlled Fallback Layer if slots are empty or only generic hazard words remain.
	words = re.findall(r"[A-Za-z0-9]+", raw_text)
	for i, word in enumerate(words):
		if word.lower() in {"in", "near", "around", "at", "from"}:
			candidate_words = []
			for w in words[i+1:i+6]:
				w_lower = w.lower()
				if w_lower == "the" and not candidate_words:
					continue
				if w_lower in connectors or w_lower in generic_location_terms:
					break
				candidate_words.append(w)
			
			if candidate_words:
				location = " ".join(candidate_words).title()
				if location.lower() not in generic_location_terms:
					unique_locations = [location] + [loc for loc in unique_locations if str(loc).lower() != location.lower()]
	
	if not unique_locations:
		for i, word in enumerate(words):
			if word.lower() in {"in", "near", "around", "at", "from"}:
				candidate_words = []
				for w in words[i+1:i+5]:
					w_lower = w.lower()
					if w_lower == "the" and not candidate_words:
						continue
					if w_lower in connectors:
						break
					candidate_words.append(w)
				
				if candidate_words:
					location = " ".join(candidate_words).title()
					if location.lower() not in generic_location_terms:
						unique_locations.append(location)
						break

	if not unique_resources:
		res_keywords = [
			"rescue boat", "water pump", "ambulance", "food", "drinking water",
			"NDRF flood rescue team", "tow truck", "sandbags", "shelter",
			"medical team", "life jackets", "rescue team", "fire engine",
		]
		text_lower = raw_text.lower()
		for kw in res_keywords:
			if kw in text_lower:
				unique_resources.append(kw)

	valid_ner_headcounts = [h for h in headcounts if _validate_headcount(h, raw_text)]
	headcount_val: int | None = valid_ner_headcounts[0] if valid_ner_headcounts else _extract_headcount_fallback(raw_text)

	return {
		"location": unique_locations if unique_locations else None,
		"resource_needed": unique_resources if unique_resources else [],
		"headcount": headcount_val,
	}


def analyze_text(text: str) -> dict[str, Any]:
	"""Run full pipeline prediction on unstructured emergency text."""
	artifacts = _get_artifacts()
	haz_model: LogisticRegression = artifacts["hazard_model"]
	haz_vec: TfidfVectorizer = artifacts["hazard_vectorizer"]

	clean_cls = clean_text_for_classification(text)

	# Fix 1: Return None and 0.0 confidence for empty/malformed text
	if not clean_cls:
		return {
			"text": text,
			"urgency_level": None,
			"urgency_confidence": 0.0,
			"hazard_type": None,
			"hazard_confidence": 0.0,
			"urgency_backend": artifacts.get("urgency_backend"),
			"entities": {"location": None, "resource_needed": [], "headcount": None},
		}

	# Urgency Prediction -- backend chosen at training time, recorded in metadata.
	#
	# Both backends receive clean_text_for_classification(text), the exact
	# transform used to build their training features. The transformer branch
	# previously received the RAW text while TF-IDF received the cleaned text,
	# a train/inference skew that would have surfaced the moment the transformer
	# was actually wired up.
	backend = artifacts.get("urgency_backend", "tfidf")
	if backend == "transformer":
		tokenizer = artifacts["urgency_tokenizer"]
		transformer_model = artifacts["urgency_model"]
		inputs = tokenizer(
			clean_cls, return_tensors="pt", truncation=True, padding=True, max_length=128
		)
		with torch.no_grad():
			logits = transformer_model(**inputs).logits
			probs = F.softmax(logits, dim=-1)[0]
		urg_idx = int(torch.argmax(probs))
		urgency_level = URGENCY_CLASSES[urg_idx]
		urgency_conf = float(probs[urg_idx])
	else:
		urg_model: LogisticRegression = artifacts["urgency_model"]
		urg_vec: TfidfVectorizer = artifacts["urgency_vectorizer"]
		urg_vec_feat = urg_vec.transform([clean_cls])
		urg_probs = urg_model.predict_proba(urg_vec_feat)[0]
		urg_idx = int(np.argmax(urg_probs))
		urgency_level = str(urg_model.classes_[urg_idx])
		urgency_conf = float(urg_probs[urg_idx])

	# Hazard Prediction
	haz_vec_feat = haz_vec.transform([clean_cls])
	haz_probs = haz_model.predict_proba(haz_vec_feat)[0]
	haz_idx = int(np.argmax(haz_probs))
	hazard_type = str(haz_model.classes_[haz_idx])
	hazard_conf = float(haz_probs[haz_idx])

	# Entity Extraction
	entities = extract_entities(text)

	return {
		"text": text,
		"urgency_level": urgency_level,
		"urgency_confidence": round(urgency_conf, 4),
		"hazard_type": hazard_type,
		"hazard_confidence": round(hazard_conf, 4),
		"urgency_backend": backend,
		"entities": entities,
	}


def process_batch(texts: list[str]) -> list[dict[str, Any]]:
	"""Process a batch of unstructured emergency texts and log misinterpretations."""
	results = [analyze_text(t) for t in texts]
	_log_misinterpretation_audit(results)
	return results


def _log_misinterpretation_audit(batch_results: list[dict[str, Any]]) -> None:
	"""Flag difficult or ambiguous cases and record in misinterpretation_audit.csv."""
	audit_file = OUTPUT_DIR / "misinterpretation_audit.csv"
	flagged_records = []

	for res in batch_results:
		flag_reasons = []
		if res["urgency_confidence"] < 0.60:
			flag_reasons.append("LOW_URGENCY_CONFIDENCE")
		if res["hazard_confidence"] < 0.60:
			flag_reasons.append("LOW_HAZARD_CONFIDENCE")
		if len(res["entities"]["resource_needed"]) > 2:
			flag_reasons.append("MULTI_RESOURCE_AMBIGUITY")
		if res["urgency_level"] in {"HIGH", "CRITICAL"} and res["entities"]["location"] is None:
			flag_reasons.append("MISSING_CRITICAL_LOCATION")

		if flag_reasons:
			flagged_records.append({
				"text": res["text"],
				"urgency_prediction": res["urgency_level"],
				"urgency_confidence": res["urgency_confidence"],
				"hazard_prediction": res["hazard_type"],
				"hazard_confidence": res["hazard_confidence"],
				"extracted_entities": json.dumps(res["entities"]),
				"reason_flagged": "; ".join(flag_reasons),
				"resolution": "PENDING_REVIEW",
			})

	if flagged_records:
		new_df = pd.DataFrame(flagged_records)
		if audit_file.exists():
			try:
				old_df = pd.read_csv(audit_file)
				combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["text"])
				combined.to_csv(audit_file, index=False)
			except Exception:
				new_df.to_csv(audit_file, index=False)
		else:
			new_df.to_csv(audit_file, index=False)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Train the Stage 03 NLP models")
	parser.add_argument(
		"--transformer-train-cap", type=int, default=8000,
		help="Maximum rows used to fine-tune the DistilBERT urgency candidate",
	)
	parser.add_argument(
		"--transformer-epochs", type=int, default=2,
		help="Fine-tuning epochs for the DistilBERT urgency candidate",
	)
	parser.add_argument(
		"--skip-transformer", action="store_true",
		help="Benchmark only the TF-IDF urgency candidate (much faster on CPU)",
	)
	cli_args = parser.parse_args()

	train_models(
		transformer_train_cap=cli_args.transformer_train_cap,
		transformer_epochs=cli_args.transformer_epochs,
		skip_transformer=cli_args.skip_transformer,
	)
