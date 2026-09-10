"""Stage 04 SLM -- Training Pipeline.

Base model  : Qwen/Qwen2.5-3B-Instruct
Adaptation  : QLoRA (4-bit NF4 quantization + LoRA rank-16)
Task        : Disaster-response grounded generation
Input       : Structured incident log (multi-entry, from Stage 03 dispatcher)
Output      : Three-part structured briefing:
                SITUATION: <priority, hazard, location, scale>
                RISK:      <severity interpretation, population at risk, escalation>
                ACTIONS:   <numbered recommended response steps>

Fallback    : If transformers / bitsandbytes are unavailable, a CPU-safe
              TF-IDF baseline is trained instead. Both paths produce the same
              saved-model interface so the rest of the pipeline is unaffected.

Run:
    python Stage04_SLM/03_slm_engineer.py [--epochs N] [--model-id ID]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"
MODEL_DIR = SCRIPT_DIR / "data" / "models"
OUTPUT_DIR = SCRIPT_DIR / "data" / "outputs"

PAIRS_CSV = PROCESSED_DIR / "SLM_Report_Summary_Pairs.csv"
PAIRS_JSONL = PROCESSED_DIR / "SLM_Report_Summary_Pairs.jsonl"

BASELINE_DIR = MODEL_DIR / "slm_baseline"
QWEN_DIR = MODEL_DIR / "qwen_slm_qlora"

SEED = 42

# ---------------------------------------------------------------------------
# Priority / severity constants (must stay in sync with 01_data_engineer.py)
# ---------------------------------------------------------------------------
PRIORITY_ORDER = ["ROUTINE", "ELEVATED", "URGENT", "IMMEDIATE"]
SEVERITY_TO_PRIORITY = {
    "LOW": "ROUTINE",
    "MEDIUM": "ELEVATED",
    "HIGH": "URGENT",
    "CRITICAL": "IMMEDIATE",
}
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# ---------------------------------------------------------------------------
# Risk interpretation templates  (new in Stage 04 -- not in Stage 03 data)
# ---------------------------------------------------------------------------
RISK_TEMPLATES = {
    "ROUTINE": (
        "Situation is contained; no immediate threat to life. "
        "{headcount} people in the affected area; escalation unlikely without environmental change."
    ),
    "ELEVATED": (
        "Moderate risk to the {headcount}-person population. "
        "Conditions may deteriorate if resources are delayed beyond 60 minutes."
    ),
    "URGENT": (
        "High risk: {headcount} people affected, with active hazards reported. "
        "Failure to deploy within 30 minutes significantly increases casualty probability."
    ),
    "IMMEDIATE": (
        "Critical life-safety threat to {headcount} people. "
        "Every minute without intervention increases mortality risk. Full deployment required NOW."
    ),
}

ACTION_STEPS = {
    # ROUTINE -- low-signal; use SITREP and STANDBY shorthand explicitly.
    "ROUTINE": [
        "Place nearest unit on STANDBY; submit SITREP every 60 min to district EOC.",
        "Monitor via ACCESS route checks; no deployment unless conditions change.",
        "Notify district EOC for awareness; hold CAS count at zero.",
    ],
    # ELEVATED -- medium alert; pre-position + rapid assessment.
    "ELEVATED": [
        "Pre-position {resources} near {location}; confirm ACCESS route is clear.",
        "Conduct rapid damage assessment within 30 min; open ICS channel.",
        "Hold {resources} at STAGING area pending SITREP from field IC.",
        "Submit SITREP every 30 min to district EOC; update CAS count.",
    ],
    # URGENT -- active hazard; explicit TRIAGE, SDRF, ACCESS, SITREP.
    "URGENT": [
        "Deploy {resources} to {location} immediately; IC to take command on arrival.",
        "Establish TRIAGE area; begin CAS assessment and first-aid triage.",
        "Confirm and clear ACCESS route before committing all units.",
        "Activate district EOC; notify SDRF -- ETA required within 10 min.",
        "Submit SITREP every 15 min to command; track CAS at STAGING.",
    ],
    # IMMEDIATE -- life-safety; EVAC, NDRF, NDMA, TRIAGE, SITREP all required.
    "IMMEDIATE": [
        "ALL units deploy to {location} -- full activation; IC assumes command NOW.",
        "Initiate EVAC of {headcount} affected persons via safest ACCESS route.",
        "Request NDRF support immediately; confirm ETA and STAGING point.",
        "Activate state-level EOC; notify NDMA and SDRF via IC channel.",
        "Begin TRIAGE at {location}; log all CAS and submit to medical coordinator.",
        "Submit SITREP every 5 min until situation stabilised; IC signs off each report.",
    ],
}

HAZARD_RISK_NOTES = {
    "Flood": "Flood water can rise rapidly; do not enter without water-rescue equipment.",
    "Rescue Emergency": "Structural instability likely; confirm entry safety before deployment.",
    "Road Blockage": "Access may be compromised; identify alternate routes.",
    "Medical Emergency": "Triage immediately; ensure ambulance access is not obstructed.",
}


# ===========================================================================
# TIER 1 -- CPU-safe TF-IDF baseline
# ===========================================================================

class SLMBaseline:
    """TF-IDF priority classifier + rule-based slot extractor -> structured output."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000,
                                          sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED,
                                      multi_class="multinomial", solver="lbfgs")
        self.trained = False

    # ---- Training ----------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        train = df[df["split"] == "train"].copy()
        print(f"[baseline] training on {len(train):,} train pairs")

        X = self.vectorizer.fit_transform(train["report"].astype(str))
        y = train["priority"].astype(str)
        self.clf.fit(X, y)
        self.trained = True

        # Validation accuracy
        val = df[df["split"] == "validation"]
        if not val.empty:
            X_val = self.vectorizer.transform(val["report"].astype(str))
            y_val = val["priority"].astype(str)
            y_pred = self.clf.predict(X_val)
            acc = (y_pred == y_val.values).mean()
            print(f"[baseline] validation priority accuracy: {acc:.3f}")
            report = classification_report(y_val, y_pred, zero_division=0)
            print(report)

    # ---- Slot extraction ---------------------------------------------------

    @staticmethod
    def _extract_slots(report_text: str) -> dict:
        """Pull structured slots from the incident log header + entries."""
        header_match = re.search(
            r"INCIDENT LOG \| (.+?) / (.+?) / (ZONE-\d+) \| (\d+) entries",
            report_text, re.IGNORECASE
        )
        if header_match:
            state = header_match.group(1).strip()
            district = header_match.group(2).strip()
            zone = header_match.group(3).strip()
            entry_count = int(header_match.group(4))
        else:
            # Fallback inline location extraction
            zone = "unknown"
            loc_match = re.search(r"in\s+([A-Za-z\s]+?),\s+([A-Za-z\s]+?)(?=\.|\n|$)", report_text)
            if loc_match:
                district = loc_match.group(1).strip()
                state = loc_match.group(2).strip()
            else:
                district = "unknown"
                state = "unknown"
            
            # Count entry markers like ERSS IDs or non-empty log lines
            erss_count = len(re.findall(r"ERSS-\d+", report_text))
            if erss_count > 0:
                entry_count = erss_count
            else:
                lines = [l for l in report_text.splitlines() if l.strip()]
                entry_count = max(1, len(lines))

        headcounts = [int(m) for m in re.findall(
            r"\b(\d{1,4})\s+(?:people|persons?|affected|residents?|individuals?|victims?)\b",
            report_text, re.IGNORECASE
        )]
        total_headcount = sum(headcounts) if headcounts else 0

        hazards_found = [h for h in ["Flood", "Rescue Emergency", "Road Blockage", "Medical Emergency"]
                         if h.lower() in report_text.lower()]
        resources_found = re.findall(
            r"\b(ambulance|fire\s+(?:truck|engine|brigade|and\s+rescue\s+unit)|"
            r"rescue\s+boat|helicopter|NDRF|SDRF|police|medical\s+team|"
            r"excavator|water\s+tanker|generator)\b",
            report_text, re.IGNORECASE
        )
        resources_found = list(dict.fromkeys(r.lower() for r in resources_found))

        locations_found = re.findall(
            r"\b(river\s+bank|residential\s+colony|main\s+road|bridge|"
            r"hospital|school|market|railway\s+station|field|village|town)\b",
            report_text, re.IGNORECASE
        )
        locations_found = list(dict.fromkeys(l.lower() for l in locations_found))

        return {
            "state": state,
            "district": district,
            "zone": zone,
            "entry_count": entry_count,
            "total_headcount": total_headcount,
            "hazards": hazards_found,
            "resources": resources_found[:3],
            "locations": locations_found[:2],
        }

    # ---- Inference ---------------------------------------------------------

    def predict_priority(self, report_text: str) -> tuple[str, float]:
        if not self.trained:
            raise RuntimeError("Call .fit() before .predict_priority()")
        X = self.vectorizer.transform([report_text])
        proba = self.clf.predict_proba(X)[0]
        classes = self.clf.classes_
        idx = int(np.argmax(proba))
        return str(classes[idx]), float(proba[idx])

    def generate(self, report_text: str) -> dict:
        """Generate a structured 3-part briefing from a report."""
        priority, confidence = self.predict_priority(report_text)
        slots = self._extract_slots(report_text)

        # Hazard phrase
        hazard_list = slots["hazards"]
        if hazard_list:
            hazard_phrase = ", ".join(h.lower() for h in hazard_list[:2])
        else:
            hazard_phrase = "disaster incident"

        # Resource phrase
        res_list = slots["resources"]
        res_phrase = ", ".join(res_list[:2]) if res_list else "available units"

        # Location phrase
        loc_list = slots["locations"]
        loc_phrase = loc_list[0] if loc_list else "the affected area"

        # Hazard-specific risk note
        hazard_note = ""
        for h in hazard_list:
            if h in HAZARD_RISK_NOTES:
                hazard_note = " " + HAZARD_RISK_NOTES[h]
                break

        # Build location description
        if slots["state"] != "unknown" and slots["district"] != "unknown":
            if slots["zone"] != "unknown":
                loc_str = f"in {slots['zone']} {slots['district']}, {slots['state']}"
            else:
                loc_str = f"in {slots['district']}, {slots['state']}"
        elif slots["district"] != "unknown":
            loc_str = f"in {slots['district']}"
        else:
            loc_str = "in affected sector"

        # Build 3-part output
        reports_label = "report" if slots["entry_count"] == 1 else "reports"
        situation = (
            f"{priority}: {hazard_phrase} {loc_str}. "
            f"{slots['entry_count']} {reports_label}, "
            f"{slots['total_headcount']} affected."
        )

        risk = RISK_TEMPLATES[priority].format(headcount=slots["total_headcount"]) + hazard_note

        action_tmpl = ACTION_STEPS[priority]
        actions = []
        for step in action_tmpl:
            actions.append(step.format(
                resources=res_phrase,
                location=loc_phrase,
                headcount=slots["total_headcount"],
            ))

        return {
            "situation": situation,
            "risk": risk,
            "actions": actions,
            "priority": priority,
            "confidence": round(confidence, 3),
            "slots": slots,
            "model": "slm_baseline",
        }

    # ---- Persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "vectorizer.joblib")
        joblib.dump(self.clf, path / "classifier.joblib")
        config = {
            "model_type": "slm_baseline",
            "description": "TF-IDF priority classifier + rule-based slot extractor",
            "classes": list(self.clf.classes_),
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"[baseline] saved -> {path}")

    @classmethod
    def load(cls, path: Path) -> "SLMBaseline":
        model = cls()
        model.vectorizer = joblib.load(path / "vectorizer.joblib")
        model.clf = joblib.load(path / "classifier.joblib")
        model.trained = True
        return model


# ===========================================================================
# TIER 2 -- Qwen2.5-3B-Instruct + QLoRA
# ===========================================================================

SYSTEM_PROMPT = (
    "You are a disaster-response briefing assistant embedded in an emergency "
    "operations centre. Given a structured incident log, produce a concise "
    "three-part briefing in EXACTLY this format:\n\n"
    "SITUATION: <one sentence: priority keyword, hazard type, location, scale>\n"
    "RISK: <one to two sentences: severity assessment, population at risk, "
    "escalation likelihood>\n"
    "ACTIONS:\n1. <first recommended action>\n2. <second>\n3. <third> ...\n\n"
    "Be factual. Do NOT assert anything not stated in the incident log. "
    "Use only the information provided."
)


def _build_prompt(report_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{report_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_training_text(instruction: str, report: str, summary: str,
                          priority: str, slots: dict | None = None) -> str:
    """Convert a training pair into the full chat-format string for SFT."""
    # Augment the target summary into SITUATION / RISK / ACTIONS format
    # so the model learns the 3-part output structure.
    headcount = slots.get("total_headcount", 0) if slots else 0
    hazard_list = [h.strip() for h in str(slots.get("hazard_types", "")).split(";") if h.strip()] \
        if slots else []
    res_list = [r.strip() for r in str(slots.get("resources_required", "")).split(";") if r.strip()] \
        if slots else []
    loc_list = [l.strip() for l in str(slots.get("locations", "")).split(";") if l.strip()] \
        if slots else []

    situation_sentence = summary  # existing 2-sentence summary as SITUATION seed

    risk = RISK_TEMPLATES.get(priority, RISK_TEMPLATES["ELEVATED"]).format(headcount=headcount)

    hazard_note = ""
    for h in hazard_list:
        if h in HAZARD_RISK_NOTES:
            hazard_note = " " + HAZARD_RISK_NOTES[h]
            break

    res_phrase = ", ".join(res_list[:2]) if res_list else "available units"
    loc_phrase = loc_list[0] if loc_list else "the affected area"

    action_steps = ACTION_STEPS.get(priority, ACTION_STEPS["ELEVATED"])
    actions_text = "\n".join(
        f"{i}. " + step.format(resources=res_phrase, location=loc_phrase, headcount=headcount)
        for i, step in enumerate(action_steps, start=1)
    )

    target = (
        f"SITUATION: {situation_sentence}\n"
        f"RISK: {risk}{hazard_note}\n"
        f"ACTIONS:\n{actions_text}"
    )

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{report}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )


def train_qwen_qlora(df: pd.DataFrame, model_id: str, epochs: int,
                     save_dir: Path) -> dict:
    """Fine-tune Qwen2.5-3B-Instruct with QLoRA. Returns training manifest."""
    try:
        import torch
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                   BitsAndBytesConfig, TrainingArguments)
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset as HFDataset
    except ImportError as exc:
        return {"status": "skipped", "reason": str(exc)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[qwen] device: {device}")

    # ---- Tokenizer ---------------------------------------------------------
    print(f"[qwen] loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Quantization config (4-bit NF4) -----------------------------------
    bnb_config = None
    if device == "cuda":
        try:
            import bitsandbytes  # noqa: F401
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("[qwen] 4-bit NF4 QLoRA quantization enabled")
        except ImportError:
            print("[qwen] bitsandbytes not available -- loading in full precision")

    # ---- Model -------------------------------------------------------------
    print(f"[qwen] loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.config.use_cache = False

    # ---- LoRA config -------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Dataset -----------------------------------------------------------
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)

    def row_to_text(row: pd.Series) -> str:
        slots = {
            "total_headcount": row.get("total_headcount", 0),
            "hazard_types": row.get("hazard_types", ""),
            "resources_required": row.get("resources_required", ""),
            "locations": row.get("locations", ""),
        }
        return _build_training_text(
            instruction=str(row.get("instruction", "")),
            report=str(row["report"]),
            summary=str(row["summary"]),
            priority=str(row["priority"]),
            slots=slots,
        )

    train_texts = [row_to_text(row) for _, row in train_df.iterrows()]
    val_texts = [row_to_text(row) for _, row in val_df.iterrows()]

    train_hf = HFDataset.from_dict({"text": train_texts})
    val_hf = HFDataset.from_dict({"text": val_texts})

    # ---- Training ----------------------------------------------------------
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 2 if device == "cuda" else 1
    grad_accum = 4

    sft_config = SFTConfig(
        output_dir=str(save_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=SEED,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    print(f"[qwen] starting QLoRA training -- {epochs} epoch(s), {len(train_texts)} examples")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ---- Save adapter ------------------------------------------------------
    trainer.model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    manifest = {
        "status": "trained",
        "model_id": model_id,
        "adapter_path": str(save_dir),
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        "epochs": epochs,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "device": device,
        "training_seconds": round(elapsed, 1),
    }
    (save_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[qwen] training complete in {elapsed:.0f}s  -> {save_dir}")
    return manifest


# ===========================================================================
# Qwen inference wrapper
# ===========================================================================

class QwenSLM:
    """Thin inference wrapper around the fine-tuned Qwen adapter."""

    def __init__(self, adapter_path: Path, base_model_id: str) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self._torch = torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path), trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.eval()
        if device_str != "cuda":
            self.model = self.model.to(self.device)

    def generate(self, report_text: str, max_new_tokens: int = 300) -> dict:
        prompt = _build_prompt(report_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        t0 = time.time()
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.time() - t0) * 1000
        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse structured output
        situation = risk = ""
        actions: list[str] = []
        for line in generated.splitlines():
            if line.startswith("SITUATION:"):
                situation = line[len("SITUATION:"):].strip()
            elif line.startswith("RISK:"):
                risk = line[len("RISK:"):].strip()
            elif re.match(r"^\d+\.\s", line):
                actions.append(line.strip())

        return {
            "situation": situation or generated,
            "risk": risk,
            "actions": actions,
            "raw_output": generated,
            "latency_ms": round(latency_ms, 1),
            "model": "qwen2.5-3b-instruct-qlora",
        }


# ===========================================================================
# Main
# ===========================================================================

def load_pairs_df() -> pd.DataFrame:
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(
            f"Pairs CSV not found: {PAIRS_CSV}\n"
            "Run `python Stage04_SLM/01_data_engineer.py` first."
        )
    return pd.read_csv(PAIRS_CSV, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID for the base Qwen model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs (default: 3)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip QLoRA training; train CPU baseline only")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 04 (SLM) TRAINING PIPELINE")
    print(f"  Base model : {args.model_id}")
    print(f"  Adaptation : QLoRA (r=16, alpha=32, 4-bit NF4)")
    print("=" * 68)

    df = load_pairs_df()
    print(f"[load] {len(df):,} pairs  (train {(df['split']=='train').sum():,} / "
          f"val {(df['split']=='validation').sum():,} / "
          f"test {(df['split']=='test').sum():,})")

    # ---- TIER 1 -- Baseline ------------------------------------------------
    print("\n--- TIER 1: TF-IDF Priority Classifier Baseline ---")
    baseline = SLMBaseline()
    baseline.fit(df)
    baseline.save(BASELINE_DIR)

    # Quick sanity-check on a test example
    test_sample = df[df["split"] == "test"].iloc[0]
    result = baseline.generate(str(test_sample["report"]))
    print("\n[baseline] sample output:")
    print(f"  SITUATION: {result['situation']}")
    print(f"  RISK:      {result['risk'][:80]}...")
    print(f"  ACTIONS:   {result['actions'][0]}")

    # ---- TIER 2 -- QLoRA ---------------------------------------------------
    qwen_manifest: dict = {"status": "skipped"}
    if not args.baseline_only:
        print("\n--- TIER 2: Qwen2.5-3B-Instruct QLoRA Fine-Tuning ---")
        qwen_manifest = train_qwen_qlora(df, args.model_id, args.epochs, QWEN_DIR)
        if qwen_manifest.get("status") == "skipped":
            print(f"[qwen] skipped: {qwen_manifest.get('reason', 'unknown')}")
            print("[qwen] install: pip install transformers peft trl bitsandbytes datasets accelerate")
    else:
        print("\n[qwen] --baseline-only flag set; skipping QLoRA training.")

    # ---- Training manifest ------------------------------------------------
    manifest = {
        "stage": "04_SLM",
        "script": "03_slm_engineer.py",
        "base_model": args.model_id,
        "adaptation": "QLoRA (r=16, alpha=32, NF4 4-bit)",
        "task": "disaster-response grounded generation",
        "output_format": ["SITUATION", "RISK", "ACTIONS"],
        "baseline": {
            "model_type": "TF-IDF + LogisticRegression priority classifier",
            "saved_to": str(BASELINE_DIR),
        },
        "qwen_qlora": qwen_manifest,
    }

    manifest_path = OUTPUT_DIR / "SLM_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[write] {manifest_path.name}")

    print("\n" + "=" * 68)
    print("TRAINING COMPLETE")
    print(f"  baseline -> {BASELINE_DIR}")
    if qwen_manifest.get("status") == "trained":
        print(f"  qwen     -> {QWEN_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
