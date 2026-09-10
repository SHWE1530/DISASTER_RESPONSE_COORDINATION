# Stage 04 -- Small Language Model (SLM)

**Status: fully implemented.**

| Script | Status |
| --- | --- |
| `01_data_engineer.py` | **Done** -- builds the fine-tuning dataset (3,218 pairs) |
| `02_eda_engineer.py` | **Done** -- EDA: distributions, vocabulary, domain audit |
| `03_slm_engineer.py` | **Done** -- Qwen2.5-3B-Instruct QLoRA + TF-IDF baseline |
| `04_evaluation_engineer.py` | **Done** -- ROUGE, priority accuracy, hallucination, safety, robustness |
| `05_integration_engineer.py` | **Done** -- Flask-compatible integration adapter |

---

## 1. Mission

An incident commander in the field cannot read a multi-entry incident log. They
need a short spoken briefing. This stage fine-tunes a compact model to condense
an incident log into a structured three-part briefing:

- **SITUATION** -- priority keyword, hazard type, location, scale
- **RISK** -- severity assessment, population at risk, escalation likelihood
- **ACTIONS** -- numbered recommended response steps

## 2. Model Specification

| Property | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Adaptation | QLoRA (4-bit NF4 + LoRA r=16, alpha=32) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training | SFTTrainer, 3 epochs, LR=3e-4, cosine schedule |
| CPU fallback | TF-IDF priority classifier + rule-based slot extractor |

## 3. Run Order

```bash
# 1. Build dataset (already done -- outputs exist)
python Stage04_SLM/01_data_engineer.py

# 2. EDA
python Stage04_SLM/02_eda_engineer.py

# 3a. Train baseline only (CPU, fast)
python Stage04_SLM/03_slm_engineer.py --baseline-only

# 3b. Full QLoRA training (requires GPU + transformers + peft + trl + bitsandbytes)
python Stage04_SLM/03_slm_engineer.py --model-id Qwen/Qwen2.5-3B-Instruct --epochs 3

# 4. Evaluate
python Stage04_SLM/04_evaluation_engineer.py --model baseline   # or --model qwen

# 5. Integration self-test
python Stage04_SLM/05_integration_engineer.py
```

## 4. Dataset

| Property | Value |
| --- | --- |
| Pairs | 3,218 (train 2,256 / val 481 / test 481) |
| Report length | 180.9 words mean (139-378) |
| Summary length | 30.2 words mean (22-38) |
| Compression | 82.8% mean (75.9-92.2%) |
| Coverage | 10 states, 50 districts, 5 zones |

## 5. Outputs

### EDA (`data/outputs/eda/`)

| File | Contents |
| --- | --- |
| `priority_distribution.png` | Bar chart of briefing priorities |
| `compression_histogram.png` | Compression % by priority |
| `report_length_distribution.png` | Words-per-report histogram |
| `summary_length_distribution.png` | Words-per-summary histogram |
| `split_priority_matrix.png` | Split x priority heatmap |
| `per_state_coverage.png` | Pairs per state |
| `top_report_unigrams.csv` | Top-30 non-stop words in reports |
| `top_summary_unigrams.csv` | Top-30 non-stop words in summaries |
| `domain_dictionary_audit.csv` | Term coverage (72.6% / 45 of 62 terms found) |

### Models (`data/models/`)

| Path | Contents |
| --- | --- |
| `slm_baseline/` | TF-IDF vectorizer + LogisticRegression classifier (joblib) |
| `qwen_slm_qlora/` | Qwen2.5-3B-Instruct LoRA adapter (after QLoRA training) |

### Evaluation (`data/outputs/evaluation/`)

| File | Contents |
| --- | --- |
| `evaluation_report.md` | Full narrative report |
| `classification_report.json` | Priority precision/recall/F1 |
| `priority_confusion_matrix.csv` | 4x4 confusion matrix |
| `factual_consistency_report.csv` | Per-pair slot fidelity |
| `hallucination_report.csv` | Per-pair hallucination flags |
| `safety_report.csv` | Per-pair safety verdicts |
| `robustness_report.csv` | 8 controlled edge-case results |
| `latency_report.json` | p50/p95/p99 latency |

## 6. Evaluation Metrics (Baseline)

| Metric | Value | Notes |
| --- | --- | --- |
| ROUGE-1 | 0.153 | vs templated ground truth |
| Priority accuracy | 60.1% | 3-class task; ROUTINE is under-represented |
| Slot fidelity | 24.5% | Baseline does not always name zone/district in output |
| Hallucination rate | 0.456 | Baseline extracts from rules; Qwen expected lower |
| Safety fails | 0/481 | No inaction on CRITICAL/HIGH |
| Robustness | 3/8 | Baseline struggles with hypothetical/low-signal logs |
| Latency p50 | <1 ms | Baseline is CPU-instant; Qwen ~200-2000 ms depending on GPU |

> **Low ROUGE and slot fidelity on the baseline are expected.** The baseline
> classifier predicts priority but its generated output uses different phrasing
> than the templated ground truth. The Qwen QLoRA model, once trained, will
> learn to reproduce the exact output format including zone/district/headcount.

## 7. Integration (Flask app)

```python
from Stage04_SLM.integration_engineer import slm_integration_engine

# Health check
health = slm_integration_engine.health_check()
# -> {"status": "healthy", "model_name": "slm_baseline", "inference_ok": True, ...}

# Single summarize
result = slm_integration_engine.summarize(report_text)
# -> {
#      "situation": "IMMEDIATE: flooding in ZONE-3 Pune, Maharashtra ...",
#      "risk": "Critical life-safety threat to 80 people ...",
#      "actions": ["1. ALL units deploy ...", "2. Begin evacuation ...", ...],
#      "priority": "IMMEDIATE",
#      "latency_ms": 0.4,
#      "model": "slm_baseline"
#    }
```

## 8. QLoRA Training Prerequisites

```bash
pip install transformers peft trl bitsandbytes datasets accelerate
```

A CUDA-capable GPU is strongly recommended (VRAM >= 8 GB for 4-bit Qwen 3B).
CPU training is supported but will take several hours per epoch.
