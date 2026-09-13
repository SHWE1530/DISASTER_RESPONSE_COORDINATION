# Stage 05 — Generative AI (GenAI)

**Status: implemented. Stress test run end to end against the real Stage 01–04 models.**

| Script | Role | Description |
| --- | --- | --- |
| `01_data_engineer.py` | Data Engineer | Reference baselines from real history: sensor distributions, dispatcher phrasing, real gauge dynamics, and a mask-labelled imagery bank |
| `02_eda_engineer.py` | EDA / Prompt Engineer | Measures 17 blind spots in the historical data and writes the 20-scenario prompt library plus the wildcard, each prompt mapped to the blind spots it targets |
| `03_genai_engineer.py` | GenAI Engineer | Scenario scaffolding, and a conditional VAE over sensor readings (the fallback generator) |
| `03b_llm_scenario_generator.py` | GenAI Engineer | LLM sequence generation, all eight techniques. Local Qwen2.5-3B (blocked: the checkpoint is corrupt, §4) or `--backend gemini` over the API (§4b) |
| `03c_slm_sequence_generator.py` | GenAI Engineer | **The generator in use: a 3.6M-parameter domain SLM trained here on this project's own corpus.** Emits each zone's sensor row and dispatcher message in one autoregressive pass |
| `04_evaluation_engineer.py` | Evaluation Engineer | Scenario audit (realism score, diversity coverage, overconfidence), distributional realism audit, and a 20-scenario stress test through the live pipeline |
| `05_integration_engineer.py` | Integration Engineer | Stress-test dashboard, run history, and a Flask blueprint (standalone `--serve`) |

---

## 0. Day 5 lecture requirements, and where each one lives

| Lecture requirement | Where it is |
| --- | --- |
| Generative family: **sequence / language**, not statistical or simulation | `03c_slm_sequence_generator.py` (running) and `03b_llm_scenario_generator.py` (Qwen, blocked) |
| Output = **structured data *and* narrative together** | one token sequence per zone carries the 9 sensor fields and the dispatcher message, sampled in a single pass |
| Autoregressive · prompt-engineered · fine-tuned · constrained decoding · temperature | §4b, running, recorded per zone in `provenance.slm.techniques` |
| Few-shot · RAG · chain-of-thought | implemented in `03b` against Qwen; **not claimed** for the SLM — they need a large pretrained model (§4b) |
| **Data Engineer**: multiple raw files, remove duplicates, handle missing values, one clean seed dataset | `01_data_engineer.py` |
| **EDA / Prompt Engineer**: explore blind spots, craft prompts, spot hallucinations | `02_eda_engineer.py` — 17 measured blind spots, 20 prompts + wildcard |
| **GenAI Engineer**: seed conditions → Mild / Moderate / Severe / Wildcard scenarios | `03c` + `03b` + `03`, severities ROUTINE / ELEVATED / URGENT / CRITICAL, plus the wildcard |
| **Evaluation Engineer**: *realism score*, *diversity coverage*, *overconfidence* | `scenario_audit()` in `04_evaluation_engineer.py`; results in section 5 |
| **Integration Engineer**: scenarios into a live stress-test dashboard with a confidence figure | `05_integration_engineer.py`, `/genai` |
| Wildcard capstone scenario invented by the team | section 6 |

---

## 1. Mission

Major disasters are rare, so the historical record barely rehearses the situations that matter most. Stage 05 **synthesises compound, multi-zone disaster scenarios** and runs them through the *existing* pipeline: Stage 01 risk, the Stage 02 CNN and LSTM, and Stage 03 NLP, fused by `fusion/decision_engine.py`, plus a Stage 04 briefing for each zone. The goal is to find where the pipeline breaks *before* a real flood does.

**Scope guarantee.** Stage 05 only *reads* earlier stages' data and calls their public adapters. Nothing in `Stage01_ML/`–`Stage04_SLM/`, `fusion/`, `app.py` or the root docs is modified.

---

## 2. Architecture

```
 Stage 01/02/03 data (read-only)
          │
  01_data_engineer ──► sensor_reference.csv, reference_distributions.json,
          │            text_phrase_bank.json, water_level_reference.json, imagery bank
          ▼
  02_eda_engineer ───► blind_spot_report.json ──► scenario_prompts.json (20 + wildcard)
          ▼
  03b_llm_scenario_generator   ◄── the generator this project uses
     └─ LLMScenarioGenerator  Qwen2.5-3B decodes, per zone, ONE JSON object
            holding the 9 sensor fields AND the dispatcher message, grounded in
            retrieved real rows and real messages. Invalid output is repaired,
            then falls back to ↓
          ▼
  03_genai_engineer
     ├─ SensorCVAE  p(9 sensor fields | risk class), trained on 10,000 real rows
     └─ ScenarioGenerator  prompt spec ─► per zone:
            sensors (CVAE + stress modifiers) · 72 h gauge history (real rise rates)
            dispatcher/citizen text (real Stage 03 phrasing + implicit/negated/noisy variants)
            Stage 04 INCIDENT LOG · imagery · ground-truth expectations
          ▼
  04_evaluation_engineer
     ├─ scenario audit  realism score, diversity coverage, overconfidence
     ├─ realism audit   KS, correlation, C2ST vs a naive baseline, Stage 01/03 cross-checks
     ├─ stress test     Stage 01 + 02 + 03 ─► fusion ─► priority/review/conflicts ; Stage 04 briefing
     └─ forecast probe  flat rivers ─► Stage 02 LSTM
          ▼
  05_integration_engineer ─► stress_test_dashboard.html · /genai · /api/genai/*
```

---

## 3. Blind spots found in the historical data (`02_eda_engineer.py`)

Every blind spot is **measured**. Structural gaps state why the data cannot represent them. Result: **11 absent, 2 rare, 4 covered**, and every rare or absent blind spot is targeted by at least one prompt.

| ID | Blind spot | Status | Evidence |
| --- | --- | --- | --- |
| BS01 | Low-risk (calm) zones | rare | 179 / 10,000 rows (1.79%) |
| BS02 | Rainfall beyond the record | absent (**censored**) | 295 rows (2.95%) sit *exactly* on the 104.02 mm ceiling: the tail was capped during cleaning |
| BS03 | Urban waterlogging (heavy rain, low river) | rare | 171 rows (1.71%) |
| BS04 | Silent rise (river high, few calls) | covered | 202 rows (2.02%) |
| BS05–07 | Receding-but-severe, dry-season severe, night onset | covered | 7.4% / 52.8% / 25.0% |
| BS08 | ≥3 Severe zones at the same moment | absent | every row is one zone in isolation |
| BS09 | Missing or faulty telemetry | absent | 0 missing values |
| BS10 | Infrastructure failure (power/telecom) | absent | no such field |
| BS11 | Cross-modal conflict | absent | no joint sensor/image/text dataset |
| BS12 | Noisy or code-mixed text | absent | 100% of sampled dispatcher texts fit one template |
| BS13 | Negated hazard language | absent | 0 matches |
| BS14 | Water-level change beyond the record | absent (**censored**) | 30 rows pinned at the 1.575 m ceiling |
| BS15 | Mass-casualty headcounts (>150) | absent | max headcount 120 |
| BS16 | Multi-hazard messages | absent | every message carries one hazard label |
| BS17 | Forward-looking labels | absent | labels describe the present only |

---

## 4. The generator in use: a domain SLM (`03c_slm_sequence_generator.py`)

The Qwen checkpoint on this machine is corrupt (§4c), so the sequence family is realised by a small language model **trained here, on this project's own data**. That turned out to suit the corpus: 60,000 dispatcher messages over a **308-word vocabulary**, none longer than 34 words. A 3.6M-parameter model learns it to a validation perplexity of **1.96** in 210 seconds on the laptop GPU.

### Why this is the sequence family, not the statistical one

Each training example is ONE token sequence carrying the conditions and the report together:

```
<bos> SEV_CRITICAL HAZ_Flood ST_Assam DI_Nagaon
      rainfall_mm_b37 river_level_m_b52 river_level_threshold_m_b28 emergency_calls_b44 ...
      <sep> flooding reported near the river bank in nagaon , assam . 58 persons affected ...
<eos>
```

A decoder-only transformer is trained on next-token prediction over that. At generation time it is conditioned on the scenario spec and samples **the nine sensor values and the narrative in a single autoregressive pass** — they come from one joint distribution. That is precisely the property the CVAE-plus-phrase-bank path cannot have: there, sensors and text are drawn independently and only *look* related because both were conditioned on the same severity label.

### Pairing the two datasets, and the bug that mattered

Stage 01's sensor record and Stage 03's message corpus were collected separately, so a message is paired with a real sensor row from the **same district** at a compatible risk class.

The first trained model hallucinated badly — CRITICAL zones with rivers metres *below* their danger threshold — and the scenario audit (§2) caught it at **7.8% overconfident**. The cause was the pairing, not the model: `CRITICAL` and `HIGH` both map to the Severe class, and only **57.6%** of real Severe rows sit above threshold, so uniform sampling taught the model that a CRITICAL zone often has a calm river. Banding candidates by river margin — the same rule `03_genai_engineer.py` already uses to pick a CVAE sample — fixed it:

| | before the fix | after |
| --- | ---: | ---: |
| Validation perplexity | 2.13 | **1.96** |
| Realism score | 0.9686 | **0.9892** |
| Overconfident zones | 4 (7.8%) | **1 (2.0%)** |

This is the audit metric from §2 doing the job it was built for, on a generator written after it.

### Techniques: five claimed, three withheld

| Technique | Status |
| --- | --- |
| Autoregressive | ✅ next-token decoding is the mechanism |
| Prompt engineering | ✅ the scenario spec is the conditioning prefix |
| Fine-tuned domain model | ✅ trained on nothing but this project's corpus |
| Constrained decoding | ✅ logits masked by position — a sensor slot accepts only that field's bins, so a malformed scenario cannot be emitted **even in principle**, which is why this path needs no repair-retry loop |
| Sampling / temperature | ✅ `--temperature` |
| Few-shot · RAG · chain-of-thought | ❌ **not claimed.** In-context learning, retrieval grounding and step-by-step reasoning do not emerge at 3.6M parameters. All three are implemented in `03b` against Qwen and run as soon as a sound checkpoint exists. |

### Results (`data/outputs/evaluation/stress_test_report_slm.md`)

| Metric | SLM suite | CVAE suite |
| --- | ---: | ---: |
| Scenarios passed | 16/20 | 17/20 |
| Zones passed | 44/51 | 47/51 |
| **Critical misses** | **0** | **0** |
| Wildcard zones passed | **4/4** | 3/4 |
| Realism score | **0.989** | 0.984 |
| Overconfident zones | **1 (2.0%)** | 2 (3.9%) |
| Condition-grid coverage | 12/16 | **16/16** |
| Fallback rate | **0%** | n/a |
| Latency per zone | 104 ms | — |

The SLM generates more coherent zones (higher realism, fewer overconfident, and it is the only suite whose wildcard passes completely) but **covers less of the condition space** — 12 of 16 grid cells against the CVAE's 16. That is the expected trade: it samples from a learned conditional distribution and concentrates near the modes, while the CVAE was explicitly pushed into the corners by stress modifiers. For a stress-test suite, breadth matters, so the two are complementary rather than one replacing the other.

---

## 4b. LLM sequence generation with Qwen (`03b_llm_scenario_generator.py`)

### Why this family

Three generative families were on the table.

| Family | What it learns | What it emits | Used here |
| --- | --- | --- | --- |
| Statistical / latent (GAN, VAE, diffusion) | a compressed distribution of real events | numbers only | fallback |
| **Sequence / language (LLM)** | token-by-token patterns in reports and logs | **numbers *and* narrative, together** | **yes** |
| Simulation (agent-based, Monte Carlo, digital twin) | the rules and behaviours of actors | behaviour traces | no |

A disaster scenario is not a row of numbers, and it is not a behaviour trace. It is paired sensor readings and a dispatcher narrative that *agree with each other* — the pipeline downstream reads both, and Stage 03 will happily contradict Stage 01 if the two were generated separately. Only a language model emits both in one sequence, so "93 mm" and "water breaching the embankment near Zone 3" come out of the same forward pass and cannot drift apart.

The model is **Qwen2.5-3B**, already on disk for Stage 04, run 4-bit on CUDA (fp32 on CPU). Nothing is sent off the machine.

### The eight techniques, and where each one lives

Every generated zone records which fired, in `provenance.llm.techniques`.

| # | Technique | Where it is in the code |
| --- | --- | --- |
| 1 | Autoregressive LLM | `QwenGenerator.__call__` — next-token decoding is the mechanism |
| 2 | Prompt engineering | `build_user_prompt` turns the scenario spec into an explicit brief (severity, hazards, stress factors, night/outage context) |
| 3 | Few-shot generation | k real `(sensor row, dispatcher message)` exemplars are pasted into the prompt |
| 4 | Retrieval-augmented | `GroundingIndex` *retrieves* those exemplars from the real record — sensor rows by risk class and nearest conditions, messages from the 60k Stage 03 dispatcher corpus by hazard and severity |
| 5 | Fine-tuned domain LLM | `--adapter` attaches Stage 04's QLoRA disaster adapter |
| 6 | Chain-of-thought | the prompt demands a `REASONING:` block that checks rainfall against river against call volume *before* any JSON; the reasoning is kept in provenance |
| 7 | Constrained decoding | `extract_json` + `validate_payload`: JSON schema, physical limits, and a cross-field rule (a CRITICAL zone's river must be at or above its threshold). A rejection is quoted back to the model as a `CORRECTION` and re-decoded |
| 8 | Sampling / temperature | `--temperature` / `--top-p` control how wild the generated edge cases get |

## 4c. Known blocker: the Qwen checkpoint is corrupt

`Stage04_SLM/data/models/qwen_base` is damaged. All 434 tensors are present and the config is correct, but the model does not work:

```
prompt : "What is 2 + 2? Answer in one word."
output : " Welcome Welcome Welcome Welcome Welcome ..."
loss on plain English: 141.32   (a working model scores 2-4)
```

This is not the quantization and not the prompt - bf16 on CPU, bf16 on CUDA, int8 and nf4 all produce it, with eager and sdpa attention alike, and weight tying is applied correctly.

**What was tried, and what it proved.** Four rows of `model.embed_tokens.weight` carry norms of 26-55 against a median of 1.13 (ids 20165, 20166, 22470, 22471 - two adjacent pairs, the signature of a lost download chunk). Qwen ties embeddings to the output head, so those rows dominate every logit. Replacing them with the mean of the healthy rows - the standard initialization for a token with no learned representation - **dropped the loss from 141 to 19.6 and changed which token it repeated, from `" Welcome"` to `"prepare"`.** Better, and still useless. The corruption reaches past the embedding matrix into the transformer weights, and nothing on disk can reconstruct those. The patch was reverted; the checkpoint is exactly as it was found.

That experiment is why the health gate is **behavioural, not statistical**: after the embedding repair the weight statistics looked clean while the model was still ruined. `QwenGenerator._check_checkpoint()` scores one easy sentence and refuses anything above a loss of 8, so a 55-zone stress suite can never be built out of a broken model and scored as if it were real.

**It also means Stage 04's `--slm qwen` path is generating nonsense, and that its QLoRA adapter was fine-tuned on a corrupt base.** Stage 04 defaults to its baseline model, so its reported results are unaffected.

**The fix is a re-download** - there is no local one:

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir Stage04_SLM/data/models/qwen_base
```

Until then `03b` runs end to end on the fallback path and reports a 100% fallback rate.

### The hosted way out: `--backend gemini`

A re-download is not the only route, and on a machine with no GPU and no room for
6 GB of weights it is not the practical one. `03b` takes `--backend gemini`, which
serves the *same* prompts from the Gemini API instead of a local checkpoint:

```bash
pip install google-genai
$env:GEMINI_API_KEY = "..."        # PowerShell; export GEMINI_API_KEY=... on Linux/macOS
python Stage05_GenAI/03b_llm_scenario_generator.py --backend gemini --limit 2
```

Only the backend changes. The prompts, the retrieved few-shot exemplars, the JSON
schema and physical limits, the repair retries and the CVAE fallback are all
backend-agnostic, because the generator only ever asks for `(system, user) -> text`.
So this unblocks techniques 1-4 and 6-8 without touching the pipeline.

What it does **not** do is technique 5. The QLoRA disaster adapter is a local
artefact and cannot be attached to a hosted model, so `--adapter --backend gemini`
is rejected rather than silently ignored, and `fine_tuned_domain_llm` stays out of
the reported `techniques` list on a Gemini run. Generation also leaves the machine
and costs a network round trip per zone, so latency is the API's, not the GPU's.
The key is read from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and never written into
any artefact -- the manifest records the model name only.

The dashboard reads the same switch from the environment:
`STAGE05_LLM_BACKEND=gemini` (optionally `STAGE05_GEMINI_MODEL`) routes
`05_integration_engineer.py`'s LLM path at the hosted model; the default stays local.

### The fallback, and why it is a reported number

A 3B model does not always return valid JSON. Any zone still invalid after `--max-retries` repairs falls back to the CVAE + phrase-bank path below, and is marked `fallback` in provenance. The suite therefore always completes, and `llm_generation_manifest.json` reports the fallback rate rather than letting a degraded run pass as a clean one.

A zone whose telemetry *and* text were destroyed (the blackout zone in the wildcard) is `skipped`: generating text for it would destroy the scenario's whole point.

---

## 4d. The fallback generator (`03_genai_engineer.py`)

**SensorCVAE** is a conditional variational autoencoder (latent 6, hidden 64, KL warm-up, inverse-√frequency class sampling) that learns the *joint* distribution of the nine Stage 01 sensor fields given a risk class. It runs offline on CPU in about 30 s, which is why it backs the LLM and why it still drives the fast interactive path in the dashboard. It also supplies all the scenario *scaffolding* the LLM path reuses: district selection, evidence loss under blackout, gauge histories, imagery, and ground-truth expectations.

- Samples include the model's measured **observation noise**. A first version decoded the mean only. That put all nine fields on a 6-D surface, inflated the rainfall-to-calls correlation to 0.996 (real value 0.824), and made samples trivially detectable (C2ST AUC 0.966). Adding observation noise fixed it (section 5).
- Plain samples are clipped to the historical record. Only named **stress modifiers** (`flash_flood`, `dam_release`, `mass_casualty`, …) may push past it, and each push is logged per zone in `provenance.sensor_adjustments` / `beyond_record_fields`. In the suite, 10 of 51 zones extrapolate, and each one is disclosed.
- Gauge histories use **real CWC rise rates**: 101 stations and 653k hourly readings, after removing 155k sentinel values. Ordinary histories stay inside the LSTM's trained range. Only the deliberate `ood_spike` trips its out-of-distribution guard.
- Text recombines real Stage 03 phrasing and adds what the corpus lacks: implicit urgency (no "urgent" or "critical" words), negation, prank messages, outage messages, and comms noise (typos, caps, Hinglish, truncated SMS).
- **Ground truth is the scenario designer's intent**, encoded per zone as pass criteria *before* the run. Expectations were never relaxed after seeing results.

---

## 5. Results (seed 42; `data/outputs/evaluation/stress_test_report.md`)

### Scenario audit: realism score, diversity coverage, overconfidence

The realism audit below is *distributional* — does a bulk sample of generated rows look like the real record? That is the right question for the generator and the wrong one for a single scenario, because a stress scenario is *meant* to sit off the historical distribution. The scenario audit asks the other question, per zone: is this particular scenario physically coherent, and does the severity it claims match the evidence it carries?

| Metric | CVAE suite (20 + wildcard, 51 scored zones) |
| --- | ---: |
| **Realism score** (mean; 1.0 = no plausibility rule violated) | **0.984** |
| Fully plausible zones | 48 / 51 (94.1%) |
| Lowest-scoring zone | 0.65 |
| **Overconfident zones** (severe label, no physical driver) | **2 / 51 (3.9%)** |
| Zones exempt by design (declared decoupling modifier or conflict archetype) | 3 |
| Blind-spot coverage (rare or absent) | 13 / 13 (100%) |
| Distinct archetypes / hazards / severities / modifiers | 21 / 4 / 4 / 21 |
| Rainfall × river grid cells occupied | 16 / 16 (100%) |

The two overconfident zones are real generator drift, not design: **S03-Z3** (URGENT, 90.4 mm rain just under the 93 mm extreme band, river 5.57 m under its 6.30 m threshold) and **S20-Z1** (URGENT, slow onset, river short of threshold). In both, the CVAE drew a sensor row that does not support the severity the designer asked for. This is exactly the failure the lecture names — *light rain and a calm grid, but labelled Severe* — and it is now measured rather than assumed absent.

**What counts as exempt, and why it is counted.** A severe zone needs at least one physical driver: the river at or above its threshold, or rainfall in the record's top 5%. Three things legitimately break that link, and each is declared on the zone: a decoupling modifier (`river_overflow`, `dam_release`, `silent_rise`, `receding`, `gauge_malfunction`, …), where the water came from upstream rather than the local sky; and the `*_conflict` archetypes, whose entire purpose is that one modality contradicts another. Exempt zones are reported as a separate count, so the exemption is visible rather than a quiet hole in the metric.

### Realism audit (distributional)

| Check | CVAE | Naive baseline (independent marginals) |
| --- | ---: | ---: |
| C2ST ROC-AUC (0.5 = indistinguishable from real) | **0.763** | 0.831 |
| Rainfall-to-calls correlation (real 0.824) | **0.797** | 0.088 |
| Stage 01 recovers the conditioning class (macro F1) | 0.784 | 0.756 |
| Mean KS statistic / max \|Δr\| | 0.053 / 0.072 | — |

| Stage 03 reading generated text | Clean | Degraded (comms noise) |
| --- | ---: | ---: |
| Hazard accuracy | 1.000 | 0.988 |
| Urgency macro F1 | 0.870 | 0.870 |
| Headcount exact | 1.000 | 0.806 |

Implicit-urgency messages scored HIGH or CRITICAL by Stage 03: 100%.

### Stress test: 20 scenarios, 51 zones

| Metric | Result |
| --- | ---: |
| Scenarios passed | **17 / 20** |
| Zones passed | 47 / 51 |
| **Critical misses** (true URGENT+ called ≤ ELEVATED) | **0 / 32** |
| Pipeline crashes | 0 |
| Priority exact / within one level | 70.6% / 90.2% |
| Over-triage (calm zones called 2+ levels high) | 26.3% |
| Human-review recall | 100% |
| Modality conflicts surfaced | 66.7% |
| Mean zone-ranking Kendall τ (17 multi-level scenarios) | 0.871 |
| Fusion latency p50 / p95 | 56 / 67 ms |

### What the stress test found

1. **Stage 02's LSTM projects large rises from flat, low rivers, and fusion escalates calm zones to URGENT.** This causes all 4 failing suite zones (S05, S09-Z2, S18-Z2, S18-Z3). In each one, Stage 01 says Low at 0.99 confidence and the text and imagery are calm. A direct probe with a *perfectly flat* 72 h history gives:

   | Flat level | 1 m | 2 m | 3 m | 4 m | 5 m | 6–9 m |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | Projected 6 h change | **+2.74** | **+1.03** | −0.11 | **+1.57** | −1.43 | −0.23 to −0.49 |

   Real gauges rise more than 1.55 m in 6 h in only 1% of windows. Fusion escalates on a projected rise of ≥0.9 m. The pattern looks like reversion toward the training mean (3.12 m), and it's non-monotonic. **Suggested follow-up (Stage 02 / fusion owners):** require a real recent rise before a forecast may escalate, or cap projected rises at the observed gauge tail.
2. **Stage 03 has no negation handling** (a documented limitation, now reproduced). "No flooding here, please ignore the forwarded message" reads as MEDIUM urgency.
3. **The overall over-triage rate is 26.3%.** The system errs toward over-response, which matches the fusion design ("under-responding costs more"), but it's the price of finding 1.
4. **The Stage 04 baseline briefing is longer than short logs.** Read-time savings average −9.1% on 1–4-entry logs, compared with +45.9% on Stage 04's 139–378-word logs. The >80% target only applies to long logs. The baseline also over-calls priority (40.8% exact) while keeping 100% high-risk recall.
5. **Single-seed sensitivity.** S20 (slow onset) flipped from FAIL to PASS when the CVAE was retrained, so individual scenario outcomes depend on the random draw. Treat the suite rates as indicative, and re-run with other `--seed` values before drawing conclusions from one scenario.

---

## 6. The wildcard: flash flood + municipal blackout (capstone)

**Scenario (invented by the team):** at 2 AM a flash flood hits an Assam district *while the municipal grid fails*. Cameras need mains power and go dark. Gauges survive only on batteries. Cell towers fail, so fewer, shorter, garbled messages get through. One neighbourhood goes completely silent. Resources: one ambulance and one 40-bed shelter, against **323 people reported** across the zones.

| Zone | Truth | Evidence left | Decision | Result |
| --- | --- | --- | --- | --- |
| Riverside settlement (battery gauge) | CRITICAL | sensors, text, gauge history | CRITICAL | PASS |
| Low-lying colony (grid down) | CRITICAL | one garbled text | **ROUTINE** (human review forced) | **FAIL** |
| District hospital (generator) | URGENT | everything | URGENT | PASS |
| Char island (no signal) | CRITICAL | nothing | `insufficient_evidence` (not ROUTINE) | PASS |

**Defence: why it matters.**
- *Silence is not safety:* the zone with zero evidence is refused and escalated to a human. It is never scored as calm. This is the fusion layer's most important safety property, and the wildcard proves it works under total blackout.
- *The failure is structural, not random.* When the only evidence is one low-confidence report, fusion multiplies the **severity level** by confidence (`level × max(conf, 0.25)`), so a lone MEDIUM report at 0.37 confidence rounds to ROUTINE. Human review is still forced (single source, low confidence), which is the safety net that caught it. A sparse, garbled signal from a blacked-out zone is *more* alarming than its words suggest, so outage context should raise priority rather than lower it. That's a concrete requirement for Stage 06.
- *It hands Stage 06 its dilemma:* two CRITICAL zones and one ambulance, with a 323-person demand against 40 beds.

---

## 7. How to run

```bash
python Stage05_GenAI/01_data_engineer.py          # --skip-imagery to reuse the committed imagery bank
python Stage05_GenAI/02_eda_engineer.py
python Stage05_GenAI/03_genai_engineer.py         # trains the CVAE (~30 s CPU); --skip-train to reuse it
python Stage05_GenAI/03c_slm_sequence_generator.py --train # train the domain SLM (~3.5 min GPU) + generate
python Stage05_GenAI/03c_slm_sequence_generator.py         # regenerate from the trained SLM
python Stage05_GenAI/03b_llm_scenario_generator.py         # LLM suite: Qwen2.5-3B, needs a sound checkpoint
python Stage05_GenAI/03b_llm_scenario_generator.py --backend gemini             # hosted LLM: no local weights, needs GEMINI_API_KEY
python Stage05_GenAI/03b_llm_scenario_generator.py --adapter --temperature 1.1   # fine-tuned, wilder
python Stage05_GenAI/03b_llm_scenario_generator.py --dry-run --limit 2          # no GPU: fallback path
python Stage05_GenAI/04_evaluation_engineer.py    # --slm qwen | none, --skip-realism, --seed N
python Stage05_GenAI/04_evaluation_engineer.py --label slm --suite data/outputs/scenarios/slm_scenario_suite.json --wildcard data/outputs/scenarios/slm_wildcard_scenario.json   # score the SLM suite
python Stage05_GenAI/05_integration_engineer.py   # self-test + static dashboard
python Stage05_GenAI/05_integration_engineer.py --serve   # live dashboard: http://127.0.0.1:5005/genai
pytest Stage05_GenAI/test/ -v                     # 71 tests, ~30 s
```

`04_evaluation_engineer.py` uses the Stage 04 **baseline** by default, because the Qwen adapter takes about 10 s per briefing. Each run appends to `stress_test_history.csv`, and the dashboard plots the pass-rate and critical-miss trend across runs.

**Dependencies:** no new pins. The scripts use packages already in `requirements.txt` (torch, pandas, scikit-learn, scipy, xgboost, flask). The one exception is **rebuilding** the imagery bank, which needs `datasets` (installed here, not pinned). The bank itself is committed, so `--skip-imagery` avoids it. The C2ST uses XGBoost rather than scikit-learn's forests, because on this machine an application-control policy blocks a scikit-learn extension that `sklearn.ensemble` imports.

### Wiring into the main dashboard (optional, not applied)

Stage 05 deliberately leaves `app.py` untouched. To mount it there:

```python
stage05 = load_module("stage05_api", BASE_DIR / "Stage05_GenAI" / "05_integration_engineer.py")
app.register_blueprint(stage05.create_blueprint())   # serves /genai and /api/genai/*
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/genai` | GET | Live dashboard (pick a prompt and seed, then generate and stress-test) |
| `/api/genai/health` | GET | Generator readiness via a real probe scenario |
| `/api/genai/prompts` | GET | Prompt library |
| `/api/genai/scenario` | POST | `{prompt_id, seed, generator}` or `{spec, seed, generator}` returns a scenario. `generator` is `"slm"` (the domain SLM), `"llm"` (Qwen2.5-3B) or `"cvae"` (default) |
| `/api/genai/stress-test` | POST | Same input; returns scored zones (loads Stage 01–04 on first call) |
| `/api/genai/report` | GET | Latest full report |

---

## 8. Artifacts

| Path | Content |
| --- | --- |
| `data/processed/` | Reference baselines, phrase bank, gauge reference, imagery manifest, `scenario_prompts.json` |
| `data/raw/imagery/` | 24 scenes (12 flooded, 12 unflooded) labelled from water masks |
| `data/models/sensor_cvae.pt` | CVAE bundle (weights-only loadable) |
| `data/outputs/eda/` | Blind-spot report, coverage grid, prompt × blind-spot matrix |
| `data/outputs/scenarios/` | `scenario_suite.json` (20), `wildcard_scenario.json`, `scenario_zones.csv`, and the `llm_*` equivalents from the LLM generator |
| `data/models/scenario_slm.pt` | The trained domain SLM (3.6M params) with its vocabulary and bin edges |
| `data/outputs/slm_generation_manifest.json` | SLM training curve, corpus stats, techniques claimed and withheld, latency |
| `data/outputs/llm_generation_manifest.json` | LLM model, techniques, temperature, fallback rate, per-zone latency |
| `data/outputs/evaluation/scenario_audit.json` | Realism score, diversity coverage, overconfidence, per scenario and per zone |
| `data/outputs/evaluation/` | Stress-test report (MD/JSON), per-zone and per-scenario CSVs, realism report, run history |
| `data/outputs/stress_test_dashboard.html` | Self-contained dashboard (no external scripts) |

## 9. Limitations

1. **The SLM is small, and three techniques are out of its reach.** Few-shot, RAG and chain-of-thought need a large pretrained model; they are implemented in `03b` but unexecuted until the Qwen checkpoint is replaced (§4c).
2. **The SLM covers less of the condition space than the CVAE** (12 of 16 grid cells against 16). It concentrates near the modes it learned, so the CVAE suite remains the broader stress test.
3. **The SLM's text is bounded by its training corpus.** A 308-word vocabulary produces fluent dispatcher register and nothing outside it — it cannot invent vocabulary for a hazard the corpus never described.
4. **Ground truth is designer intent**, not observed outcomes. Pass rates measure agreement with documented expectations.
5. **The imagery is likely the Stage 02 CNN's own training pool.** It's the only labelled imagery available, so visual evidence in the stress test is optimistic.
6. **The CVAE is still distinguishable from real data** (C2ST 0.763): it's better than the naive baseline but not indistinguishable. Observation noise pushes 3.6–11.7% of raw values past the record edges, and those are clipped.
7. **Text realism is bounded by Stage 03's own templated corpus.** Clean generated text scores near-perfect partly because it shares that template.
8. **Scenario results come from a single seed** (see finding 5).
