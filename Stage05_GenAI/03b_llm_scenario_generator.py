"""Stage 05 GenAI -- LLM-based sequence generation (the approach this project uses).

Day 5 fixes the generative family for this project: SEQUENCE / LANGUAGE
generation driven by an LLM. The reason is the shape of the artefact we need.
A disaster scenario is not a row of numbers and it is not a behaviour trace --
it is paired SENSOR READINGS and a realistic DISPATCHER NARRATIVE that agree
with each other. A statistical latent model (GAN/VAE) samples numbers only; a
simulation samples behaviour only. Only a language model emits both in one
sequence, so that the 93 mm of rain and the words "water breaching the
embankment" come out of the same forward pass and cannot drift apart.

This module is that generator. It runs the local Qwen2.5-3B already downloaded
for Stage 04 (4-bit on CUDA, fp32 on CPU) and produces, per zone, one JSON
object holding the nine Stage 01 sensor fields AND the message text.

Eight sequence-generation techniques are exercised, and each one is recorded in
`provenance.llm.techniques` so a reader can audit which actually fired:

  1 autoregressive LLM      next-token decoding is the generation mechanism
  2 prompt engineering      a system + task prompt built from the scenario spec
  3 few-shot generation     k real (sensor row -> dispatcher text) exemplars
  4 retrieval-augmented     those exemplars are RETRIEVED from the real record
                            by risk class and hazard, so output stays grounded
  5 fine-tuned domain LLM   --adapter loads Stage 04's QLoRA disaster adapter
  6 chain-of-thought        the model reasons under REASONING: before the JSON
  7 constrained decoding    JSON schema + physical limits, with repair retries
  8 sampling / temperature  --temperature / --top-p control scenario diversity

Everything the LLM cannot produce validly after --max-retries attempts falls
back to the CVAE + phrase-bank path in 03_genai_engineer.py, and that zone is
marked `fallback` in provenance. The suite therefore always completes, and the
fallback rate is a reported number rather than a silent degradation.

Run:
    python Stage05_GenAI/03b_llm_scenario_generator.py                 # suite + wildcard
    python Stage05_GenAI/03b_llm_scenario_generator.py --adapter       # fine-tuned Qwen
    python Stage05_GenAI/03b_llm_scenario_generator.py --dry-run       # no model, fallback only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
SCENARIO_DIR = OUTPUT_DIR / "scenarios"

QWEN_BASE_DIR = REPO_ROOT / "Stage04_SLM" / "data" / "models" / "qwen_base"
QWEN_ADAPTER_DIR = REPO_ROOT / "Stage04_SLM" / "data" / "models" / "qwen_slm_qlora"
DISPATCHER_CORPUS = (REPO_ROOT / "Stage03_NLP" / "data" / "processed"
                     / "Dispatcher_Log_Master_60000_Processed.csv")

LLM_SUITE_JSON = SCENARIO_DIR / "llm_scenario_suite.json"
LLM_WILDCARD_JSON = SCENARIO_DIR / "llm_wildcard_scenario.json"
LLM_ZONES_CSV = SCENARIO_DIR / "llm_scenario_zones.csv"
LLM_MANIFEST_JSON = OUTPUT_DIR / "llm_generation_manifest.json"

SEED = 42
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.92
DEFAULT_MAX_RETRIES = 2
DEFAULT_SHOTS = 3
MAX_NEW_TOKENS = 260
# Health checks for the checkpoint. Both are needed, and the second is the one
# that actually matters.
#
# The norm check is a cheap smell test: Qwen ties its input embeddings to its
# output head, so one blown-up embedding row dominates every logit and the model
# emits that token forever. 10x the median row norm sits well above healthy
# spread (the largest legitimate row is near 2x).
#
# But a checkpoint can pass that and still be ruined -- repairing the embedding
# rows of the damaged copy in Stage04_SLM dropped its loss from 145 to 19.6 and
# changed which token it repeated, because the corruption also reached the
# transformer weights. So the real gate is behavioural: score a sentence any
# language model finds trivial. A healthy Qwen2.5-3B scores about 2-4 here;
# anything past 8 is not a working model, whatever its weight statistics say.
EMBEDDING_OUTLIER_RATIO = 10.0
HEALTH_PROBE_TEXT = "The capital of France is Paris. The capital of Germany is Berlin."
HEALTH_PROBE_MAX_LOSS = 8.0
RETRIEVAL_POOL = 4000   # rows of the dispatcher corpus held for retrieval

# Hosted backend. Flash is the right default here: the task is short, highly
# constrained JSON and it is called once per zone, so throughput and cost matter
# more than headroom.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_ATTEMPTS = 3
GEMINI_BACKOFF_S = 2.0


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENAI = _load_module("stage05_genai_engineer", BASE_DIR / "03_genai_engineer.py")

TECHNIQUES = [
    "autoregressive_llm",
    "prompt_engineering",
    "few_shot",
    "retrieval_augmented",
    "chain_of_thought",
    "constrained_decoding",
    "sampling_temperature",
]


# ===========================================================================
# 1. Retrieval -- grounding exemplars in the real record (technique 4)
# ===========================================================================

class GroundingIndex:
    """Retrieves real sensor rows and real dispatcher messages as exemplars.

    Retrieval is the cheap, honest kind: filter the real record by the risk
    class / hazard / severity being asked for, then rank by distance to the
    target conditions. No embedding model is involved, because the query is
    fully structured -- an embedding would add a dependency and lose the exact
    class match that matters here.
    """

    RANK_FIELDS = ["rainfall_mm", "river_level_m", "emergency_calls"]

    def __init__(self, processed_dir: Path = PROCESSED_DIR,
                 corpus_path: Path = DISPATCHER_CORPUS) -> None:
        self.sensors = pd.read_csv(Path(processed_dir) / "sensor_reference.csv")
        self.scale = {f: max(float(self.sensors[f].std()), 1e-6) for f in self.RANK_FIELDS}
        self.texts: pd.DataFrame | None = None
        if Path(corpus_path).is_file():
            cols = ["text", "severity", "hazard_type"]
            frame = pd.read_csv(corpus_path, usecols=cols, nrows=RETRIEVAL_POOL)
            self.texts = frame.dropna(subset=cols)

    def sensor_exemplars(self, risk_class: str, target: dict[str, float], k: int) -> list[dict]:
        pool = self.sensors[self.sensors["zone_risk"] == risk_class]
        if pool.empty:
            pool = self.sensors
        distance = np.zeros(len(pool), dtype=float)
        for field in self.RANK_FIELDS:
            if field in target:
                distance += ((pool[field].to_numpy() - float(target[field])) / self.scale[field]) ** 2
        order = np.argsort(distance)[:max(k, 0)]
        picked = pool.iloc[order]
        return [{f: (int(row[f]) if f in GENAI.INTEGER_FEATURES else round(float(row[f]), 2))
                 for f in GENAI.NUMERIC_FEATURES} for _, row in picked.iterrows()]

    def text_exemplars(self, hazard: str, text_severity: str, k: int) -> list[str]:
        if self.texts is None or k <= 0:
            return []
        pool = self.texts[(self.texts["hazard_type"] == hazard)
                          & (self.texts["severity"] == text_severity)]
        if pool.empty:
            pool = self.texts[self.texts["hazard_type"] == hazard]
        if pool.empty:
            return []
        return [str(t) for t in pool["text"].head(k).tolist()]


# ===========================================================================
# 2. The LLM backend (techniques 1, 5, 8)
# ===========================================================================

class QwenGenerator:
    """Autoregressive decoding over the local Qwen2.5-3B.

    `use_adapter` attaches Stage 04's QLoRA adapter -- the fine-tuned domain
    model. That adapter was trained to WRITE briefings from incident logs, so
    it carries dispatcher register but not this JSON task; base-model decoding
    is the default and the adapter is offered for comparison.
    """

    def __init__(self, base_dir: Path = QWEN_BASE_DIR, use_adapter: bool = False,
                 adapter_dir: Path = QWEN_ADAPTER_DIR) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.use_adapter = bool(use_adapter)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        tokenizer_src = adapter_dir if (use_adapter and Path(adapter_dir).is_dir()) else base_dir

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src), trust_remote_code=True)
        if device_str == "cuda":
            from transformers import BitsAndBytesConfig
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(base_dir), quantization_config=quantization,
                device_map="auto", trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(base_dir), torch_dtype=torch.float32, trust_remote_code=True)
        if use_adapter and Path(adapter_dir).is_dir():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
        if device_str != "cuda":
            model = model.to(self.device)
        self.model = model
        self.model_id = f"Qwen2.5-3B{'+qlora' if use_adapter else ''} ({device_str})"
        self._check_checkpoint()

    def _check_checkpoint(self) -> None:
        """Refuse a damaged checkpoint instead of generating nonsense from it.

        This is not hypothetical: the copy of Qwen2.5-3B in Stage04_SLM scores a
        loss of 145 on plain English and repeats one token forever. Without this
        gate a 55-zone stress suite would have been built out of that and scored
        as though it were real.
        """
        problems = []

        embeddings = self.model.get_input_embeddings().weight
        norms = embeddings.detach().float().norm(dim=1)
        median = float(norms.median())
        outliers = int((norms > EMBEDDING_OUTLIER_RATIO * median).sum())
        if outliers:
            problems.append(
                f"{outliers} embedding rows exceed {EMBEDDING_OUTLIER_RATIO:.0f}x the median row "
                f"norm (median {median:.2f}, largest {float(norms.max()):.2f})")

        loss = self._probe_loss()
        if loss > HEALTH_PROBE_MAX_LOSS:
            problems.append(f"loss on plain English is {loss:.2f} (a working model scores "
                            f"under {HEALTH_PROBE_MAX_LOSS:.0f})")

        if problems:
            raise RuntimeError(
                f"{self.model_id} checkpoint is not usable: " + "; ".join(problems) +
                ". Generation will collapse to a single repeated token. Re-download the base "
                "model, e.g. `huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir "
                "Stage04_SLM/data/models/qwen_base`.")

    def _probe_loss(self) -> float:
        """Cross-entropy on one easy sentence: a single short forward pass."""
        ids = self.tokenizer(HEALTH_PROBE_TEXT, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            output = self.model(**ids, labels=ids["input_ids"])
        return float(output.loss)

    def __call__(self, system: str, user: str, temperature: float, top_p: float,
                 seed: int, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._torch.manual_seed(int(seed) % (2**31 - 1))
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(float(temperature), 1e-4),
                top_p=float(top_p),
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class GeminiGenerator:
    """The same decoding contract as `QwenGenerator`, served by the Gemini API.

    The generator only ever asks the backend for `(system, user) -> text`, so a
    hosted model drops into the same slot as the local one. Nothing else in the
    pipeline changes: the prompts, the retrieved few-shot exemplars, the JSON
    schema, the repair retries and the CVAE fallback are all backend-agnostic.

    Trade-offs against the local path, so a reader can judge the swap:
      * no 6 GB download, no GPU, no checkpoint health gate -- the weights are
        somebody else's problem, which is also why `_check_checkpoint` has no
        counterpart here;
      * technique 5 (fine-tuned domain LLM) is NOT available -- Stage 04's QLoRA
        adapter is a local artefact and cannot be attached to a hosted model, so
        `--adapter` and `--backend gemini` are mutually exclusive;
      * generation leaves the machine, and every zone costs a network round
        trip, so latency is dominated by the API rather than by decoding.

    The key is read from GEMINI_API_KEY (or GOOGLE_API_KEY) and is never written
    into any artefact -- the manifest records the model name only.
    """

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None,
                 max_attempts: int = GEMINI_MAX_ATTEMPTS) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:                                # pragma: no cover
            raise RuntimeError(
                "the Gemini backend needs the google-genai SDK: `pip install google-genai`"
            ) from error

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "no API key: set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment, "
                "e.g. PowerShell `$env:GEMINI_API_KEY = '...'`")

        self._types = types
        self._client = genai.Client(api_key=key)
        self._model = str(model)
        self._max_attempts = int(max_attempts)
        self.use_adapter = False
        self.model_id = f"{self._model} (gemini api)"

    def __call__(self, system: str, user: str, temperature: float, top_p: float,
                 seed: int, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=max(float(temperature), 0.0),
            top_p=float(top_p),
            seed=int(seed) % (2**31 - 1),
            max_output_tokens=int(max_new_tokens),
        )
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=user, config=config)
                return (response.text or "").strip()
            except Exception as error:                              # pragma: no cover - network
                # Rate limits and 5xx are the expected failures on a free key;
                # back off and retry rather than dropping the zone to fallback
                # over a transient. A persistent failure still falls back, which
                # is exactly what the CVAE path is there for.
                last_error = error
                if attempt == self._max_attempts - 1:
                    break
                time.sleep(GEMINI_BACKOFF_S * (2 ** attempt))
        raise RuntimeError(f"gemini call failed: {last_error}")


def build_backend(kind: str = "qwen", use_adapter: bool = False,
                  gemini_model: str = DEFAULT_GEMINI_MODEL):
    """Construct the requested backend, or raise with a reason the CLI can print."""
    if kind == "gemini":
        if use_adapter:
            raise RuntimeError("--adapter is a local QLoRA artefact and cannot be attached "
                               "to the hosted Gemini model; drop one of the two flags")
        return GeminiGenerator(model=gemini_model)
    if not QWEN_BASE_DIR.is_dir():
        raise RuntimeError(f"{QWEN_BASE_DIR} not found")
    return QwenGenerator(use_adapter=use_adapter)



# ===========================================================================
# 3. Prompt construction (techniques 2, 3, 6, 7)
# ===========================================================================

SYSTEM_PROMPT = (
    "You are a disaster-scenario generator for an Indian flood response agency. "
    "You invent PLAUSIBLE synthetic incidents that are used to stress-test an "
    "automated triage pipeline. You always answer in exactly two parts: a short "
    "REASONING block, then a single JSON object. You never invent fields that "
    "were not asked for, and the numbers you emit must be consistent with the "
    "message you write -- heavy rain implies many emergency calls, a calm zone "
    "implies few."
)

SEVERITY_GUIDE = {
    "ROUTINE": "a calm zone: light rain, river well under its threshold, a handful of calls",
    "ELEVATED": "a worsening zone: moderate rain, river approaching its threshold, some closures",
    "URGENT": "a serious zone: heavy rain, river over its threshold, many calls, roads cut",
    "CRITICAL": "a life-threatening zone: extreme rain, river far over threshold, rescue needed now",
}

FIELD_HINTS = {
    "rainfall_mm": "rainfall in the last hour, mm",
    "river_level_m": "observed river level, m",
    "river_level_threshold_m": "the danger threshold for this gauge, m",
    "emergency_calls": "emergency calls received this hour, integer",
    "road_closures": "roads closed, integer",
    "bridge_closures": "bridges closed, integer",
    "flood_history_count": "past floods on record for this district, integer",
    "population_affected": "people affected, integer",
    "water_level_change_m": "river level change over the last hour, m (negative if falling)",
}


def build_user_prompt(zone: dict, spec_context: dict, sensor_shots: list[dict],
                      text_shots: list[str], repair_note: str | None = None) -> str:
    """The task prompt: instructions + retrieved few-shot exemplars + schema."""
    lines: list[str] = []

    # -- technique 2: prompt engineering. The spec becomes an explicit brief.
    lines.append("TASK: generate one disaster zone for a synthetic stress-test scenario.")
    lines.append("")
    lines.append("SCENARIO BRIEF")
    lines.append(f"  intent          : {spec_context.get('prompt', '')}")
    night = " (night)" if "night" in (spec_context.get("global_modifiers") or []) else ""
    lines.append(f"  time of onset   : {spec_context.get('start_time', '')}{night}")
    lines.append(f"  location        : {zone['location']}, {zone['district']}, {zone['state']}")
    lines.append(f"  target severity : {zone['true_severity']} -- "
                 f"{SEVERITY_GUIDE[zone['true_severity']]}")
    lines.append(f"  hazards         : {', '.join(zone['hazards'])}")
    if zone.get("modifiers"):
        lines.append(f"  stress factors  : {', '.join(zone['modifiers'])}")
    if spec_context.get("global_modifiers"):
        lines.append(f"  scenario-wide   : {', '.join(spec_context['global_modifiers'])}")
    lines.append(f"  message style   : {zone.get('text_style', 'emergency dispatcher log entry')}")
    lines.append("")

    # -- techniques 3 + 4: retrieved real examples, numbers and language.
    if sensor_shots:
        risk_class = GENAI.SEVERITY_TO_CLASS[zone["true_severity"]]
        lines.append(f"REAL SENSOR ROWS retrieved from the historical record for a "
                     f"{risk_class}-risk zone (match this scale, do not copy):")
        for shot in sensor_shots:
            lines.append("  " + json.dumps(shot))
        lines.append("")
    if text_shots:
        lines.append("REAL DISPATCHER MESSAGES retrieved for this hazard and severity "
                     "(match this register, do not copy):")
        for shot in text_shots:
            lines.append(f"  - {shot.strip()}")
        lines.append("")

    # -- technique 6: chain of thought, before any JSON is emitted.
    lines.append("STEP 1 -- REASONING. In at most three sentences, state the physical "
                 "situation you are about to generate and check it is internally "
                 "consistent: does the rainfall justify the river level, and do both "
                 "justify the call volume and the severity you were asked for?")
    lines.append("")

    # -- technique 7: the schema the decoder must satisfy.
    lines.append("STEP 2 -- JSON. Then emit one JSON object, and nothing after it, with "
                 "exactly these keys:")
    for field in GENAI.NUMERIC_FEATURES:
        low, high = GENAI.PHYSICAL_LIMITS[field]
        kind = "integer" if field in GENAI.INTEGER_FEATURES else "number"
        lines.append(f'  "{field}": {kind}, {FIELD_HINTS[field]}, between {low} and {high}')
    lines.append('  "message": string, one realistic message of 20-45 words in the style above, '
                 f'naming {zone["location"]} and {zone["district"]}, reporting how many people '
                 'are affected')
    lines.append("")
    lines.append("FORMAT")
    lines.append("REASONING: <your reasoning>")
    lines.append("JSON: {...}")
    if repair_note:
        lines.append("")
        lines.append(f"CORRECTION -- your previous answer was rejected: {repair_note} "
                     "Emit the REASONING and JSON again, fixed.")
    return "\n".join(lines)


# ===========================================================================
# 4. Constrained decoding -- parse, validate, clamp (technique 7)
# ===========================================================================

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str) -> dict:
    """Pull the JSON object out of a REASONING/JSON answer.

    The model is asked for one object; it sometimes wraps it in a fence or
    trails a sentence after it. Both are recovered here rather than retried,
    because a retry costs a full decode and the fix is unambiguous.
    """
    text = raw.split("JSON:", 1)[1] if "JSON:" in raw else raw
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object in the answer")
    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Trailing commas are the one malformation worth repairing locally.
        parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
    if not isinstance(parsed, dict):
        raise ValueError("the JSON value is not an object")
    return parsed


def validate_payload(payload: dict, zone: dict) -> tuple[dict, str, list[str]]:
    """Schema + physics check. Returns (sensors, message, clamped_fields).

    Raises ValueError -- which triggers a repair retry -- when a field is
    missing, non-numeric, or the message is unusable. Values that are merely
    out of range are CLAMPED to the physical limits and reported, because a
    slightly hot number is still a usable scenario and a missing one is not.
    """
    missing = [f for f in GENAI.NUMERIC_FEATURES if f not in payload]
    if missing:
        raise ValueError(f"missing keys {missing}")
    message = str(payload.get("message", "")).strip()
    if len(message.split()) < 8:
        raise ValueError("'message' is missing or shorter than 8 words")

    sensors: dict[str, float] = {}
    clamped: list[str] = []
    for field in GENAI.NUMERIC_FEATURES:
        try:
            value = float(payload[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{field}' is not a number") from error
        if not np.isfinite(value):
            raise ValueError(f"'{field}' is not finite")
        low, high = GENAI.PHYSICAL_LIMITS[field]
        if not low <= value <= high:
            value = float(np.clip(value, low, high))
            clamped.append(field)
        sensors[field] = int(round(value)) if field in GENAI.INTEGER_FEATURES else round(value, 3)

    # One cross-field rule the schema cannot express: a zone the designer called
    # CRITICAL or URGENT must have its river at or above the danger threshold,
    # otherwise the sensors contradict the narrative generated alongside them.
    if zone["true_severity"] in {"URGENT", "CRITICAL"} and \
            sensors["river_level_m"] < sensors["river_level_threshold_m"]:
        raise ValueError(
            f"river_level_m ({sensors['river_level_m']}) is below river_level_threshold_m "
            f"({sensors['river_level_threshold_m']}) in a {zone['true_severity']} zone")
    return sensors, message, clamped


def _reasoning_of(raw: str) -> str:
    """The chain-of-thought block, kept for audit (truncated)."""
    if "REASONING:" not in raw:
        return ""
    block = raw.split("REASONING:", 1)[1].split("JSON:", 1)[0].strip()
    return block[:400]


# ===========================================================================
# 5. The generator
# ===========================================================================

class LLMScenarioGenerator:
    """Prompt spec -> multi-zone scenario, with sensors and text from the LLM.

    The structural scaffolding (district choice, evidence loss under blackout,
    gauge histories, imagery, ground-truth expectations) is reused from
    `ScenarioGenerator` -- that logic is about the SCENARIO, not about
    generation, and duplicating it would let the two paths drift. What this
    class replaces is the generative core: the sensor row and the message,
    which the LLM emits together in a single decode per zone.
    """

    def __init__(self, backend: QwenGenerator | GeminiGenerator | None = None,
                 processed_dir: Path = PROCESSED_DIR,
                 temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,
                 shots: int = DEFAULT_SHOTS, max_retries: int = DEFAULT_MAX_RETRIES,
                 verbose: bool = False) -> None:
        self.verbose = bool(verbose)
        self.base = GENAI.ScenarioGenerator(processed_dir=processed_dir)
        self.index = GroundingIndex(processed_dir)
        self.backend = backend
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.shots = int(shots)
        self.max_retries = int(max_retries)
        self.stats: dict[str, Any] = {"zones": 0, "llm_zones": 0, "fallback_zones": 0,
                                      "retries": 0, "clamped_fields": 0, "latency_ms": []}

    # ---- public API --------------------------------------------------------

    def generate(self, spec: dict, seed: int = SEED) -> dict[str, Any]:
        scenario = self.base.generate(spec, seed=seed)
        context = {
            "prompt": scenario.get("prompt", ""),
            "start_time": scenario.get("start_time", ""),
            "global_modifiers": scenario.get("global_modifiers", []),
        }
        for index, zone in enumerate(scenario["zones"]):
            self._rewrite_zone(zone, context, seed=seed + index)
            if self.verbose:
                state = zone["provenance"].get("llm", {})
                print(f"  [{scenario['scenario_id']}/{zone['zone_id']}] "
                      f"{state.get('status', '?')}"
                      f"{' in ' + str(state['latency_ms']) + ' ms' if state.get('latency_ms') else ''}",
                      flush=True)
        total = int(sum(z["headcount_reported"] for z in scenario["zones"]))
        scenario["demand"]["total_headcount_reported"] = total
        scenario["demand"]["shelter_gap"] = max(
            0, total - int(scenario["demand"].get("shelter_beds", 0)))
        scenario["generator"] = {
            "family": "sequence_llm",
            "model": self.backend.model_id if self.backend else "unavailable (CVAE fallback)",
            "techniques": self._techniques(),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "few_shot_k": self.shots,
            "seed": int(seed),
            "fallback_path": "SensorCVAE + phrase bank (03_genai_engineer.py)",
        }
        return scenario

    def generate_suite(self, prompts: list[dict], base_seed: int = SEED) -> list[dict]:
        return [self.generate(spec, seed=base_seed + index * 10)
                for index, spec in enumerate(prompts)]

    def _techniques(self) -> list[str]:
        extra = ["fine_tuned_domain_llm"] if (self.backend and self.backend.use_adapter) else []
        return TECHNIQUES + extra

    # ---- per-zone generation ----------------------------------------------

    def _rewrite_zone(self, zone: dict, context: dict, seed: int) -> None:
        """Replace one zone's sensors and message with an LLM generation.

        A zone whose telemetry and text were BOTH lost (a blackout zone) has
        nothing for the LLM to write, and must stay silent -- generating text
        for it would destroy the scenario's whole point.
        """
        self.stats["zones"] += 1
        available = zone["provenance"]["evidence_available"]
        if not (available["sensors"] or available["text"]):
            zone["provenance"]["llm"] = {"status": "skipped", "reason": "no surviving evidence"}
            return
        if self.backend is None:
            self.stats["fallback_zones"] += 1
            zone["provenance"]["llm"] = {"status": "fallback", "reason": "no LLM backend loaded",
                                         "techniques": []}
            return

        target = dict(zone["inputs"]["sensors"] or {})
        risk_class = zone["provenance"]["sensor_class"]
        sensor_shots = self.index.sensor_exemplars(risk_class, target, self.shots)
        text_severity = GENAI.SEVERITY_TO_TEXT[zone["true_severity"]]
        text_shots = self.index.text_exemplars(zone["hazards"][0], text_severity, self.shots)
        zone_view = dict(zone)
        zone_view["text_style"] = "emergency dispatcher log entry"

        note: str | None = None
        started = time.time()
        for attempt in range(self.max_retries + 1):
            user = build_user_prompt(zone_view, context, sensor_shots, text_shots, note)
            try:
                raw = self.backend(SYSTEM_PROMPT, user, self.temperature, self.top_p,
                                   seed=seed + attempt * 977)
                sensors, message, clamped = validate_payload(extract_json(raw), zone)
            except (ValueError, json.JSONDecodeError) as error:
                note = str(error)
                self.stats["retries"] += 1
                continue

            latency = (time.time() - started) * 1000
            self.stats["latency_ms"].append(round(latency, 1))
            self.stats["llm_zones"] += 1
            self.stats["clamped_fields"] += len(clamped)
            self._apply(zone, sensors, message)
            zone["provenance"]["llm"] = {
                "status": "ok",
                "model": self.backend.model_id,
                "techniques": self._techniques(),
                "attempts": attempt + 1,
                "clamped_fields": clamped,
                "retrieved_sensor_rows": len(sensor_shots),
                "retrieved_texts": len(text_shots),
                "reasoning": _reasoning_of(raw),
                "latency_ms": round(latency, 1),
            }
            return

        self.stats["fallback_zones"] += 1
        zone["provenance"]["llm"] = {
            "status": "fallback",
            "reason": f"rejected after {self.max_retries + 1} attempts: {note}",
            "model": self.backend.model_id,
            "techniques": self._techniques(),
        }

    def _apply(self, zone: dict, sensors: dict, message: str) -> None:
        """Write the LLM's sensors and message back into the zone."""
        if zone["inputs"]["sensors"] is not None:
            zone["inputs"]["sensors"].update(sensors)
        if zone["inputs"]["text"] is not None and zone["text_entries"]:
            top = max(range(len(zone["text_entries"])),
                      key=lambda k: (GENAI.TEXT_LEVELS.index(zone["text_entries"][k]["severity"]), k))
            entry = zone["text_entries"][top]
            entry["text"] = message
            entry["source"] = "llm"
            zone["inputs"]["text"] = message
            zone["inputs"]["incident_log"] = GENAI.ScenarioGenerator._incident_log(
                zone["text_entries"], zone["state"], zone["district"], zone["zone_id"])
            zone["headcount_reported"] = int(sum(e["headcount"] for e in zone["text_entries"]))
        # The gauge history must end where the LLM put the river, or the two
        # pieces of evidence contradict each other.
        history = zone["inputs"].get("water_levels")
        if history and "river_level_m" in sensors:
            offset = float(sensors["river_level_m"]) - float(history[-1])
            zone["inputs"]["water_levels"] = [round(float(v) + offset, 3) for v in history]

    # ---- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        latencies = self.stats["latency_ms"]
        attempted = self.stats["llm_zones"] + self.stats["fallback_zones"]
        return {
            "zones": self.stats["zones"],
            "llm_generated_zones": self.stats["llm_zones"],
            "fallback_zones": self.stats["fallback_zones"],
            "fallback_rate": round(self.stats["fallback_zones"] / max(attempted, 1), 4),
            "repair_retries": self.stats["retries"],
            "clamped_fields": self.stats["clamped_fields"],
            "latency_ms": {
                "p50": round(float(np.percentile(latencies, 50)), 1) if latencies else None,
                "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
            },
        }


# ===========================================================================
# 6. CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based scenario generation (Stage 05)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="higher = wilder edge cases (sampling / temperature)")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                        help="retrieved few-shot exemplars per zone")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="constrained-decoding repair attempts before fallback")
    parser.add_argument("--adapter", action="store_true",
                        help="attach Stage 04's fine-tuned QLoRA disaster adapter (qwen only)")
    parser.add_argument("--backend", choices=("qwen", "gemini"), default="qwen",
                        help="qwen = local Qwen2.5-3B; gemini = hosted Gemini API "
                             "(needs GEMINI_API_KEY, no local download)")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL,
                        help=f"hosted model name (default {DEFAULT_GEMINI_MODEL})")
    parser.add_argument("--limit", type=int, default=0, help="generate only the first N prompts")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip the LLM entirely and exercise the fallback path")
    parser.add_argument("--quiet", action="store_true", help="suppress per-zone progress")
    args = parser.parse_args()

    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    library = GENAI.load_prompts()
    prompts = library["prompts"][:args.limit] if args.limit else library["prompts"]

    backend = None
    if not args.dry_run:
        if args.backend == "gemini":
            print(f"[llm] connecting to {args.gemini_model} over the Gemini API ...")
        else:
            print(f"[llm] loading {QWEN_BASE_DIR.name}"
                  f"{' + qlora adapter' if args.adapter else ''} ...")
        t0 = time.time()
        try:
            backend = build_backend(args.backend, use_adapter=args.adapter,
                                    gemini_model=args.gemini_model)
            print(f"[llm] {backend.model_id} ready in {time.time() - t0:.1f}s")
        except RuntimeError as error:
            print(f"[llm] LLM BACKEND UNAVAILABLE -- {error}")
            print("[llm] continuing on the CVAE fallback path; every zone will be "
                  "marked `fallback` and the manifest will show a 100% fallback rate.")

    generator = LLMScenarioGenerator(backend, temperature=args.temperature, top_p=args.top_p,
                                     shots=args.shots, max_retries=args.max_retries,
                                     verbose=not args.quiet)

    print(f"[llm] generating {len(prompts)} scenarios + wildcard ...")
    suite = generator.generate_suite(prompts, base_seed=args.seed)
    wildcard = generator.generate(library["wildcard"], seed=args.seed + 100)

    # Same envelope as 03_genai_engineer.py writes, so 04_evaluation_engineer.py
    # can score an LLM suite with --suite without any special-casing.
    LLM_SUITE_JSON.write_text(
        json.dumps({"seed": args.seed, "generator": "sequence_llm", "scenarios": suite},
                   indent=2), encoding="utf-8")
    LLM_WILDCARD_JSON.write_text(json.dumps(wildcard, indent=2), encoding="utf-8")
    GENAI.flatten_zones(suite + [wildcard]).to_csv(LLM_ZONES_CSV, index=False)

    summary = generator.summary()
    LLM_MANIFEST_JSON.write_text(json.dumps({
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "family": "sequence_llm",
        "model": backend.model_id if backend else "unavailable (CVAE fallback)",
        "techniques": generator._techniques(),
        "temperature": args.temperature, "top_p": args.top_p, "few_shot_k": args.shots,
        "max_retries": args.max_retries, "seed": args.seed,
        "scenarios": len(suite) + 1,
        "summary": summary,
    }, indent=2), encoding="utf-8")

    print(f"[llm] zones {summary['zones']} | LLM {summary['llm_generated_zones']} | "
          f"fallback {summary['fallback_zones']} ({summary['fallback_rate']:.1%}) | "
          f"retries {summary['repair_retries']} | clamped {summary['clamped_fields']}")
    if summary["latency_ms"]["p50"] is not None:
        print(f"[llm] latency p50 {summary['latency_ms']['p50']} ms / "
              f"p95 {summary['latency_ms']['p95']} ms per zone")
    print(f"[llm] -> {LLM_SUITE_JSON.name}, {LLM_WILDCARD_JSON.name}, {LLM_MANIFEST_JSON.name}")


if __name__ == "__main__":
    main()
