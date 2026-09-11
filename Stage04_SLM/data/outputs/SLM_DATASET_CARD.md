# Stage 04 (SLM) — Dataset Card

**Task:** abstractive summarisation of multi-entry disaster incident logs into a
two-sentence tactical briefing, for fine-tuning a small language model that runs
offline on edge hardware.

**Built from:** `Stage03_NLP/data/processed/Dispatcher_Log_Master_60000_Processed.csv`
(60,000 records), exactly as the Stage 04 brief specifies ("report-to-summary
training pairs generated from Stage 03").

**Reproduce with:**
```bash
python build_slm_dataset.py --source Stage03_NLP --out slm_dataset \
       --min-entries 5 --max-entries 12 --window-minutes 480
```
Seed is fixed at 42. The build is deterministic.

---

## 1. Files

| File | Rows | Purpose |
| --- | ---: | --- |
| `SLM_Report_Summary_Pairs.jsonl` | 3,218 | Fine-tuning file, instruction format |
| `SLM_Report_Summary_Pairs.csv` | 3,218 | Same rows, tabular, for inspection |
| `SLM_Domain_Dictionary.csv` | 62 | Evacuation terms, resource codes, shorthand |
| `SLM_dataset_statistics.json` | — | Measured statistics (source of every number below) |
| `build_slm_dataset.py` | — | The builder |

---

## 2. Why the reports are multi-entry

A single Stage 03 dispatcher message is one or two sentences. Summarising one
message into two sentences achieves **no compression** and would not train a
summariser at all.

Real incident logs are a *stream* of entries about the same developing
situation. So records are grouped into incident logs — same **state + district +
zone**, inside a rolling **8-hour** window, 5 to 12 entries each — and the
concatenated entries become the report. That produces genuine compression, which
is what the brief's ">80% reduction" huddle target requires.

---

## 3. Measured statistics

| Property | Value |
| --- | --- |
| Pairs | **3,218** |
| Splits | train 2,256 · validation 481 · test 481 |
| Entries per report | 5 – 12 (mean 5.9) |
| Report length | 139 – 378 words (mean **180.9**) |
| Summary length | 22 – 38 words (mean **30.2**) |
| **Compression** | 75.9% – 92.2% (**mean 82.8%**) |
| Geographic coverage | 10 states, 50 districts, 5 zones |
| Source records consumed | 18,971 of 60,000 (each used at most once) |

### Priority balance

| Briefing priority | Worst entry | Pairs |
| --- | --- | ---: |
| IMMEDIATE | CRITICAL | 983 |
| URGENT | HIGH | 983 |
| ELEVATED | MEDIUM | 983 |
| ROUTINE | LOW | **269** |

Three classes are balanced exactly. **ROUTINE is under-represented at 269** and
this is a genuine ceiling, not a choice: LOW is only 8,977 of the 60,000 source
records, spread across 500 location-zone combinations, so there are only so many
all-LOW logs that can be formed. Weight the loss or oversample ROUTINE during
fine-tuning if the model under-produces it.

### Meeting the brief's timing target

At a normal radio speaking rate of ~2.5 words/second:

- reading the full log ≈ **72 seconds**
- reading the summary ≈ **12 seconds**
- **time saving 83.3%** — clears the brief's ">80% reduction" test

> **One honest discrepancy.** The brief asks for both a "5-second voice briefing"
> *and* "2 crisp sentences". Those conflict: two informative sentences run about
> 30 words, which is ~12 seconds spoken. This dataset targets the two-sentence
> spec. If the 5-second figure is the hard requirement, cut to a single ~13-word
> sentence — shorten `SITUATION_TEMPLATES` and drop `ACTION_TEMPLATES` in the
> builder — and accept that the resource and action detail is lost.

---

## 4. Schema

### `SLM_Report_Summary_Pairs.jsonl`

```json
{
  "pair_id": "SLM-000001",
  "split": "train",
  "grouping": "natural",
  "instruction": "You are a disaster-response briefing assistant. Read the incident log and reply with exactly two sentences: first the situation (priority, hazard, location, scale), then the required action. Be terse enough to be read aloud over radio in under five seconds.",
  "input": "INCIDENT LOG | Kerala / Kottayam / ZONE-5 | 7 entries\n[02-02-2026 22:44] ERSS-010582 | Medical emergency reported ...",
  "output": "URGENT: medical emergencies and rescue emergencies plus 2 more, ZONE-5 Kottayam. 7 reports, 115 affected. Send ambulance and fire and rescue unit plus 1 more to residential colony; commit teams, verify routes.",
  "metadata": { "priority": "URGENT", "worst_severity": "HIGH", "hazard_types": ["Medical Emergency", "Rescue Emergency"], "total_headcount": 115, "resources_required": ["ambulance", "fire and rescue unit"], "locations": ["residential colony"], "state": "Kerala", "district": "Kottayam", "zone": "ZONE-5", "entry_count": 7, "report_words": 201, "summary_words": 32, "compression_pct": 84.1 }
}
```

### CSV columns

`pair_id`, `source`, `grouping`, `report`, `summary`, `priority`,
`worst_severity`, `hazard_types`, `total_headcount`, `resources_required`,
`locations`, `state`, `district`, `zone`, `entry_count`, `window_minutes`,
`first_timestamp`, `last_timestamp`, `incident_ids`, `report_words`,
`summary_words`, `compression_pct`, `split`

`incident_ids` lists the Stage 03 records behind each pair, so any briefing can
be traced back to source.

---

## 5. Summary format

Sentence 1 — **situation**: priority keyword, hazard mix, zone and district,
number of reports, total people affected.
Sentence 2 — **action**: resources required, primary location, directive.

The priority keyword is the worst severity in the log, mapped as
LOW→ROUTINE, MEDIUM→ELEVATED, HIGH→URGENT, CRITICAL→IMMEDIATE. Taking the worst
entry rather than the average is deliberate: a log containing one CRITICAL entry
is a critical situation regardless of what surrounds it.

---

## 6. Validation performed

Every check below was run on the final build and passed at 3,218/3,218.

| Check | Result | Why it matters |
| --- | --- | --- |
| Zone named in summary appears in report | 3,218/3,218 | A summary asserting an unstated fact teaches hallucination |
| District appears in report | 3,218/3,218 | Same |
| State appears in report | 3,218/3,218 | Same |
| Headcount equals the sum in the report | 3,218/3,218 | Numbers must be derivable, not invented |
| Entry count equals the log's line count | 3,218/3,218 | Same |
| Primary location appears in report | 3,218/3,218 | Same |
| Incident IDs leaking across splits | **0** | A test report must contain no text seen in training |
| Records reused across pairs | **0** | No two reports share content |
| Duplicate reports / summaries | 0 / 0 | No trivially memorisable duplicates |

Three defects were found and fixed during construction, and they are worth
knowing about because they are the failure modes to watch for if you regenerate:

1. **Zone was unfaithful.** Stage 03's `text` field names only the district and
   state, but summaries mentioned the zone — 0/4,184 reports contained it. Fixed
   by adding an `INCIDENT LOG | state / district / zone | N entries` header,
   which is metadata a real log carries anyway.
2. **1,548 incident IDs leaked across splits.** The natural and severity-banded
   grouping passes drew from the same records, so one entry could appear in two
   groups landing in different splits. Fixed by claiming every record at most
   once.
3. **ROUTINE was degenerate at 15 examples.** Because the priority is the worst
   entry, mixed logs are almost always HIGH or CRITICAL. Fixed by reserving
   scarce severity bands before natural grouping runs.

---

## 7. The domain dictionary

62 terms in `SLM_Domain_Dictionary.csv`, each tagged with an `origin` column so
it is always clear what is real and what is a project convention.

| Category | Terms |
| --- | ---: |
| Resource | 19 |
| Location type | 15 |
| Incident-command term | 10 |
| Agency | 6 |
| Hazard type | 4 |
| Severity level | 4 |
| Briefing priority | 4 |

| `origin` value | Terms | Meaning |
| --- | ---: | --- |
| derived from Stage 03 corpus | 38 | Extracted from actual corpus vocabulary, with occurrence counts |
| standard incident-command terminology | 10 | Real, widely-published ICS terms (IC, SITREP, TRIAGE, EVAC, ETA …) |
| real agency referenced in project data | 6 | NDRF, SDRF, CWC, IMD, NDMA, ERSS |
| project-defined (Stage 03 label space) | 4 | The four severity levels |
| project-defined (Stage 04 summary format) | 4 | The four briefing priorities |

> **On radio shorthand.** The `shorthand` column is a **project-defined
> controlled vocabulary** (e.g. `RES-AMBU`, `HZ-F`, `SEV-C`), generated
> systematically from the corpus. It is **not** an official agency codebook, and
> no entry claims to be a real ten-code. Official codes vary by jurisdiction; if
> you need genuine ones, source them from the relevant agency. The ten
> incident-command terms are real and standard.

---

## 8. Limitations — read before quoting results

1. **The summaries are templated.** Sentence structures are sampled from 4
   situation and 3 action variants with 3 directives per severity, so a
   fine-tuned model learns to reproduce *these templates*, not to summarise
   open-domain English. Strong scores on the held-out split mean the model
   learned the format and the extraction, not general summarisation.
2. **The source corpus is itself synthetic.** Stage 03's dispatcher text is
   generated from f-string templates. Every limitation of that corpus is
   inherited here.
3. **The supervision is extractive in disguise.** Every fact in a summary is
   aggregated from structured columns of the same records. That is what makes it
   faithful and checkable, but it also means the task is closer to
   *structured-field aggregation and verbalisation* than to true abstractive
   summarisation. Say so when presenting results.
4. **ROUTINE is under-represented** (269 vs 983) — see §3.
5. **Only the dispatcher corpus is used.** Stage 03's social feeds and safety
   SOPs are not included; social posts are single short messages and do not form
   multi-entry logs.
6. **18,971 of 60,000 source records are consumed.** The rest could not be
   placed in a disjoint 5+ entry group within the window. Widening
   `--window-minutes` or lowering `--min-entries` uses more of them, at the cost
   of compression.
7. **No held-out human reference summaries.** There is no human-written gold
   standard to compare against, so ROUGE or BERTScore here measures agreement
   with the template, not with human judgement.

---

## 9. Suggested use

- Fine-tune with parameter-efficient tuning (LoRA / QLoRA) on the `train` split;
  the brief's Stage 04 spec names PEFT explicitly.
- Select on `validation`; report on `test` once.
- Report **compression ratio and spoken-time saving** alongside any text-overlap
  metric — they are the numbers the brief's huddle test actually asks for.
- Consider weighting or oversampling ROUTINE so the model retains the ability to
  say "monitor only", which is the operationally safest output and the rarest
  label here.
- Feed the domain dictionary in as a token-preservation check: the brief's EDA
  role asks you to "audit token distribution to ensure specialized radio codes
  are retained".
