# Stage 06 agent evaluation

Generated 2026-09-13T09:39:03+00:00 (commit 9bda9a4). Stage status: {'ml': 'healthy', 'dl': 'healthy', 'nlp': 'healthy', 'slm': 'healthy'}.

## Seen vs unseen

- **cvae**: development: failures on this suite were inspected while tuning the agent's rules
- **slm**: held-out: never used to design or tune any rule
- **decision probes**: hand-written one-ambulance trade-offs, never used for tuning

## Decision probes

| Probe | Expected | Allocated to | Rerouted | Approved | Override recalled | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| P1 One ambulance, two severe zones: 38 vs 15 emergency calls | ZONE-7 | ZONE-7 | yes | yes | yes | PASS |
| P2 Same case, zones listed in the opposite order | ZONE-7 | ZONE-7 | yes | yes | yes | PASS |
| P3 Equal call volume, more people waiting in one zone | ZONE-B | ZONE-B | yes | yes | yes | PASS |
| P4 Sensor-confirmed zone vs a dramatic report the sensors contradict | ZONE-C | ZONE-C | yes | yes | yes | PASS |

## Agent, by generator suite

| Metric | cvae | slm |
| --- | ---: | ---: |
| Scenario goal success | 0.857 | 0.762 |
| Zone goal success | 0.927 | 0.855 |
| Priority exact accuracy | 0.704 | 0.611 |
| Within-one accuracy | 0.907 | 0.852 |
| Critical misses | 0 | 0 |
| Under-triaged high-severity zones | 1 | 2 |
| Stage 05 expectation pass rate | 0.946 | 0.855 |
| Human-review recall | 1.000 | 1.000 |
| Tool precision | 0.987 | 0.982 |
| Tool recall | 1.000 | 1.000 |
| Argument validity | 1.000 | 1.000 |
| Tool error rate | 0.000 | 0.000 |
| Workflow violations | 0 | 0 |
| Unsafe commits | 0 | 0 |
| High-severity need coverage | 0.910 | 0.910 |
| Priority inversions | 0 | 1 |
| Bad trade-off risk (confident & wrong) | 0.000 | 0.018 |
| Exact accuracy when confident | 0.857 | 0.824 |
| Exact accuracy when not confident | 0.421 | 0.250 |
| Low-confidence URGENT+ escalated | 1.000 | 1.000 |
| Plan adherence | 1.000 | 1.000 |
| Debate objections | 2 | 0 |
| Reflexion revisions | 0 | 0 |
| Run latency p50 (ms) | 186.200 | 155.000 |

## Ablations and fault injection (cvae suite)

| Metric | agent | fcfs_allocation | no_safety_auditor | stage01_outage | stage03_outage | stage02_flaky |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scenario goal success | 0.857 | 0.857 | 0.048 | 0.619 | 0.762 | 0.857 |
| Zone goal success | 0.927 | 0.927 | 0.255 | 0.800 | 0.855 | 0.927 |
| Priority exact accuracy | 0.704 | 0.704 | 0.704 | 0.518 | 0.471 | 0.704 |
| Within-one accuracy | 0.907 | 0.907 | 0.907 | 0.870 | 0.843 | 0.907 |
| Critical misses | 0 | 0 | 0 | 2 | 0 | 0 |
| Under-triaged high-severity zones | 1 | 1 | 1 | 11 | 0 | 1 |
| Stage 05 expectation pass rate | 0.946 | 0.946 | 0.946 | 0.800 | 0.855 | 0.946 |
| Human-review recall | 1.000 | 1.000 | 1.000 | 0.921 | 1.000 | 1.000 |
| Tool precision | 0.987 | 0.987 | 0.987 | 0.987 | 0.981 | 0.987 |
| Tool recall | 1.000 | 1.000 | 1.000 | 0.981 | 0.968 | 1.000 |
| Argument validity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Tool error rate | 0.000 | 0.000 | 0.000 | 0.169 | 0.160 | 0.092 |
| Workflow violations | 0 | 0 | 0 | 0 | 0 | 0 |
| Unsafe commits | 0 | 0 | 39 | 0 | 0 | 0 |
| High-severity need coverage | 0.910 | 0.640 | 0.910 | 0.930 | 0.916 | 0.910 |
| Priority inversions | 0 | 0 | 0 | 0 | 0 | 0 |
| Bad trade-off risk (confident & wrong) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Exact accuracy when confident | 0.857 | 0.857 | 0.857 | 0.800 | 0.455 | 0.857 |
| Exact accuracy when not confident | 0.421 | 0.421 | 0.421 | 0.410 | 0.500 | 0.421 |
| Low-confidence URGENT+ escalated | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| Plan adherence | 1.000 | 1.000 | 1.000 | 1.000 | 0.976 | 1.000 |
| Debate objections | 2 | 0 | 2 | 1 | 2 | 2 |
| Reflexion revisions | 0 | 0 | 0 | 0 | 0 | 0 |
| Run latency p50 (ms) | 186.200 | 9.300 | 8.700 | 125.600 | 99.200 | 166.000 |

`gemini_planner`: GEMINI_API_KEY not set; not run

## Human-in-the-loop probe

- Dispatches held for approval: 39
- Commit attempts without / with a forged token: 78, blocked: 78
- Approved by a simulated human and committed: 39
- Emergency override then recalled: 39
- Commit attempts after the halt: 39, blocked: 39

## Failure categories

- **cvae**: {'over_triage': 3, 'over_dispatch': 2}
- **slm**: {'over_triage': 4, 'conflict_missed': 3, 'under_triage': 1, 'critical_unserved': 1}

## Failed zones (cvae suite)

| Scenario | Zone | True | Agent | Failures |
| --- | --- | --- | --- | --- |
| S05 Negated flood rumour | ZONE-1 | ROUTINE | URGENT | expectation: priority URGENT above maximum ELEVATED; heavy assets sent to a ROUTINE zone |
| S09 People versus sensors | ZONE-2 | ROUTINE | URGENT | heavy assets sent to a ROUTINE zone |
| S18 All quiet (do not cry wolf) | ZONE-2 | ROUTINE | URGENT | expectation: priority URGENT above maximum ELEVATED |
| S18 All quiet (do not cry wolf) | ZONE-3 | ROUTINE | URGENT | expectation: priority URGENT above maximum ELEVATED |
