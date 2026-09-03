# Integration Engineer Guide

## Scope

This foundation connects the Data, EDA, ML, and Evaluation outputs when their contracts are finalized. It currently serves health and readiness information only. **PENDING TEAM INPUT** is the intentional state for every team-dependent value.

## Required handoff

**Data Engineer** must provide the approved artifact reference, schema reference, row identity, time semantics, provenance, delivery format, and a quality report.

**EDA Engineer** must provide the validated-data reference, quality findings, approved transformations, distribution or drift notes, and known limitations.

**ML Engineer** must provide the model artifact and version, exact input schema, feature order, preprocessing definition, target definition, output schema, inference requirements, and the risk formula and thresholds if applicable. The integration layer will not infer any of these.

**Evaluation Engineer** must provide the evaluation artifact, complete metrics, acceptance criteria, validation protocol, error analysis, and known limitations.

## Enablement gate

1. Replace all template values in `integration_contracts/` with team-approved values.
2. Verify that contract versions are mutually compatible.
3. Add concrete adapter implementations beside the pending adapters.
4. Add contract and integration tests using approved fixtures.
5. Change `inference_enabled` only after the Evaluation Engineer and integration owner approve the compatibility decision.

Until then, `/api/v1/model/predict` returns HTTP 501 and the dashboard shows empty states.
