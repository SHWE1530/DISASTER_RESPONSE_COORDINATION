# Integration Contracts

This directory contains versioned boundaries between the four Stage-01 team outputs and the integration service.

## Rule

Every field that depends on another team is **PENDING TEAM INPUT** until the owning team supplies and signs off on it. The integration layer must not infer dataset names, columns, feature order, target, preprocessing, model, thresholds, formulas, metrics, or predictions.

## Required contract files

- `data_output_contract.json`: Data Engineer output location, schema/version, provenance, quality status, and load instructions.
- `eda_output_contract.json`: EDA findings, validation checks, limitations, and approved data-quality notes.
- `ml_output_contract.json`: Model artifact reference, exact input/output schema, feature order, preprocessing, target mapping, and inference lifecycle.
- `evaluation_output_contract.json`: Evaluation artifact reference, metrics, acceptance criteria, test split/protocol, and limitations.
- `integration_contract.json`: Approved cross-team versions, compatibility status, and ownership.

The JSON files in this directory are templates only. Replace `PENDING TEAM INPUT` values with team-approved content before enabling real inference.
