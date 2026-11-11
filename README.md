# Retinal research code package

Prepared 2026-11-11 for reproducible PMACS deployment. Exact payload bytes are listed in
MANIFEST.json. This package extends the earlier synthetic-tested deployment at
95f3305084fd8b3f899e7dbba207f5e9c8ef9bd9; it is not that Git revision.

Contents: source modules, synthetic tests, placeholder examples, schema/discovery
helpers and selected technical documentation. It contains no study data, private
configs, donor mappings, credentials, handoffs or model weights.

Use the existing verified scientific Python environment and PYTHONPATH=src.
Set PYTHONDONTWRITEBYTECODE=1 (or use python -B) to keep the package unchanged.
The minimal pyproject does not install optional scientific/model backends.

Entry points:

- python -m ftd_prediction.pmacs_ready verify --code-directory ABS_CODE --manifest ABS_CODE/MANIFEST.json
- python scripts/pmacs_discover_inputs.py --root APPROVED_ROOT --output NEW_PRIVATE_DIRECTORY
- python -m ftd_prediction.descriptive --config PRIVATE_CONFIG --output NEW_PRIVATE_RUN
- python -m ftd_prediction.descriptive_report --run COMPLETED_RUN --output NEW_PRIVATE_REPORT
- python -m ftd_prediction.molecular --config PRIVATE_CONTRAST_CONFIG --output NEW_PRIVATE_SUMMARY
- python -m ftd_prediction.neural --config PRIVATE_CONFIG --output NEW_PRIVATE_RUN

Study execution requires PMACS/LSF context and authoritative mappings/count
provenance. Examples deliberately contain unresolved placeholders and fail until
verified facts are supplied. Reports and outputs remain in approved private
storage. See the separately supplied morning checklist for deployment, resources,
explicit configuration and authorized submission commands. Deployment must export
only the manifest-listed payload and MANIFEST.json into a new directory: a Git
checkout's .git metadata and .gitignore are not part of the verified runtime package.

RNA and spatial assays are handled as separate, unpaired donor cohorts.
Specimen-level description does not establish replicated disease associations or
validated donor prediction. Synthetic checks establish software behavior only.
Public Novae weights require separate acquisition and the documented pinned
expression-only compatibility mode; strict loading remains the default.
