# Retinal RNA and spatial research pipeline

Separate RNA (scVI) and spatial (Novae) analysis, with donor-separated technical benchmarks and descriptive outputs. This is research software, not a validated FTD predictor.

Use Python 3.12 with scvi-tools 1.5.2 and novae 1.1.3. CPU checks also used PyTorch 2.15.0. Package compatibility on your deployment machine requires verification.

Run synthetic checks with `PYTHONPATH=src python -m unittest discover -s tests -q`. Example configs intentionally contain unresolved placeholders and must not be treated as verified biological mappings. Keep all study inputs, mappings, outputs and trained weights in approved research storage.

The command interface is `PYTHONPATH=src python -m ftd_prediction --help`. The separate neural runner is `PYTHONPATH=src python -m ftd_prediction.neural --help`.

This code-only folder contains no study data, private configuration, model weights, virtual environments, project handoff or lab setup documents. The staged implementation passed 203 synthetic tests locally; this does not establish biological performance or cluster readiness.
