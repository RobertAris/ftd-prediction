# Private specimen-level molecular contrasts

Study work requires explicit execution authorization. Study work and all outputs
remain in approved private PMACS storage; this document grants no job, transfer
or sharing permission. Synthetic tests establish no biological finding.

Use an existing NumPy/SciPy environment:

```sh
PYTHONPATH=src python -m ftd_prediction.molecular \
  --config /PRIVATE/molecular.json --output /PRIVATE/outputs/NEW_MOLECULAR_RUN \
  --mode study --execution-context pmacs
```

Callable: `ftd_prediction.molecular.execute(config_path, output, mode='study',
execution_context='pmacs')`. Study requires positive numeric `LSB_JOBID`; the
runtime scope guard precedes all source/universe access. Invented fixtures use
`data_kind: "synthetic"`, `--mode synthetic --execution-context local`.
CLI messages are generic. Failures after output creation retain private error
logs and failed provenance; pre-creation failures cannot have an output log.

Copy the [failing example](../../configs/molecular-contrast.example.json) to
private storage. Resolve every placeholder. Schema 1 rejects unknown fields,
duplicate JSON keys, empty/false/unresolved declarations and threshold overrides.

- Declare matching `data_kind`, `assay: "rna" | "spatial"`, `source_run_dir`
  (absolute in study; synthetic relative paths resolve against the config),
  and `source_run_sha256`: exact lowercase SHA256 of its accepted
  `provenance/run.json`. Acceptance itself is an owner responsibility.
- Supply resolved `source_annotation_evidence_ref` and
  `feature_universe_evidence_ref`; these owner declarations are not independently
  validated or opened as files.
- Declare exactly one of `endogenous_gene_ids_file` (absolute private CSV, sole
  column `gene_id`, unique nonempty IDs all present in source) OR
  `all_features_endogenous_verified: true`. Remove the other key entirely.
  Controls must be excluded from both denominator and reported rows. No symbol
  or prefix identifies controls; all-features verification asserts none remain.
- Every contrast requires a unique `contrast_id`, `case_donor_id`, unique nonempty
  `reference_donor_ids` excluding the case, exact `retinal_region` and `cell_type`,
  and resolved `label_evidence_ref`/`matching_evidence_ref`. All donors must exist;
  clinical labels, identities and comparable populations are never inferred.

Only the source run record and its `pseudobulk_counts.npz`,
`pseudobulk_groups.csv`, `pseudobulk_genes.csv` are consumed, in addition to the
private config/universe. Source schema 1 must be completed, matching kind/assay,
training=false, disease-inference=false, with consistent dimensions/missing
flags and exact SHA256/byte sizes for consumed artifacts. Spatial sources need
recorded physical sections and resolved `configuration.segmentation_ref`;
compare segmentation variants in separate runs. Extra grouping columns fail.
Raw upstream inputs, H5AD, cell mappings and models are never opened/imported.
No network, training, gene programs or inference is implemented.

Paths reject symlinks including ancestors. A shared group/world-writable ancestor
must be followed by an owned directory with no group/other permissions; otherwise
it fails. Later writable descendants require another such private boundary.
Root-owned sticky temporary directories are exempt. This supports private
workspaces beneath shared lab storage, without accepting a writable source
directory itself. Actual PMACS permissions remain unverified. Use physical paths, including for
macOS synthetic temporary directories. Config/universe need mode 600 or a private
owned enclosing directory. Output must be a fresh child of an existing trusted
parent, disjoint in both directions from source/config/universe. Output directories
are 700, files 600; choose a new output after failure.

Sum raw counts and cell numbers across all sections of each exact donor × region
× cell type **before** normalization. Other regions, missing annotations/regions
and missing spatial sections are excluded with reasons. Unrecorded RNA sections
are not counted as a fabricated section. Each population needs **20 cells** and
positive endogenous library `L_d`. Fixed definitions: `CPM = 1e6 * counts / L_d`,
`logCPM = log2(CPM + 1)`. Each contrast yields case-minus-each-reference and
case-minus-mean(reference logCPM), weighting normalized donors equally. Neither
pooled reference libraries nor log(mean CPM) defines that mean. All declared
references must be eligible; none is silently dropped. Each gene needs **10 raw
counts** across donors used for that row, otherwise unavailable/low_counts.
Missing populations are unavailable, never substituted with zero.

`molecular_contrasts.csv` records gene IDs, contrast/donor aliases, raw counts,
libraries, cells, reference numbers, normalized values/differences and status/reason.
Reference-valued columns are JSON lists in alias order; null means absent.
`reference_logcpm` is the individual or equal-donor mean. Unavailable normalized
values/differences are blank. `matched_groups.csv` preserves original group index,
donor/section/region/annotation, exclusions and pooled coverage/eligibility.
Selected sections may belong to an ineligible population. `summary.json` and
`report.md` use aliases. Private `provenance/run.json` records source correspondence,
config, code hashes, environment/job, input pins checked before/after execution,
and all other output hashes/bytes. It does not hash itself. Upstream raw-input,
code/config evidence is labelled `inherited_not_reverified`; it is not reopened.

Limits fail rather than truncate: 128 donors, 64 contrasts, 16 references per
contrast, 60,000 genes, 100,000 groups, 4 million sparse entries, 250,000 contrast
rows, 250,000 coverage rows, 2 million dense population elements per contrast.
NPZ compressed bytes: 64 MiB; expanded arrays: 96 MiB; input CSV: 32 MiB; config:
1 MiB; source record: 8 MiB. ZIP/NPY headers, member sets, dtypes, dimensions,
pointers and duplicate coordinates are checked before population densification.
Counts must be finite nonnegative integers, total <= `2**53 - 1`. No full matrix
is densified; output hashes stream in bounded memory.

These are **specimen-level relative-transcript-abundance contrasts**. One affected
donor cannot support replicated FTD associations. Inherited cell annotations are
not independent validation. Relative CPM can reflect compositional closure.
Assays remain separate; cells/sections do not add independent disease donors.
No p-values, CIs, AUROC, DE significance, ranked biomarkers or inferential error
bars are produced. Actual study execution remains unverified.

Controller-owned synthetic acceptance (no nested compute reservation):

```sh
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_molecular.py -q
```
