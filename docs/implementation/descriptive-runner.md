# Private descriptive H5AD runner

This runner prepares descriptive artifacts from explicitly configured inputs.
Synthetic checks establish pipeline behavior, not study findings. Assays
remain separate and unpaired. Cells and sections do not provide independent
disease-donor replication. There is no training, checkpoint loading/download,
prediction, disease significance testing or gene-program selection.

Unverified count provenance, donor/physical-section mapping, input paths,
coordinate units and the approved environment/job resources are launch blockers.
This document establishes no study result or job approval.

Use an existing environment with NumPy, SciPy, pandas, h5py and anndata. Model
backends are not imported; their installation and versions are not required.
Study inputs and **all outputs**, including aggregates, mappings, logs and run
records, stay in approved private PMACS storage. This command grants no transfer
or sharing permission. The researcher runs study commands through an approved LSF job;
the study mode requires a positive `LSB_JOBID` and `--execution-context pmacs`.

```sh
PYTHONPATH=src python -m ftd_prediction.descriptive \
  --config /PRIVATE/descriptive-rna.json --output /PRIVATE/outputs/NEW_RUN \
  --mode study --execution-context pmacs
```

For invented fixtures only, use `--mode synthetic --execution-context local` and
`"data_kind": "synthetic"`. The callable is
`ftd_prediction.descriptive.execute(config_path, output, mode='study', execution_context='pmacs')`.
The CLI emits only generic completion/failure text, never paths, rows or IDs.
Validation failures before output creation cannot have a private run log.

The JSON must have integer `schema_version: 1`, matching `data_kind`, explicit
`assay: "rna" | "spatial"`, `input_file`, `count_source`, verified
`counts_provenance`, `mapping_verified: true`, `mapping_evidence_ref` and
`obs_columns`. This example deliberately starts with `mapping_verified: false`
and unresolved placeholders; it must fail validation. The input owner must verify
the records, replace every placeholder and then explicitly set the mapping flag
to true before study execution:

```json
{
  "schema_version": 1,
  "data_kind": "study",
  "assay": "rna",
  "input_file": "/REPLACE_WITH_APPROVED_CANONICAL/REPLACE_WITH_COUNTS.h5ad",
  "canonical_input_roots": ["/REPLACE_WITH_APPROVED_CANONICAL"],
  "count_source": "layers/REPLACE_WITH_VERIFIED_COUNTS",
  "counts_provenance": {
    "kind": "REPLACE_WITH_VERIFIED_COUNT_KIND",
    "evidence_ref": "REPLACE_WITH_VERIFIED_PREPROCESSING_RECORD"
  },
  "mapping_verified": false,
  "mapping_evidence_ref": "REPLACE_WITH_VERIFIED_MAPPING_RECORD",
  "obs_columns": {"donor_id": "REPLACE_WITH_VERIFIED_DONOR_COLUMN", "cell_type": "REPLACE_WITH_ANNOTATION_COLUMN"},
  "seed": 11
}
```

Counts must be finite, nonnegative, unnormalized integers with nonzero cell
libraries. Count sources are `X`, `raw/X` or exact `layers/name`;
`background_corrected_umi` is also an accepted provenance kind. Evidence
declarations require actual verification by the input owner. No donor, section,
diagnosis or biological label is inferred. Missing donor/section mapping fails.
Required donor and spatial physical-section identities reject true nulls and
case/whitespace variants of empty, unknown, unresolved, null, none, nan, NA,
unavailable, `__unavailable__`, conflicting, not_applicable and restricted. This
validation applies with or without a graph and after an explicit crosswalk join;
no replacement identity is inferred and accepted identity strings are preserved.

For Xenium, use `counts_provenance.kind: measured_transcripts` with the actual
transcript-quality filtering and cell-assignment record. This provenance kind is
accepted only for the spatial assay. Xenium decoded transcripts are not
sequencing UMIs; the integer-count numerical checks remain the same.

Optional loading/provenance keys are `mapping_file` (absolute CSV), `mapping_keys`,
`coordinate_units`, `spatial_key`, `seed`, `cell_cap_per_donor`, `max_source_cells`,
`max_cells`, `max_genes`, `max_matrix_bytes`, `shared_preprocessing_scope`,
`source_version`, `panel_ref`, `segmentation_ref`, `coordinate_transform_ref`.
`obs_columns` maps the loader's explicit `donor_id`, `sample_id`, `section_id`,
`retinal_region`, `acquisition_region`, `cell_type`, `layer`, `tissue_component`
fields. A crosswalk uses selected sample/region keys to supply donor, physical
section, retinal region and/or tissue-component fields, without replacing existing
fields. A region supplied by the crosswalk must not also be declared in
`obs_columns`; use an unambiguous verified sample/acquisition key and never infer
region from a filename. The loader enforces its
shape/memory limits and records any deterministic RNA per-donor cell cap (default
seed 11). Denominators describe **loaded cells**, including any declared sampling.
Spatial cell thinning is rejected.

Study inputs and crosswalks must resolve inside explicit absolute canonical
roots. The output must be new and outside those roots and input/config paths.
Its directory mode is 0700; keep its parent storage private as well. Input,
crosswalk and config hashes are checked before/after loading and after analysis.
Unknown keys are rejected, including `model`, `split_strategy`, `fold_indices`
and any training/split/fold settings from neural configs.

For spatial input, require verified physical `section_id`, `coordinate_units:
"um"` and an explicit `spatial_key` selecting numeric cells-by-two coordinates.
Only an optional `spatial_graph` object with **both** positive finite numeric
`radius_um` and positive integer `max_neighbors` requests a graph; these are
explicit analysis choices, with no radius or neighbor-cap defaults. The optional
positive integer `max_descriptor_bytes` graph key defaults to
**134217728 bytes** (128 MiB).
Before graph construction or `neighborhood_fractions`, the runner rejects
`N * K * 16 > max_descriptor_bytes`, accounting for both the int64 counts and
float64 fractions matrices. `N` is the loaded cell count; `K` includes each
observed cell-type category and one missing category when present. With no
annotations, `K` is one. This budget bounds **only those two dense descriptor
matrices**, not total RSS, expression, edges, metadata, temporary arrays or file
compression. The effective bound, dimensions and matrix byte estimate are saved
under `descriptor_memory`. Expression and pseudobulk remain sparse. No other
graph keys or tissue masks are supported. Edges are directed and capped per source, within
`(donor_id, section_id)` and, when supplied, `tissue_component`. Stable cell IDs
break distance ties. Without masks these establish section/component locality
only: edges can cross tissue gaps or holes. No tissue-area density or
anatomy-adjusted claim is supported. Absent `spatial_graph` records
`not_requested` and creates no graph files.

Outputs:

- `pseudobulk_counts.npz` is sparse groups-by-genes counts, aligned with
  `pseudobulk_groups.csv` and `pseudobulk_genes.csv`. Donor, available section,
  retinal region and cell type remain separate grouping dimensions. There is no
  string sentinel: null and literal `__unavailable__` groups have separate sums.
  Each grouping field has a `<field>_missing` flag; cell type uses
  `annotation_missing`. Missing values are empty CSV fields with true flags.
- `composition.csv` counts cell types within donor and available section/retinal
  region fields. Fractions divide by **all loaded cells in each group**, including
  missing annotations. Empty cell-type fields plus `annotation_missing=true`
  denote missingness, not an invented biological category. Pseudobulk, composition
  and neighborhood annotations use **one policy**: true nulls and the existing
  spatial utility's missing markers (empty, unknown, unresolved, conflicting,
  none, null, not_applicable, restricted; case insensitive, surrounding whitespace
  ignored for marker detection) form one explicit missing group. The same policy
  applies to optional section/region grouping fields. All other strings, including
  literal `__unavailable__`, retain their exact values and separate groups.
  Read CSVs with `keep_default_na=False` and the missing flags to avoid a reader
  interpreting literal strings as nulls. `private_cell_mapping.csv` preserves
  original metadata and ordered cell identities. Without a cell-type column composition is
  `unavailable` and no composition CSV is written.
- Requested graphs produce `spatial_edges.npz` (`edges`, shape 2-by-E) and
  `neighborhood_fractions.npz` (`counts`, `fractions`, `degree`, `is_isolate`,
  `unknown_label_count`, `annotation_missing`). All row/edge indices refer to
  zero-based ordered mapping rows. `neighborhood_fractions.json` gives category
  order (JSON null means missing), settings and limitations. Fractions include
  missing neighbors in the denominator; isolate rows have explicit flags and
  zero placeholders. Without annotations all labels are missing, not inferred.
- `summary.json`, `private_execution.log` and `provenance/run.json` retain private
  details. The run record contains status/error type, config/input/crosswalk
  hashes, code/environment, seed, LSF identity, selection and artifact hashes
  (excluding the run record itself). Failures retain partial artifact hashes;
  incomplete outputs must not be treated as completed evidence.

Focused invented-fixture check: `PYTHONPATH=src python -m unittest discover -s tests -p test_descriptive.py -q`.
