# Private descriptive figures

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python -m ftd_prediction.descriptive_report \
  --run /PRIVATE/COMPLETED_RUN --output /PRIVATE/NEW_REPORT
```

Output parent must exist. Study requires PMACS metadata, positive original job
ID and actual positive `LSB_JOBID` (environment guard, not authentication).
Synthetic may run locally. Study outputs remain private on PMACS; no sharing.

Completed schema-1 provenance must disable training/disease inference. All
inventory hashes/sizes precede interpretation. Unsafe/nonregular paths, symlinks,
unlisted files and inconsistent metadata fail. CSV/NPZ reconcile with ordered
mapping and graph boundaries; NPZ uses `allow_pickle=False`. Reader budgets: 256 MiB interpreted file,
32 MiB run JSON, 512 MiB uncompressed NPZ. Hashes cannot prove scientific truth. H5AD, original counts, crosswalk/config
files and models never reopen; pseudobulk is only hashed.

Offline HTML embeds PNG; Markdown, PNG, editable PDF, private error log/cache
and provenance are written under umask 077 with exclusive creation. Reuse/run
overlap fails. Generic CLI errors retain private logs/partial output without
completed provenance; early path failures have no log. Retry in a new directory.

Composition sums unequal section counts before recomputing donor/region
fractions, including missing mass; never averages fractions or sums repeated
group_cells. Spatial sections have separate equal-scale um maps.
Physical-section panels and aliases apply only to spatial inputs. Missing optional
RNA section labels do not exclude cells from donor/region composition.
Heatmaps show
**mean outgoing-neighbor fraction among non-isolated cells**, equal focal-cell
weight, missing neighbors retained. Degree/isolate summaries include all cells.
Unavailable/unrequested panels are explicit. No fitting, molecular analysis, disease inference or density.

PDF: editable TrueType/fonttype 42, Arial else explicit DejaVu Sans. HTML is
escaped/offline. Display defaults: 5,000 cells/map, 24 panels/categories, 30 groups/
table rows. CLI overrides have maxima 50,000/100/64/100; minimum panels/categories
2/3. Ordered-row sampling/category combinations affect display only. Provenance
retains full analyses, aliases, omissions/selections, source/config/code/helper/
output hashes (except itself); sources are rehashed before completion.

Synthetic checks cover actual figure files/text, pooling/missing/isolate
arithmetic, sections/aspect, display limits, permissions, escaping and integrity/
schema/object-array/path/mode failures:

```sh
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_descriptive_report.py -q
```

For retained visual fixtures, use `PYTHONPATH=src:tests`, call `fixture` from
`test_descriptive_report` in scratch, then `build_report` into a new sibling.
Tests normally delete fixtures. Lead visual review, real LSF execution and
study validation remain necessary; no study finding is established.
