# Public Novae human-0 expression compatibility

2026-11-11: the default loader and all scratch, export and reload paths use
`strict=True`. For the single pinned public checkpoint below, explicitly add this
to the spatial **model configuration**, alongside the existing donor split and
graph settings:

```json
{
  "mode": "frozen",
  "checkpoint_path": "/approved/local/path/to/pinned/novae-human-0",
  "checkpoint_compatibility": "novae-human-0-expression-v1"
}
```

Omission means `"checkpoint_compatibility": "strict"`; unknown modes fail.
Compatibility is also available with the existing explicit `"fine_tune"` mode;
it does not authorize fitting, change scientific choices, or apply to scratch
models. It requires installed Novae **1.1.3**, the inspected constructor contract,
and the complete expected tensor schema. Only local files are read; no download,
package patch or installation occurs.

The identity is `prism-oncology/novae-human-0`, revision
`b8c0a5d7612bac6bc719ab57ed3cd16ad814728c`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| config.json | 1025622 | `71070bb73a5f252ad2394a1b7132ff92b2a3833e6f4420640aa8a48ffc39a62f` |
| model.safetensors | 128175588 | `7d1e4a051469d14e796994faa34a41556183e78757f5dc70a99dbd9f6a19ce87` |

Both files are authenticated before construction/loading. All 78 original tensor
names, shapes and dtypes must match; their values pass through unchanged and are
checked byte for byte after loading. Exactly these six absent float32 tensors
are initialized to zero before a complete `load_state_dict(..., strict=True)`:

| Prefix `encoder.mlp_fusion.` | Shape |
| --- | --- |
| 0.weight / 0.bias | (114, 114) / (114,) |
| 2.weight / 2.bias | (64, 114) / (64,) |
| 4.weight / 4.bias | (64, 64) / (64,) |

The task's upstream evidence is official Novae commit
`8f9374ad9fdb708babd642843312cb876957cc36`, `tests/test_model.py:270–285`
and `novae/module/encode.py:44–50`. The installed 1.1.3 encoder's forward method
uses fusion only when `histo_embeddings` is present. Compatibility rejects that
official H&E key in input `AnnData.obsm` and directly at the loaded encoder.
Active missing/extra/misshaped/mistyped tensors, a different file identity,
runtime version or constructor/state schema all fail. There is no permissive
fallback.

Run `summary.json` and `provenance.json` both record the compatibility receipt:
original identities, runtime/schema, the six synthesized keys/shapes/dtypes,
zero initialization, active-byte verification, rationale and helper source hash.
Exports contain all 84 tensors and reload normally with strict loading. Their
expression-only limitation still applies; the Python input guard itself is not
serialized. Reusing an export uses the default strict setting, not this pinned
compatibility setting (its file identity has changed).

The controller's managed `test_novae_checkpoint.py` check sets
`FTD_PUBLIC_NOVAE_DIR` to the already-present public directory. With that variable
set, missing files/dependencies fail. Without it, only the public smoke skips;
no acquisition occurs. The smoke checks actual default strict rejection, opt-in
loading, original active bytes, frozen inference, H&E rejection and exact strict
export/reload weights and latents. It uses CPU with one thread, eight invented
cells, 32 checkpoint genes, two four-node graphs and at most one complete model
resident at once; it never densifies a full-vocabulary expression matrix. Its
`PUBLIC_NOVAE_CHECK` JSON line is execution evidence only when that check passes.

These are software compatibility checks, not study runs, measured retinal
performance, clinical validity or complete study readiness. No weights license
has been established. Public weights and temporary smoke exports are excluded
from the code snapshot/bundle; do not redistribute them through this change.
