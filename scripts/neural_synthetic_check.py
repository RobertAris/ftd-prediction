#!/usr/bin/env python3
"""Exercise H5AD-to-model execution with invented data and tiny CPU models only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from ftd_prediction.neural import execute

    rng = np.random.default_rng(11)
    grid = np.array([(x, y) for x in range(8) for y in range(8)], dtype=float) * 10
    xy = np.concatenate([grid, grid, grid]) + rng.normal(0, .05, (192, 2))
    rates = np.ones((192, 32))
    rates[xy[:, 0] < 40, :8] = 6
    rates[xy[:, 0] >= 40, 8:16] = 6
    counts = csr_matrix(rng.poisson(rates).astype('float32'))
    a = ad.AnnData(X=counts.copy(), obs=pd.DataFrame({
        'donor': ['invented_train'] * 128 + ['invented_query'] * 64,
        'section': ['invented_A'] * 64 + ['invented_B'] * 64 + ['invented_C'] * 64,
        'type': np.where(xy[:, 0] < 40, 'invented_type_A', 'invented_type_B'),
    }, index=[f'invented_cell_{i}' for i in range(192)]),
        var=pd.DataFrame(index=[f'invented_gene_{i}' for i in range(32)]))
    a.layers['counts'] = counts
    a.obsm['spatial'] = xy
    source = root / 'invented.h5ad'
    a.write_h5ad(source)
    base = dict(schema_version=1, data_kind='synthetic', input_file=str(source),
                canonical_input_roots=[], count_source='layers/counts',
                counts_provenance=dict(kind='measured_umi', evidence_ref='invented-poisson-generator-seed11'),
                mapping_verified=True, mapping_evidence_ref='invented-fixture',
                obs_columns=dict(donor_id='donor', cell_type='type'),
                shared_preprocessing_scope='synthetic_generator', split_strategy='explicit')
    model = dict(train_donors=['invented_train'], test_donors=['invented_query'],
                 seed=11, max_epochs=2, batch_size=16, accelerator='cpu', max_minutes=2)
    rna = {**base, 'assay': 'rna', 'model': {**model, 'max_genes': 16,
        'n_hidden': 16, 'n_latent': 4, 'n_layers': 1, 'num_threads': 1, 'kl_warmup_epochs': 1}}
    spatial = {**base, 'assay': 'spatial', 'coordinate_units': 'um', 'spatial_key': 'spatial',
        'obs_columns': {**base['obs_columns'], 'section_id': 'section'},
        'model': {**model, 'mode': 'synthetic_smoke', 'num_workers': 0,
                  'radius_um': 18.0, 'domain_level': 2}}
    for assay, config in [('rna', rna), ('spatial', spatial)]:
        config_path = root / (assay + '.json')
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        result = execute(config_path, root / (assay + '_run'), mode='synthetic', execution_context='local')
        print(json.dumps(dict(assay=assay, status='completed', output=str(result),
                             evidence_scope='invented_data_software_check')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
