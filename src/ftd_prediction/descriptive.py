"""Private descriptive H5AD execution; no training or disease inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

from .cli import protect_output_path, validate_runtime_scope
from .h5ad_io import allowed_count_kinds, load_h5ad
from .neural import _environment, _private_logging, _resolve
from .provenance import file_sha256, source_snapshot, utc_now, write_json
from .spatial import _labels, build_radius_graph, neighborhood_fractions


_ALLOWED = {
    'schema_version', 'data_kind', 'assay', 'input_file', 'canonical_input_roots',
    'count_source', 'counts_provenance', 'mapping_verified', 'mapping_evidence_ref',
    'obs_columns', 'coordinate_units', 'spatial_key', 'seed', 'cell_cap_per_donor',
    'max_source_cells', 'max_cells', 'max_genes', 'max_matrix_bytes',
    'shared_preprocessing_scope', 'source_version', 'panel_ref', 'segmentation_ref',
    'coordinate_transform_ref', 'mapping_file', 'mapping_keys', 'spatial_graph',
}
_DEFAULT_DESCRIPTOR_BYTES = 128 * 1024**2


def _validate(config, mode, execution_context):
    if (not isinstance(config, dict) or type(config.get('schema_version')) is not int
            or config['schema_version'] != 1 or set(config) - _ALLOWED):
        raise ValueError('Unsupported descriptive configuration schema')
    if mode not in ('study', 'synthetic') or execution_context not in ('pmacs', 'local'):
        raise ValueError('Unsupported execution scope')
    validate_runtime_scope(config, mode=mode, execution_context=execution_context)
    if config.get('assay') not in ('rna', 'spatial'):
        raise ValueError('An explicit assay is required')
    provenance = config.get('counts_provenance')
    if (not isinstance(provenance, dict)
            or provenance.get('kind') not in allowed_count_kinds(config.get('assay'))
            or not isinstance(provenance.get('evidence_ref'), str)
            or not provenance['evidence_ref'].strip()):
        raise ValueError('Verified count provenance is required')
    if (config.get('mapping_verified') is not True
            or not isinstance(config.get('mapping_evidence_ref'), str)
            or not config['mapping_evidence_ref'].strip()):
        raise ValueError('Verified donor/section mapping evidence is required')
    source, columns = config.get('count_source'), config.get('obs_columns')
    if not isinstance(source, str) or not (source in ('X', 'raw/X') or (
            source.startswith('layers/') and source.count('/') == 1 and source[7:])):
        raise ValueError('An explicit count source is required')
    if not isinstance(columns, dict) or ('donor_id' not in columns and not config.get('mapping_file')):
        raise ValueError('An explicit donor mapping is required')
    if config['assay'] == 'spatial' and (config.get('coordinate_units') != 'um'
            or ('section_id' not in columns and not config.get('mapping_file'))):
        raise ValueError('Spatial input requires explicit sections and micron coordinates')
    if 'spatial_graph' in config:
        graph = config['spatial_graph']
        if (config['assay'] != 'spatial' or not isinstance(graph, dict)
                or not {'radius_um', 'max_neighbors'}.issubset(graph)
                or set(graph) - {'radius_um', 'max_neighbors', 'max_descriptor_bytes'}):
            raise ValueError('Spatial graphs require radius, neighbor cap and optional descriptor budget')
        radius, cap = graph['radius_um'], graph['max_neighbors']
        budget = graph.get('max_descriptor_bytes', _DEFAULT_DESCRIPTOR_BYTES)
        if (type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0
                or type(cap) is not int or cap <= 0
                or type(budget) is not int or budget <= 0):
            raise ValueError('Invalid spatial graph settings')


def _metadata_values(values):
    """Use spatial missing markers and true nulls, without a string sentinel."""
    import pandas as pd
    return _labels([None if pd.isna(value) else value for value in values])


def _annotations(adata):
    if 'cell_type' not in adata.obs:
        return None
    return _metadata_values(adata.obs['cell_type'])


def _group_frame(adata):
    import pandas as pd
    keys = ['donor_id'] + [k for k in ('section_id', 'retinal_region', 'cell_type') if k in adata.obs]
    return pd.DataFrame({key: _metadata_values(adata.obs[key]) for key in keys}, index=adata.obs_names)


def _missing_flags(frame, keys):
    for key in keys:
        flag = 'annotation_missing' if key == 'cell_type' else key + '_missing'
        frame[flag] = frame[key].isna()


def _describe(adata, output):
    """Keep sparse sums and null grouping distinct from every literal category."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    frame = _group_frame(adata)
    keys = list(frame.columns)
    grouped = frame.groupby(keys, dropna=False, observed=True, sort=True)
    codes = grouped.ngroup().to_numpy(dtype=np.int64)
    groups = grouped.size().rename('n_cells').reset_index()
    aggregator = sparse.csr_matrix((np.ones(adata.n_obs, dtype=np.int64),
        (codes, np.arange(adata.n_obs))), shape=(len(groups), adata.n_obs))
    sparse.save_npz(output / 'pseudobulk_counts.npz', (aggregator @ adata.layers['counts']).tocsr())
    _missing_flags(groups, keys)
    groups.to_csv(output / 'pseudobulk_groups.csv', index=False)
    pd.DataFrame({'gene_id': adata.var_names}).to_csv(output / 'pseudobulk_genes.csv', index=False)
    identity = adata.obs.copy()
    identity.insert(0, 'cell_id', adata.obs_names)
    if 'spatial' in adata.obsm:
        identity['x_um'] = adata.obsm['spatial'][:, 0]
        identity['y_um'] = adata.obsm['spatial'][:, 1]
    identity.to_csv(output / 'private_cell_mapping.csv', index=False)
    return {'n_cells': int(adata.n_obs), 'n_genes': int(adata.n_vars),
        'n_donors': int(adata.obs['donor_id'].nunique()),
        'n_groups': len(groups), 'grouping_columns': keys,
        'cell_type_annotation_available': 'cell_type' in adata.obs,
        'missing_value_policy': 'nulls_and_spatial_missing_markers; explicit_flags; no_string_sentinel',
        'disease_effect_inference': False}


def _composition(adata, output):
    labels = _annotations(adata)
    if labels is None:
        return {'status': 'unavailable', 'reason': 'cell_type column absent'}
    frame = _group_frame(adata)
    keys = [k for k in frame if k != 'cell_type']
    counts = frame.groupby(keys + ['cell_type'], dropna=False, observed=True).size()
    counts = counts.rename('n_cells').reset_index()
    counts['group_cells'] = counts.groupby(keys, dropna=False, observed=True)['n_cells'].transform('sum')
    counts['fraction'] = counts['n_cells'] / counts['group_cells']
    _missing_flags(counts, keys + ['cell_type'])
    counts.to_csv(output / 'composition.csv', index=False)
    return {'status': 'completed', 'grouping_columns': keys,
            'denominator': 'all_loaded_cells_in_group', 'missing_annotation_cells': labels.count(None)}


def _graph(adata, settings, output):
    import numpy as np
    labels = _annotations(adata)
    labels = labels if labels is not None else [None] * adata.n_obs
    budget = settings.get('max_descriptor_bytes', _DEFAULT_DESCRIPTOR_BYTES)
    if type(budget) is not int or budget <= 0:
        raise ValueError('Invalid descriptor memory budget')
    # Check before either dense N x K matrix (or graph construction) is allocated.
    n_categories = len(set(labels))
    matrix_bytes = int(adata.n_obs) * n_categories * (np.dtype('int64').itemsize + np.dtype('float64').itemsize)
    if matrix_bytes > budget:
        raise ValueError('Neighborhood descriptor matrices exceed max_descriptor_bytes')
    obs = adata.obs
    edges = build_radius_graph(adata.obsm['spatial'],
        list(zip(obs['donor_id'], obs['section_id'])),
        radius_um=settings['radius_um'], max_neighbors=settings['max_neighbors'],
        cell_ids=adata.obs_names.tolist(),
        tissue_components=obs['tissue_component'].tolist() if 'tissue_component' in obs else None)
    np.savez_compressed(output / 'spatial_edges.npz', edges=edges)
    result = neighborhood_fractions(edges, labels)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    arrays['annotation_missing'] = np.asarray([label is None for label in labels])
    np.savez_compressed(output / 'neighborhood_fractions.npz', **arrays)
    description = {'status': 'completed', 'settings': settings, 'coordinate_units': 'um',
        'descriptor_memory': {'max_descriptor_bytes': budget, 'matrix_bytes': matrix_bytes,
            'n_cells': int(adata.n_obs), 'n_categories': n_categories,
            'scope': 'int64_counts_and_float64_fractions_only; not_total_RSS'},
        'boundary_columns': ['donor_id', 'section_id'] + (['tissue_component'] if 'tissue_component' in obs else []),
        'tissue_masks_available': False,
        'categories': result['categories'], 'n_unknown_cells': result['n_unknown_cells'],
        'cell_type_annotation_available': 'cell_type' in obs,
        'n_edges': int(edges.shape[1]), 'directed': True,
        'mapping': 'private_cell_mapping.csv; zero-based row order for both NPZ files',
        'fraction_denominator': 'all_outgoing_neighbors_including_missing_labels',
        'limitations': ['Section/component locality only; tissue masks are absent.',
                        'No tissue-area density or anatomy-adjusted interpretation.',
                        'Isolate fractions are zero placeholders, not measured composition.']}
    write_json(output / 'neighborhood_fractions.json', description)
    return description


def execute(config_path, output, mode='study', execution_context='pmacs'):
    """Write a fresh private run; reject unsafe scope before reading assay data."""
    config_path = Path(config_path).resolve()
    snapshot = config_path.read_bytes()
    config = json.loads(snapshot)
    _validate(config, mode, execution_context)
    roots = config.get('canonical_input_roots', [])
    if (not isinstance(roots, list)
            or any(not isinstance(r, str) or not Path(r).is_absolute() for r in roots)
            or (mode == 'study' and not roots)):
        raise ValueError('Explicit absolute canonical input roots are required')
    source = _resolve(config_path.parent, config.get('input_file'))
    inputs, loader_config = [source], dict(config)
    if config.get('mapping_file') is not None:
        mapping = config['mapping_file']
        if not isinstance(mapping, str) or not Path(mapping).is_absolute():
            raise ValueError('Mapping file requires an explicit absolute path')
        mapping = Path(mapping).resolve()
        inputs.append(mapping)
        loader_config['mapping_file'] = str(mapping)
    if mode == 'study' and any(not any(p.is_relative_to(Path(r).resolve()) for r in roots) for p in inputs):
        raise ValueError('Input resolves outside canonical input roots')
    destination = protect_output_path(output, roots)
    if any(destination == p or p in destination.parents for p in inputs + [config_path]):
        raise ValueError('Output overlaps an input')
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    (destination / 'provenance').mkdir(mode=0o700)
    record_path = destination / 'provenance' / 'run.json'
    record = {'schema_version': 1, 'status': 'running', 'started_at': utc_now(),
        'data_kind': mode, 'execution_context': execution_context,
        'lsf_job_id': os.environ.get('LSB_JOBID'), 'configuration': config,
        'configuration_sha256': hashlib.sha256(snapshot).hexdigest(),
        'configuration_path': str(config_path), 'seed': config.get('seed', 11),
        'source_file': str(source), 'output_directory': str(destination),
        'disease_inference_enabled': False, 'training_enabled': False}
    write_json(record_path, record)
    try:
        with _private_logging(destination / 'private_execution.log'):
            record.update(code=source_snapshot(), environment=_environment())
            hashes = {str(p): file_sha256(p) for p in inputs}
            record['input_sha256'] = hashes[str(source)]
            record['input_files'] = [{'path': p, 'sha256': h} for p, h in hashes.items()]

            def unchanged():
                if (config_path.read_bytes() != snapshot
                        or any(file_sha256(p) != h for p, h in hashes.items())):
                    raise ValueError('Input, crosswalk or configuration changed during execution')

            unchanged()
            adata = load_h5ad(source, loader_config)
            unchanged()
            record['input_selection'] = adata.uns['ftd_input']
            record['descriptive'] = _describe(adata, destination)
            record['composition'] = _composition(adata, destination)
            record['spatial_graph'] = (_graph(adata, config['spatial_graph'], destination)
                if 'spatial_graph' in config else {'status': 'not_requested'})
            unchanged()
            write_json(destination / 'summary.json', {
                'data_kind': mode, 'evidence_scope': 'synthetic_pipeline_check' if mode == 'synthetic' else 'descriptive_only',
                **{k: record[k] for k in ('descriptive', 'composition', 'spatial_graph')},
                'disease_inference_enabled': False, 'training_enabled': False})
        if any(path.is_symlink() for path in destination.rglob('*')):
            raise ValueError('Unexpected artifact symlink')
        record['status'] = 'completed'
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__)
        with (destination / 'private_execution.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    finally:
        record['finished_at'] = utc_now()
        record['artifacts'] = []
        for path in sorted(destination.rglob('*')):
            if not path.is_symlink() and path.is_file() and path != record_path:
                record['artifacts'].append({'path': path.relative_to(destination).as_posix(),
                    'sha256': file_sha256(path), 'bytes': path.stat().st_size})
        write_json(record_path, record, exclusive=False)
    return destination


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command arguments')


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('study', 'synthetic'), default='study')
    parser.add_argument('--execution-context', choices=('pmacs', 'local'), default='pmacs')
    try:
        args = parser.parse_args(argv)
        execute(args.config, args.output, args.mode, args.execution_context)
    except Exception:
        print('Descriptive run failed. Inspect the private log if a run directory was created.')
        return 1
    print('Descriptive run completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
