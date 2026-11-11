"""Private, provenance-recorded neural execution on explicitly mapped H5AD input."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import sys
import traceback

from .cli import protect_output_path, validate_runtime_scope
from .h5ad_io import load_h5ad
from .provenance import environment_record, file_sha256, source_snapshot, utc_now, write_json


def _environment():
    result = environment_record()
    result['packages'] = {}
    for name in ('numpy', 'scipy', 'pandas', 'h5py', 'anndata', 'scanpy', 'scikit-learn',
                 'scvi-tools', 'novae', 'torch', 'torch-geometric', 'lightning'):
        try:
            result['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result['packages'][name] = None
    return result


def _resolve(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError('An input path is required')
    p = Path(value)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _validate_environment(assay):
    """Fail before reading study inputs when the tested model API is absent."""
    expected = {'scvi-tools': '1.5.2'} if assay == 'rna' else {'novae': '1.1.3'}
    for package, version in expected.items():
        if metadata.version(package) != version:
            raise ValueError(f'This backend requires {package}=={version}')
    for package in ('anndata', 'h5py', 'scipy', 'numpy', 'pandas', 'torch'):
        metadata.version(package)


@contextlib.contextmanager
def _private_logging(path):
    """Capture console handlers and release references before closing the log."""
    def handlers():
        loggers = [logging.getLogger()] + [v for v in logging.Logger.manager.loggerDict.values()
                                           if isinstance(v, logging.Logger)]
        return {h for logger in loggers for h in logger.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)}

    original_stderr = sys.stderr
    streams = {h: h.stream for h in handlers()}
    with path.open('x') as log:
        own = logging.StreamHandler(log)
        logging.getLogger().addHandler(own)
        for handler in streams:
            if getattr(handler.stream, 'closed', False):
                handler.stream = log
            else:
                handler.setStream(log)
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                yield
        finally:
            for handler in handlers() | set(streams):
                if handler is own:
                    continue
                if handler in streams:
                    target = streams[handler]
                    if getattr(target, 'closed', False):
                        target = original_stderr
                elif handler.stream is log:
                    target = original_stderr
                else:
                    continue
                if getattr(handler.stream, 'closed', False):
                    handler.stream = target
                else:
                    handler.setStream(target)
            logging.getLogger().removeHandler(own)
            own.close()


def _split_configs(adata, config):
    model = config.get('model')
    if not isinstance(model, dict):
        raise ValueError('An explicit model configuration is required')
    strategy = config.get('split_strategy', 'explicit')
    donors = sorted(adata.obs['donor_id'].unique().tolist())
    if strategy != 'leave_one_donor_out' and config.get('fold_indices') is not None:
        raise ValueError('Fold indices apply only to leave-one-donor-out execution')
    if strategy == 'explicit':
        return [dict(model)]
    if strategy not in ('leave_one_donor_out', 'all_donor_descriptive'):
        raise ValueError('Unsupported donor split strategy')
    if 'train_donors' in model or 'test_donors' in model:
        raise ValueError('Automatic and explicit split settings cannot be combined')
    if strategy == 'all_donor_descriptive':
        return [{**model, 'train_donors': donors, 'test_donors': []}]
    if len(donors) < 2:
        raise ValueError('A donor-transfer comparison requires at least two donors')
    folds = config.get('fold_indices', list(range(len(donors))))
    if (not isinstance(folds, list) or not folds or len(folds) != len(set(folds))
            or any(type(i) is not int or not 0 <= i < len(donors) for i in folds)):
        raise ValueError('Invalid prespecified donor fold indices')
    return [{**model, 'train_donors': [d for d in donors if d != donors[i]],
             'test_donors': [donors[i]]} for i in folds]


def _describe(adata, output):
    """Private sparse pseudobulk and composition, never disease significance."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    keys = ['donor_id'] + [c for c in ('section_id', 'retinal_region', 'cell_type') if c in adata.obs]
    frame = adata.obs[keys].astype(object).where(adata.obs[keys].notna(), '__unavailable__')
    tuples = pd.MultiIndex.from_frame(frame)
    codes, unique = pd.factorize(tuples, sort=True)
    aggregator = sparse.csr_matrix((np.ones(adata.n_obs), (codes, np.arange(adata.n_obs))),
                                   shape=(len(unique), adata.n_obs))
    pseudobulk = (aggregator @ adata.layers['counts']).tocsr()
    sparse.save_npz(output / 'pseudobulk_counts.npz', pseudobulk)
    groups = unique.to_frame(index=False)
    groups.columns = keys
    groups['n_cells'] = np.bincount(codes, minlength=len(unique))
    groups.to_csv(output / 'pseudobulk_groups.csv', index=False)
    pd.DataFrame({'gene_id': adata.var_names}).to_csv(output / 'pseudobulk_genes.csv', index=False)
    # Ordered identity/mapping/coordinate sidecar makes each model row traceable.
    identity = adata.obs.copy()
    identity.insert(0, 'cell_id', adata.obs_names)
    if 'spatial' in adata.obsm:
        identity['x_um'] = adata.obsm['spatial'][:, 0]
        identity['y_um'] = adata.obsm['spatial'][:, 1]
    identity.to_csv(output / 'private_cell_mapping.csv', index=False)
    return {'n_cells': int(adata.n_obs), 'n_genes': int(adata.n_vars),
            'n_donors': int(adata.obs['donor_id'].nunique()),
            'n_groups': len(unique), 'grouping_columns': keys,
            'cell_type_annotation_available': 'cell_type' in adata.obs,
            'disease_effect_inference': False}


def execute(config_path, output, *, mode='study', execution_context='pmacs', backend=None):
    """Execute declared folds; retain private failure records and immutable inputs."""
    config_path = Path(config_path).resolve()
    snapshot = config_path.read_bytes()
    config = json.loads(snapshot)
    allowed = {'schema_version', 'data_kind', 'assay', 'input_file', 'canonical_input_roots',
               'count_source', 'counts_provenance', 'mapping_verified', 'mapping_evidence_ref',
               'obs_columns', 'coordinate_units', 'spatial_key', 'seed', 'cell_cap_per_donor',
               'max_source_cells', 'max_cells', 'max_genes', 'max_matrix_bytes', 'model',
               'split_strategy', 'fold_indices', 'shared_preprocessing_scope', 'source_version',
               'panel_ref', 'segmentation_ref', 'coordinate_transform_ref', 'mapping_file', 'mapping_keys'}
    if (not isinstance(config, dict) or type(config.get('schema_version')) is not int
            or config['schema_version'] != 1 or set(config) - allowed):
        raise ValueError('Unsupported neural configuration schema')
    validate_runtime_scope(config, mode=mode, execution_context=execution_context)
    if config.get('assay') not in ('rna', 'spatial'):
        raise ValueError('Unsupported assay')
    if mode != 'synthetic' and config.get('model', {}).get('mode') == 'synthetic_smoke':
        raise ValueError('Synthetic model initialization cannot be used for study inputs')
    roots = config.get('canonical_input_roots', [])
    if not isinstance(roots, list) or any(not isinstance(v, str) or not Path(v).is_absolute() for v in roots):
        raise ValueError('Canonical roots must be explicit absolute paths')
    if mode != 'synthetic' and not roots:
        raise ValueError('Study inputs require canonical input roots')
    source = _resolve(config_path.parent, config.get('input_file'))
    inputs = [source]
    loader_config = dict(config)
    if config.get('mapping_file') is not None:
        mapping = Path(config['mapping_file'])
        if not mapping.is_absolute():
            raise ValueError('Mapping file requires an explicit absolute path')
        mapping = mapping.resolve()
        inputs.append(mapping)
        loader_config['mapping_file'] = str(mapping)
    if mode != 'synthetic' and any(not any(p.is_relative_to(Path(r).resolve()) for r in roots) for p in inputs):
        raise ValueError('Input resolves outside canonical input roots')
    destination = protect_output_path(output, roots)
    if any(destination == p or p in destination.parents for p in inputs + [config_path]):
        raise ValueError('Output overlaps an input')
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    record = {'schema_version': 1, 'status': 'running', 'started_at': utc_now(),
              'data_kind': mode, 'execution_context': execution_context,
              'lsf_job_id': os.environ.get('LSB_JOBID'), 'configuration': config,
              'code': source_snapshot(), 'environment': _environment(),
              'configuration_sha256': hashlib.sha256(snapshot).hexdigest(),
              'source_file': str(source), 'disease_inference_enabled': False,
              'shared_preprocessing_scope': config.get('shared_preprocessing_scope', 'unverified'),
              'output_directory': str(destination)}
    write_json(destination / 'run.json', record)
    try:
        if backend is None:
            _validate_environment(config['assay'])
        input_hashes = {str(p): file_sha256(p) for p in inputs}
        before = input_hashes[str(source)]
        record['input_sha256'] = before
        record['input_files'] = [{'path': path, 'sha256': digest} for path, digest in input_hashes.items()]
        if config_path.read_bytes() != snapshot:
            raise ValueError('Configuration changed after validation')
        with _private_logging(destination / 'private_execution.log'):
            adata = load_h5ad(source, loader_config)
            # Catch input mutation during loading before any model is fitted.
            if any(file_sha256(p) != digest for p, digest in input_hashes.items()):
                raise ValueError('Input changed during loading')
            record['input_selection'] = adata.uns['ftd_input']
            record['descriptive'] = _describe(adata, destination)
            splits = _split_configs(adata, config)
            write_json(destination / 'private_splits.json', splits)
            if backend is None:
                if config['assay'] == 'rna':
                    from .rna_training import run_rna
                    backend = run_rna
                else:
                    from .spatial_training import run_spatial
                    backend = run_spatial
            results = []
            for index, model_config in enumerate(splits):
                fold = destination / f'fold_{index:02d}'
                result = backend(adata, model_config, fold)
                if result.get('status') != 'completed':
                    raise RuntimeError('Backend did not complete successfully')
                results.append(result)
                write_json(destination / f'fold_{index:02d}_summary.json', result)
        if any(file_sha256(p) != digest for p, digest in input_hashes.items()) or config_path.read_bytes() != snapshot:
            raise ValueError('Input or configuration changed during analysis')
        write_json(destination / 'summary.json', {'data_kind': mode,
                   'evidence_scope': 'synthetic_pipeline_check' if mode == 'synthetic' else 'descriptive_and_technical_donor_transfer',
                   'descriptive': record['descriptive'], 'folds': results,
                   'disease_inference_enabled': False})
        record['artifacts'] = []
        for path in sorted(destination.rglob('*')):
            if path.is_symlink():
                raise ValueError('Unexpected artifact symlink')
            if path.is_file() and path.name != 'run.json':
                record['artifacts'].append({'path': path.relative_to(destination).as_posix(),
                                            'sha256': file_sha256(path), 'bytes': path.stat().st_size})
        record['status'] = 'completed'
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__)
        with (destination / 'private_execution.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    finally:
        record['finished_at'] = utc_now()
        write_json(destination / 'run.json', record, exclusive=False)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('study', 'reference', 'synthetic'), default='study')
    parser.add_argument('--execution-context', choices=('pmacs', 'local'), default='pmacs')
    args = parser.parse_args(argv)
    try:
        destination = execute(args.config, args.output, mode=args.mode,
                              execution_context=args.execution_context)
    except Exception as error:
        # Private logs/configuration stay on the cluster; stdout exposes no values.
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'message': 'Inspect the private run record and configured input contract.'}))
        return 1
    print(json.dumps({'status': 'completed', 'output': str(destination)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
