"""Build private offline technical reports; never load assays or model objects.

Study/reference runs require an active positive LSB_JOBID. Artifact bytes are
streamed only for integrity; only run/summary JSON is interpreted. Recorded
reload checks and input/source hashes are provenance, not independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

_HASH = re.compile(r'[0-9a-f]{64}\Z')
_PACKAGES = ('numpy', 'scipy', 'pandas', 'h5py', 'anndata', 'scanpy',
             'scikit-learn', 'scvi-tools', 'novae', 'torch', 'torch-geometric', 'lightning')
# Fixed paths exclude donor-keyed dictionaries, gene names, paths, and free text.
_DIAGNOSTICS = (
    'pca_train_log_expression_mse', 'pca_test_log_expression_mse',
    'optimization.training_steps', 'optimization.finite_loss_steps',
    'optimization.observed_epochs', 'optimization.fit_seconds',
    'optimization.inference_reload_max_abs_error',
    'optimization.full_checkpoint_reload_verified', 'optimization.exact_resume_verified',
    'scvi_embedding.effective_rank', 'scvi_embedding.mean_latent_variance', 'scvi_embedding.rank',
    'pca_embedding.effective_rank', 'pca_embedding.mean_latent_variance', 'pca_embedding.rank',
    'cross_component_edges', 'cross_donor_edges', 'cross_section_edges', 'directed_edges',
    'max_distance_error_um', 'reload_max_absolute_error', 'masks_checked', 'test_section_adaptation',
) + tuple(f'{group}.{key}' for group in ('train', 'test') for key in (
    'cells', 'collapsed', 'invalid_neighborhoods', 'valid_neighborhoods', 'latent_rank',
    'annotation_agreement.adjusted_rand_index', 'annotation_agreement.labelled_valid_cells',
)) + tuple(f'expression_baseline.annotation_agreement.{group}.{key}'
           for group in ('train', 'test') for key in ('accuracy', 'labelled_cells'))
_TRAINING_ANNOTATION = tuple(
    f'annotation_agreement.per_representation.{rep}.training_resubstitution.{key}'
    for rep in ('expression', 'pca', 'scvi')
    for key in ('balanced_accuracy', 'macro_f1', 'n_labelled'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _no_symlinks(path):
    """Check every existing component, including ancestors of the run root."""
    for part in (path, *path.parents):
        _require(not part.is_symlink(), 'Symlinks are forbidden')


def _inside(root, value):
    _require(isinstance(value, str) and value and '\\' not in value and '\x00' not in value,
             'Invalid artifact path')
    parts = value.split('/')
    _require(not PurePosixPath(value).is_absolute() and all(p not in ('', '.', '..') for p in parts)
             and ':' not in parts[0], 'Artifact paths must be contained relative paths')
    path = root.joinpath(*parts)
    _no_symlinks(path)
    _require(path.is_relative_to(root) and stat.S_ISREG(path.stat().st_mode),
             'Artifacts must be contained regular files')
    return path


def _bytes(path):
    _require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= 32 * 1024 * 1024,
             'Invalid or oversized summary JSON')
    return path.read_bytes()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    _require(isinstance(value, dict), 'JSON object required')
    return value


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return value if isinstance(value, str) and _HASH.fullmatch(value) else 'unavailable'


def _get(mapping, path):
    for key in path.split('.'):
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def _scalar(value):
    if type(value) is bool:
        return str(value).lower()
    if type(value) in (int, float) and math.isfinite(value):
        return str(value)
    return 'unavailable'


def _environment(record):
    rows = []
    env = record.get('environment', {})
    if not isinstance(env, dict):
        return [('environment', 'unavailable')]
    for key in ('python', 'platform', 'machine', 'numpy'):
        value = env.get(key)
        if key == 'python' and isinstance(value, str):
            match = re.match(r'\d+\.\d+\.\d+', value)
            value = match[0] if match else None
        if not isinstance(value, str) or len(value) > 200 or any(c in value for c in '/\\\n\r\t'):
            value = 'unavailable'
        rows.append((key, value))
    packages = env.get('packages', {})
    for name in _PACKAGES:
        value = packages.get(name) if isinstance(packages, dict) else None
        rows.append((name, value if isinstance(value, str) and re.fullmatch(r'[0-9A-Za-z.+_-]{1,80}', value)
                     else 'unavailable'))
    return rows


def _heldout_rows(fold):
    """Validate donor-keyed RNA metrics and expose only ordinal aliases."""
    annotation = _get(fold, 'metrics.annotation_agreement')
    _require(annotation is None or isinstance(annotation, dict), 'Invalid annotation diagnostics')
    representations = annotation.get('per_representation') if annotation is not None else None
    _require(representations is None or isinstance(representations, dict), 'Invalid annotation representations')
    representations = representations or {}
    _require(not set(representations) - {'expression', 'pca', 'scvi'}, 'Unknown annotation representation')
    rows = []
    for representation in ('expression', 'pca', 'scvi'):
        detail = representations.get(representation)
        _require(detail is None or isinstance(detail, dict), 'Invalid annotation representation')
        heldout = detail.get('heldout_per_donor') if detail is not None else None
        _require(heldout is None or isinstance(heldout, dict), 'Invalid held-out annotation diagnostics')
        heldout = heldout or {}
        _require(not set(heldout) - set(fold['test_donors']), 'Undeclared held-out annotation donor')
        for ordinal, donor in enumerate(fold['test_donors'], 1):
            values = heldout.get(donor)
            _require(values is None or isinstance(values, dict), 'Invalid held-out annotation metrics')
            values = values or {}
            for key in ('balanced_accuracy', 'macro_f1'):
                value = values.get(key)
                _require(value is None or (type(value) in (int, float) and math.isfinite(value)
                         and 0 <= value <= 1), 'Invalid held-out annotation score')
            count = values.get('n_labelled')
            _require(count is None or (type(count) is int and count >= 0), 'Invalid annotation count')
            _require(count != 0 or all(values.get(k) is None for k in ('balanced_accuracy', 'macro_f1')),
                     'Annotation scores require labelled cells')
            unsupported = values.get('unsupported_classes')
            _require(unsupported is None or (isinstance(unsupported, list)
                     and all(isinstance(item, str) for item in unsupported)), 'Invalid unsupported-class count')
            alias = f'{representation} / held-out donor {ordinal} / '
            rows += [(alias + key, _scalar(values.get(key)))
                     for key in ('balanced_accuracy', 'macro_f1', 'n_labelled')]
            rows.append((alias + 'unsupported class count',
                         str(len(unsupported)) if unsupported is not None else 'unavailable'))
    return rows or [('held-out annotation agreement', 'unavailable (no held-out donors declared)')]


def _load(root):
    root = Path(os.path.abspath(root))
    _no_symlinks(root)
    _require(root.is_dir(), 'Run directory required')
    raw = _bytes(_inside(root, 'run.json'))
    record = _json(raw)
    kind = record.get('data_kind')
    _require(kind in ('synthetic', 'study', 'reference'), 'Unsupported data kind')
    if kind != 'synthetic':
        job = os.environ.get('LSB_JOBID', '')
        _require(bool(re.fullmatch(r'[0-9]+', job)) and int(job) > 0,
                 'Study/reference reports require an active positive LSB_JOBID')
    _require(type(record.get('schema_version')) is int and record['schema_version'] == 1,
             'Unsupported run schema')
    status = record.get('status')
    _require(status in ('completed', 'failed', 'running'), 'Unsupported run status')
    result = {'root': root, 'record': record, 'run_sha256': hashlib.sha256(raw).hexdigest(), 'folds': []}
    if status != 'completed':
        return result  # Never open partial-run artifacts or expose partial metrics.
    _require(record.get('disease_inference_enabled') is False, 'Disease inference must be disabled')
    artifacts = record.get('artifacts')
    _require(isinstance(artifacts, list) and artifacts, 'Completed artifacts required')
    approved = {}
    seen = set()
    for artifact in artifacts:
        _require(isinstance(artifact, dict), 'Invalid artifact record')
        name = artifact.get('path')
        path = _inside(root, name)
        _require(name not in seen and name != 'run.json', 'Duplicate or invalid artifact entry')
        seen.add(name)
        expected = artifact.get('sha256')
        _require(_digest(expected) != 'unavailable' and type(artifact.get('bytes')) is int
                 and artifact['bytes'] >= 0, 'Invalid artifact integrity metadata')
        _require(path.stat().st_size == artifact['bytes'] and _hash(path) == expected,
                 'Artifact integrity verification failed')
        if name == 'summary.json' or re.fullmatch(r'fold_[0-9]{2,}(?:_summary\.json|/summary\.json)', name):
            payload = _bytes(path)
            _require(hashlib.sha256(payload).hexdigest() == expected, 'Summary changed during verification')
            approved[name] = _json(payload)
    _require('summary.json' in approved, 'Hashed summary required')
    summary = approved['summary.json']
    _require(summary.get('data_kind') == kind and summary.get('disease_inference_enabled') is False,
             'Run/summary scope mismatch')
    desc = record.get('descriptive')
    _require(isinstance(desc, dict) and desc == summary.get('descriptive')
             and desc.get('disease_effect_inference') is False, 'Descriptive summary mismatch')
    for key in ('n_cells', 'n_genes', 'n_donors'):
        _require(type(desc.get(key)) is int and desc[key] > 0, 'Invalid descriptive counts')
    config = record.get('configuration', {})
    _require(isinstance(config, dict) and config.get('data_kind') == kind, 'Configuration scope mismatch')
    assay = config.get('assay')
    _require(assay in ('rna', 'spatial'), 'Unsupported assay')
    for key in ('input_sha256', 'configuration_sha256'):
        _require(_digest(record.get(key)) != 'unavailable', 'Completed provenance digest required')
    _require(_digest(_get(record, 'code.source_sha256')) != 'unavailable', 'Source digest required')
    folds = summary.get('folds')
    _require(isinstance(folds, list) and folds, 'Completed folds required')
    expected_names = {f'fold_{i:02d}_summary.json' for i in range(len(folds))}
    _require({n for n in approved if re.fullmatch(r'fold_[0-9]{2,}_summary\.json', n)} == expected_names,
             'Fold summary inventory mismatch')
    for i, fold in enumerate(folds):
        _require(isinstance(fold, dict) and fold.get('status') == 'completed'
                 and fold.get('method') == {'rna': 'scvi', 'spatial': 'novae'}[assay], 'Invalid fold status/method')
        _require(approved[f'fold_{i:02d}_summary.json'] == fold, 'Fold summary mismatch')
        backend_summary = approved.get(f'fold_{i:02d}/summary.json')
        _require(backend_summary is None or backend_summary == fold, 'Backend summary mismatch')
        train, test = fold.get('train_donors'), fold.get('test_donors')
        _require(isinstance(train, list) and isinstance(test, list)
                 and all(isinstance(d, str) and d for d in train + test), 'Invalid donor count provenance')
        _require(len(set(train + test)) == len(train + test) <= desc['n_donors'], 'Fold donor overlap or count mismatch')
        if fold['method'] == 'scvi':
            _heldout_rows(fold)
    _require(_hash(root / 'run.json') == result['run_sha256'], 'Run record changed during verification')
    result['folds'] = folds
    return result


def _tables(runs):
    tables = []
    for index, run in enumerate(runs, 1):
        record = run['record']
        status = record['status']
        provenance = [('source run status', status), ('run.json SHA256', run['run_sha256']),
                      ('source digest (recorded)', _digest(_get(record, 'code.source_sha256'))),
                      ('input SHA256 (recorded)', _digest(record.get('input_sha256'))),
                      ('configuration SHA256 (recorded)', _digest(record.get('configuration_sha256')))]
        tables.append((f'Run {index}: provenance', provenance))
        tables.append((f'Run {index}: environment (recorded)', _environment(record)))
        if status != 'completed':
            error = record.get('error_type')
            error = error if isinstance(error, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,99}', error) else 'unavailable'
            tables.append((f'Run {index}: incomplete', [('status', status), ('error_type', error),
                                                       ('metrics and diagnostics', 'unavailable')]))
            continue
        desc = record['descriptive']
        tables.append((f'Run {index}: input counts', [(key, str(desc[key])) for key in ('n_cells', 'n_genes', 'n_donors')]))
        for i, fold in enumerate(run['folds']):
            mode = fold.get('mode', 'unavailable')
            mode = mode if mode in ('frozen', 'finetune', 'fine_tune', 'synthetic_smoke') else 'unavailable'
            rows = [('method', fold['method']), ('mode', mode), ('execution fold number (zero based)', str(i)),
                    ('training donors', str(len(fold['train_donors']))), ('test donors', str(len(fold['test_donors']))),
                    ('selected genes', str(len(fold['selected_genes'])) if isinstance(fold.get('selected_genes'), list) else 'unavailable')]
            rows += [(key + ' (recorded)', _scalar(fold.get(key))) for key in ('training_steps', 'encoder_updated', 'reload_verified')]
            tables.append((f'Run {index}: fold {i}', rows))
            metrics = fold.get('metrics', {})
            tables.append((f'Run {index}: fold {i} scalar diagnostics (recorded)',
                           [(key, _scalar(_get(metrics, key))) for key in _DIAGNOSTICS]))
            if fold['method'] == 'scvi':
                tables.append((f'Run {index}: fold {i} training resubstitution annotation agreement (not held out)',
                               [(key, _scalar(_get(metrics, key))) for key in _TRAINING_ANNOTATION]))
                tables.append((f'Run {index}: fold {i} held-out annotation agreement (recorded)', _heldout_rows(fold)))
    return tables


_NOTICE = (
    'PRIVATE: keep this report on approved storage. It is not approved for sharing. '
    'Disease inference is disabled. No disease metrics, clinical probabilities or p-values are reported. '
    'The donor is the independent disease evaluation unit; cells do not provide independent donor replication. '
    'Annotation agreement is not independent biological truth. Training resubstitution is not held-out evaluation. '
    'Novae uses test-section adaptation during inference; its transfer results require that qualification. '
    'Synthetic runs establish software behavior only. Missing diagnostics are unavailable. '
    'Listed completed-run artifact paths, sizes and SHA256 hashes were verified using opaque byte streams; '
    'only summary JSON was interpreted. Reload checks and environment are recorded runner evidence. '
    'Source assays, checkpoints, embeddings, donor tables and logs were not interpreted. '
    'Recorded input/source/configuration digests were not recomputed from original sources. '
    'Hash consistency does not independently authenticate the originating run.'
)


def build_report(run_paths, output):
    """Write report.md/report.html into a new private directory; inputs stay read-only."""
    _require(bool(run_paths), 'At least one run required')
    destination = Path(os.path.abspath(output))
    _no_symlinks(destination)
    _require(not os.path.lexists(destination), 'Output must be a new directory')
    runs = [_load(path) for path in run_paths]
    _require(len({run['root'] for run in runs}) == len(runs), 'Duplicate runs')
    _require(len({run['record']['data_kind'] for run in runs}) == 1, 'Cannot mix data kinds')
    _require(all(not destination.is_relative_to(run['root']) and not run['root'].is_relative_to(destination)
                 for run in runs), 'Output must not overlap source runs')
    kind = runs[0]['record']['data_kind']
    intro = _NOTICE + f' Data kind: {kind}.'
    tables = _tables(runs)
    def md(value):
        return html.escape(str(value)).replace('|', '&#124;').replace('\n', ' ')
    markdown = ['# Private neural technical report', '', intro, '']
    body = ['<h1>Private neural technical report</h1>', f'<p>{html.escape(intro)}</p>']
    for title, rows in tables:
        markdown += ['## ' + title, '', '| Field | Value |', '| --- | --- |']
        markdown += [f'| {md(key)} | {md(value)} |' for key, value in rows]
        markdown.append('')
        body += [f'<h2>{html.escape(title)}</h2>', '<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>']
        body += [f'<tr><th scope="row">{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>' for key, value in rows]
        body.append('</tbody></table>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Private neural technical report</title><style>'
                'body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5}'
                'table{border-collapse:collapse;width:100%;margin-bottom:2rem}th,td{border:1px solid #bbb;padding:.4rem;text-align:left;overflow-wrap:anywhere}'
                '@media(max-width:600px){body{margin:1rem auto}th,td{font-size:.85rem}}'
                '</style><body>' + ''.join(body) + '</body></html>')
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in (('report.md', '\n'.join(markdown)), ('report.html', document)):
        descriptor = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(content)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        build_report(args.run, args.output)
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'message': 'Report unavailable; check private run integrity and execution scope.'}))
        return 1
    print(json.dumps({'status': 'completed', 'message': 'Private report created.'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
