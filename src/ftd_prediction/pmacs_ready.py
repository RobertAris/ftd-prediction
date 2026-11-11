"""Offline, stdlib-only integrity checks and private PMACS request preparation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys
import traceback


class PreparationError(ValueError):
    pass


DESCRIPTIVE_FIELDS = {
    'schema_version', 'data_kind', 'assay', 'input_file', 'canonical_input_roots',
    'count_source', 'counts_provenance', 'mapping_verified', 'mapping_evidence_ref',
    'obs_columns', 'coordinate_units', 'spatial_key', 'seed', 'cell_cap_per_donor',
    'max_source_cells', 'max_cells', 'max_genes', 'max_matrix_bytes',
    'shared_preprocessing_scope', 'source_version', 'panel_ref', 'segmentation_ref',
    'coordinate_transform_ref', 'mapping_file', 'mapping_keys', 'spatial_graph',
}
STUDY_MODULES = ('ftd_prediction.descriptive', 'ftd_prediction.neural')
_MARKER = re.compile(r'replace|unverified|placeholder|[<>]', re.I)
_DIGEST = re.compile(r'[0-9a-f]{64}')
_ABSOLUTE = re.compile(r'/[A-Za-z0-9_./-]+')


def _require(condition, message):
    if not condition:
        raise PreparationError(message)


def _absolute(value):
    value = str(value) if isinstance(value, Path) else value
    _require(isinstance(value, str) and _ABSOLUTE.fullmatch(value)
             and '..' not in Path(value).parts and not _MARKER.search(value),
             'Expected a plain absolute path without placeholders')
    return Path(value)


def _no_links(path):
    for node in (path, *path.parents):
        _require(not node.is_symlink(), f'Symlink is forbidden: {node}')


def _regular(path, *, private=False):
    _no_links(path)
    mode = path.stat().st_mode
    _require(stat.S_ISREG(mode), f'Expected a regular file: {path}')
    if private:
        _require(mode & 0o077 == 0, f'Expected a private file: {path}')


def _sha(path):
    _regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(value):
    path = _absolute(value)
    _require(path.suffix in {'.json', '.txt', '.lock', '.toml', '.yaml', '.yml'},
             'Pinned metadata must be a config or environment record, not an assay/crosswalk')
    return path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda _: _require(False, 'Nonfinite JSON value'))


def _read(path, *, private=False):
    _regular(path, private=private)
    _require(path.stat().st_size <= 8 * 1024**2, f'Metadata file is too large: {path}')
    return path.read_bytes()


def _code_path(value):
    _require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_./-]+', value),
             'Unsafe manifest path')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and path.as_posix() == value
             and not any(p in ('.', '..') or p.startswith('.') for p in path.parts),
             'Unsafe manifest path')
    parts = path.parts
    allowed = (value in {'pyproject.toml', 'README.md', 'LICENSE'}
               or (parts[:2] == ('src', 'ftd_prediction') and path.suffix == '.py')
               or (parts[0] in {'scripts', 'tests'} and path.suffix == '.py')
               or (parts[0] == 'configs' and value.endswith('.example.json'))
               or (parts[0] == 'docs' and path.suffix == '.md'))
    _require(allowed, 'Manifest payload must contain code, examples or documentation only')
    return path


def verify_manifest(code_directory, manifest, *, expected_sha256=None):
    """Verify a clean, explicit package; never follow links or modify its files."""
    code, manifest = _absolute(code_directory), _absolute(manifest)
    _no_links(code)
    _require(code.is_dir() and manifest == code / 'MANIFEST.json',
             'MANIFEST.json must be at the code package root')
    raw = _read(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require(isinstance(expected_sha256, str) and _DIGEST.fullmatch(expected_sha256)
                 and digest == expected_sha256, 'Code manifest digest mismatch')
    record = _json(raw)
    _require(isinstance(record, dict) and set(record) == {'schema_version', 'files'}
             and type(record['schema_version']) is int and record['schema_version'] == 1
             and isinstance(record['files'], list) and record['files'], 'Invalid manifest schema')
    files = {}
    for entry in record['files']:
        _require(isinstance(entry, dict) and set(entry) == {'path', 'sha256', 'size_bytes'},
                 'Invalid manifest entry')
        relative = _code_path(entry['path'])
        _require(str(relative) not in files, 'Duplicate manifest path')
        _require(isinstance(entry['sha256'], str) and _DIGEST.fullmatch(entry['sha256'])
                 and type(entry['size_bytes']) is int and entry['size_bytes'] >= 0,
                 'Invalid manifest hash or size')
        files[str(relative)] = entry
    # Reject unlisted payloads, including bytecode that could bypass source hashes.
    actual = set()
    for directory, dirs, names in os.walk(code, followlinks=False):
        for name in dirs + names:
            _no_links(Path(directory) / name)
        actual.update((Path(directory) / name).relative_to(code).as_posix() for name in names)
    _require(actual == set(files) | {'MANIFEST.json'}, 'Missing or unlisted package files')
    for relative, entry in files.items():
        path = code / relative
        _regular(path)
        _require(path.stat().st_size == entry['size_bytes'] and _sha(path) == entry['sha256'],
                 f'Code payload mismatch: {relative}')
    return {'sha256': digest, 'file_count': len(files), 'files': files}


def _resolved_strings(value):
    if isinstance(value, str):
        _require(not _MARKER.search(value) and not any(ord(c) < 32 or ord(c) == 127 for c in value),
                 'Unresolved placeholder or control character in study configuration')
    elif isinstance(value, dict):
        for key, child in value.items():
            _resolved_strings(key)
            _resolved_strings(child)
    elif isinstance(value, list):
        for child in value:
            _resolved_strings(child)


def _provenance(config, key):
    value = config.get(key)
    _require(isinstance(value, str) and value.strip() == value and bool(value),
             f'Explicit provenance string required: {key}')


def validate_study_config(config, module):
    """Validate declarations and path metadata only; do not open assays/crosswalks."""
    _require(module in STUDY_MODULES and isinstance(config, dict), 'Unsupported study module or config')
    # An unknown shared preprocessing history is a recorded limitation, not an
    # attestation about counts or identities. All other placeholders still fail.
    unresolved = dict(config)
    if unresolved.get('shared_preprocessing_scope') == 'unverified':
        unresolved.pop('shared_preprocessing_scope')
    _resolved_strings(unresolved)
    allowed = DESCRIPTIVE_FIELDS if module.endswith('.descriptive') else (
        DESCRIPTIVE_FIELDS - {'spatial_graph'} | {'model', 'split_strategy', 'fold_indices'})
    _require(not set(config) - allowed and type(config.get('schema_version')) is int
             and config['schema_version'] == 1 and config.get('data_kind') == 'study'
             and config.get('assay') in ('rna', 'spatial'), 'Unsupported study configuration schema')
    provenance = config.get('counts_provenance')
    count_kinds = {'measured_umi', 'background_corrected_umi'}
    if config.get('assay') == 'spatial':
        count_kinds.add('measured_transcripts')
    _require(isinstance(provenance, dict) and set(provenance) == {'kind', 'evidence_ref'}
             and provenance.get('kind') in count_kinds, 'Verified assay-appropriate count provenance is required')
    _provenance(provenance, 'evidence_ref')
    _require(config.get('mapping_verified') is True, 'Verified mapping attestation is required')
    for key in ('mapping_evidence_ref', 'source_version', 'shared_preprocessing_scope'):
        _provenance(config, key)
    source = config.get('count_source')
    _require(isinstance(source, str) and (source in ('X', 'raw/X') or re.fullmatch(r'layers/[^/]+', source)),
             'An explicit count matrix source is required')
    columns = config.get('obs_columns')
    permitted = {'donor_id', 'sample_id', 'section_id', 'retinal_region', 'acquisition_region',
                 'cell_type', 'layer', 'tissue_component'}
    _require(isinstance(columns, dict) and not set(columns) - permitted
             and all(isinstance(v, str) and v.strip() == v and v and '/' not in v for v in columns.values()),
             'Explicit observation columns are required')
    mapping = config.get('mapping_file')
    _require('donor_id' in columns or mapping is not None, 'Donor mapping is required')
    if mapping is not None:
        keys = config.get('mapping_keys')
        _require(isinstance(keys, list) and keys and all(isinstance(k, str) for k in keys)
                 and len(set(keys)) == len(keys) and set(keys) <= columns.keys()
                 and set(keys) <= {'sample_id', 'retinal_region', 'acquisition_region'},
                 'Explicit crosswalk join keys are required')
    else:
        _require('mapping_keys' not in config, 'Crosswalk keys require a mapping file')
    spatial = config['assay'] == 'spatial'
    if spatial:
        _require(config.get('coordinate_units') == 'um' and ('section_id' in columns or mapping is not None),
                 'Spatial input requires verified physical sections and micron coordinates')
        for key in ('spatial_key', 'panel_ref', 'segmentation_ref', 'coordinate_transform_ref'):
            _provenance(config, key)
        _require('cell_cap_per_donor' not in config, 'Spatial cell thinning is unsupported')
    for key in ('max_source_cells', 'max_cells', 'max_genes', 'max_matrix_bytes', 'cell_cap_per_donor'):
        if key in config:
            _require(type(config[key]) is int and config[key] > 0, f'Positive integer required: {key}')
    if 'seed' in config:
        _require(type(config['seed']) is int and config['seed'] >= 0, 'Invalid seed')
    if 'spatial_graph' in config:
        graph = config['spatial_graph']
        _require(spatial and isinstance(graph, dict)
                 and {'radius_um', 'max_neighbors'} <= set(graph)
                 and not set(graph) - {'radius_um', 'max_neighbors', 'max_descriptor_bytes'},
                 'Spatial graph settings require an explicit radius and neighbor cap')
        radius, cap = graph['radius_um'], graph['max_neighbors']
        _require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0
                 and type(cap) is int and cap > 0, 'Invalid spatial graph settings')
        if 'max_descriptor_bytes' in graph:
            _require(type(graph['max_descriptor_bytes']) is int and graph['max_descriptor_bytes'] > 0,
                     'Positive integer descriptor memory budget required')
    if module.endswith('.neural'):
        model = config.get('model')
        _require(isinstance(model, dict) and model and model.get('mode') != 'synthetic_smoke',
                 'An explicit study model configuration is required')
        _require(config.get('split_strategy') in ('explicit', 'leave_one_donor_out', 'all_donor_descriptive'),
                 'An explicit donor split strategy is required')
        if spatial:
            checkpoint = _absolute(model.get('checkpoint_path'))
            _require(checkpoint.is_dir() and (checkpoint / 'config.json').is_file()
                     and (checkpoint / 'model.safetensors').is_file(), 'Existing local Novae checkpoint is required')
    roots = config.get('canonical_input_roots')
    _require(isinstance(roots, list) and roots, 'Explicit canonical input roots are required')
    roots = [_absolute(root).resolve() for root in roots]
    inputs = [_absolute(config.get('input_file'))]
    if mapping is not None:
        inputs.append(_absolute(mapping))
    for path in inputs:
        _require(path.is_file() and any(path.resolve().is_relative_to(root) for root in roots),
                 'Input path must exist within approved canonical roots')
    return roots


def verify_output_scope(workspace, run, output):
    """Recheck destination ancestry immediately before the study runner starts."""
    workspace, run, output = (_absolute(value) for value in (workspace, run, output))
    for path in (workspace, run, output):
        _no_links(path)
    _require(workspace != Path('/') and workspace.is_dir() and run.is_dir(),
             'Existing personal workspace and run directory required')
    _require(run != workspace and run.resolve().is_relative_to(workspace.resolve()),
             'Run directory must remain inside the personal workspace')
    _require(output != run and output.is_relative_to(run)
             and output.resolve().is_relative_to(run.resolve()),
             'Output must remain a new child of the approved run directory')
    _require(not output.exists() and not output.is_symlink(), 'Output is already in use')
    _require(run.stat().st_mode & 0o077 == 0, 'Run directory must remain private')


def _write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        handle.write(content)


def _private_error(directory, filename):
    try:
        _no_links(directory)
        _require(directory.is_dir() and directory.stat().st_mode & 0o077 == 0, 'Private directory required')
        _write(directory / filename, traceback.format_exc())
    except Exception:
        pass  # Never replace evidence, or leak error details to stdout/stderr.


# Run verified verifier bytes, avoiding imports from unverified code at startup.
_BOOTSTRAP = '''import hashlib, pathlib, sys
try:
    p = pathlib.Path(sys.argv[1])
    if any(n.is_symlink() for n in (p, *p.parents)):
        raise ValueError()
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sys.argv[2]:
        raise ValueError()
except Exception:
    print('{"status":"failed","stage":"verifier_integrity"}')
    sys.exit(1)
sys.argv = [str(p), *sys.argv[3:]]
exec(compile(raw, str(p), 'exec'), {'__name__': '__main__', '__file__': str(p)})
'''


def prepare(cluster_config, study_config, manifest, interpreter, request_directory, output,
            *, module='ftd_prediction.descriptive', environment_lock=None):
    from .cluster import render_lsf_text

    cluster_path, study_path = _absolute(cluster_config), _absolute(study_config)
    _require(cluster_path.suffix == study_path.suffix == '.json', 'Configuration paths must be JSON files')
    cluster_raw = _read(cluster_path, private=True)
    cluster = _json(cluster_raw)
    _require(isinstance(cluster, dict), 'Cluster config must be an object')
    workspace, code, run = (_absolute(cluster.get(key)) for key in ('workspace', 'code_directory', 'run_directory'))
    request, output = _absolute(request_directory), _absolute(output)
    _require(workspace != Path('/'), 'Personal workspace cannot be the filesystem root')
    for path in (workspace, code, run, cluster_path, study_path, request, output):
        _no_links(path)
        _require(path.is_relative_to(workspace), 'Preparation paths must stay within the personal workspace')
    _require(run.is_dir() and run.stat().st_mode & 0o077 == 0, 'Existing private log directory required')
    _require(request != run and request.is_relative_to(run) and output != run and output.is_relative_to(run),
             'Request and output must be new children of the log directory')
    _require(not request.is_relative_to(code) and not output.is_relative_to(code)
             and not output.is_relative_to(request) and not request.is_relative_to(output),
             'Code, request and output must not overlap')
    _require(not request.exists() and not output.exists() and request.parent.is_dir(),
             'Request and output must be unused; request parent must exist')
    request.mkdir(mode=0o700)
    try:
        study_raw = _read(study_path, private=True)
        config = _json(study_raw)
        roots = validate_study_config(config, module)
        for path in (code, cluster_path, study_path, request, output):
            _require(not any(path.is_relative_to(root) or root.is_relative_to(path) for root in roots),
                     'Preparation artifacts must not overlap canonical input roots')
        python = _absolute(interpreter)
        _require(str(python) == cluster.get('python_executable') and python.is_file() and os.access(python, os.X_OK),
                 'Existing executable must match the cluster configuration')
        integrity = verify_manifest(code, manifest)
        required = {'src/ftd_prediction/__init__.py', 'src/ftd_prediction/cluster.py',
                    'src/ftd_prediction/pmacs_ready.py', 'src/ftd_prediction/' + module.rsplit('.', 1)[1] + '.py'}
        _require(required <= integrity['files'].keys(), 'Manifest must include the runner and preparation code')
        frozen = request / 'study.json'
        _write(frozen, study_raw.decode('utf-8'))
        argv = [str(python), '-m', module, '--config', str(frozen), '--output', str(output),
                '--mode', 'study', '--execution-context', 'pmacs']
        rendered_config = {**cluster, 'module': module, 'arguments': argv[3:]}
        script = render_lsf_text(rendered_config)
        pins = [(study_path, hashlib.sha256(study_raw).hexdigest()),
                (frozen, hashlib.sha256(study_raw).hexdigest()),
                (cluster_path, hashlib.sha256(cluster_raw).hexdigest())]
        if environment_lock is not None:
            lock = _metadata_path(environment_lock)
            _require(lock.is_relative_to(workspace), 'Environment lock must be within the personal workspace')
            pins.append((lock, _sha(lock)))
        verifier = code / 'src/ftd_prediction/pmacs_ready.py'
        verify_argv = [str(python), '-I', '-B', '-c', _BOOTSTRAP, str(verifier),
                       integrity['files']['src/ftd_prediction/pmacs_ready.py']['sha256'],
                       'verify', '--code-directory', str(code), '--manifest', str(manifest),
                       '--manifest-sha256', integrity['sha256'], '--error-directory', str(request),
                       '--scope-workspace', str(workspace), '--scope-run', str(run),
                       '--scope-output', str(output)]
        for path, digest in pins:
            verify_argv += ['--file-digest', str(path), digest]
        cache = request / 'cache'
        startup = ['if ! [[ "${LSB_JOBID:-}" =~ ^[1-9][0-9]*$ ]]; then',
                   '  echo \'{"status":"failed","stage":"lsf_context"}\'; exit 1', 'fi', shlex.join(verify_argv),
                   f'test ! -e {shlex.quote(str(output))}', f'test ! -L {shlex.quote(str(output))}',
                   'mkdir -m 700 ' + shlex.quote(str(cache))]
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            startup.append(f'export {key}={cluster["cpu_slots"]}')
        for key, value in {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                           'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                           'WANDB_MODE': 'offline', 'TOKENIZERS_PARALLELISM': 'false',
                           'XDG_CACHE_HOME': str(cache), 'HF_HOME': str(cache / 'hf'),
                           'TORCH_HOME': str(cache / 'torch'), 'MPLCONFIGDIR': str(cache / 'mpl'),
                           'NUMBA_CACHE_DIR': str(cache / 'numba')}.items():
            startup.append(f'export {key}={shlex.quote(value)}')
        script = script.replace('exec ' + shlex.join(argv), '\n'.join(startup) + '\nexec ' + shlex.join(argv))
        receipt = {'schema_version': 1, 'status': 'prepared', 'lsf_job_id': 'pending',
                   'argv': argv, 'verification_argv': verify_argv, 'output_path': str(output),
                   'interpreter_path': str(python), 'resolved_interpreter_path': str(python.resolve()),
                   'code_manifest_sha256': integrity['sha256'], 'code_file_count': integrity['file_count'],
                   'config_sha256': hashlib.sha256(study_raw).hexdigest(),
                   'pinned_files': [{'path': str(p), 'sha256': d} for p, d in pins],
                   'resources': {key: rendered_config.get(key, 'ftd_preflight') for key in ('queue', 'job_name', 'cpu_slots', 'wall_time',
                       'resource_expression', 'memory_scope', 'memory_units', 'memory_semantics_verified', 'run_directory')},
                   'environment_versions': 'Recorded by the actual runner; not measured during preparation',
                   'input_validation': 'Declarations and path metadata only; assay/crosswalk contents unopened',
                   'disease_inference_enabled': False}
        # Catch edits made during preparation, before publishing an executable request.
        verify_manifest(code, manifest, expected_sha256=integrity['sha256'])
        _require(all(_sha(p) == d for p, d in pins), 'Configuration changed during preparation')
        _write(request / 'job.lsf', script)
        _write(request / 'receipt.json', json.dumps(receipt, indent=2) + '\n')
    except Exception:
        _private_error(request, 'preparation_error.log')
        raise
    return request / 'receipt.json'


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise PreparationError('Invalid command line; use --help')


def main(argv=None):
    error_directory = None
    try:
        parser = _Parser(description=__doc__, allow_abbrev=False)
        commands = parser.add_subparsers(dest='command', required=True)
        verify = commands.add_parser('verify', allow_abbrev=False)
        verify.add_argument('--code-directory', required=True)
        verify.add_argument('--manifest', required=True)
        verify.add_argument('--manifest-sha256')
        verify.add_argument('--file-digest', nargs=2, action='append', default=[])
        verify.add_argument('--error-directory')
        for name in ('scope-workspace', 'scope-run', 'scope-output'):
            verify.add_argument('--' + name)
        prep = commands.add_parser('prepare', allow_abbrev=False)
        for name in ('cluster-config', 'study-config', 'manifest', 'interpreter', 'request-directory', 'output'):
            prep.add_argument('--' + name, required=True)
        prep.add_argument('--module', choices=STUDY_MODULES, default=STUDY_MODULES[0])
        prep.add_argument('--environment-lock')
        supplied = list(sys.argv[1:] if argv is None else argv)
        seen = set()
        for argument in supplied:
            if argument.startswith('--'):
                flag = argument.split('=', 1)[0]
                _require(flag not in seen or flag == '--file-digest', 'Duplicate CLI option')
                seen.add(flag)
        args = parser.parse_args(supplied)
        if args.command == 'verify':
            error_directory = _absolute(args.error_directory) if args.error_directory else None
            record = verify_manifest(args.code_directory, args.manifest, expected_sha256=args.manifest_sha256)
            for path, digest in args.file_digest:
                _require(_DIGEST.fullmatch(digest) and _sha(_metadata_path(path)) == digest, 'Pinned metadata digest mismatch')
            scope = (args.scope_workspace, args.scope_run, args.scope_output)
            if any(value is not None for value in scope):
                _require(all(value is not None for value in scope), 'Complete output scope is required')
                verify_output_scope(*scope)
            print(json.dumps({'status': 'verified', 'file_count': record['file_count']}))
        else:
            prepare(args.cluster_config, args.study_config, args.manifest, args.interpreter,
                    args.request_directory, args.output, module=args.module, environment_lock=args.environment_lock)
            print(json.dumps({'status': 'prepared', 'job_id': 'pending'}))
    except Exception as error:
        if error_directory is not None:
            _private_error(error_directory, 'startup_error.log')
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
