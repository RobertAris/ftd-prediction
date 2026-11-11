"""Bounded, private specimen-level relative-transcript-abundance contrasts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
import traceback
import zipfile
from importlib.metadata import version

import numpy as np
from scipy import sparse

from .cli import validate_runtime_scope
from .provenance import utc_now


PARAMETERS = dict(min_cells=20, min_gene_counts=10, pseudocount_cpm=1)
LIMITS = dict(contrasts=64, donors=128, references=16, genes=60000,
              groups=100000, nnz=4000000, output_rows=250000,
              matched_rows=250000, dense_elements=2000000,
              npz_bytes=64 * 1024**2, array_bytes=96 * 1024**2,
              csv_bytes=32 * 1024**2, config_bytes=1024**2,
              run_bytes=8 * 1024**2, count_total=2**53 - 1)
ARTIFACTS = ('pseudobulk_counts.npz', 'pseudobulk_groups.csv', 'pseudobulk_genes.csv')
_FIELDS = {'schema_version', 'data_kind', 'assay', 'source_run_dir',
           'source_run_sha256', 'source_annotation_evidence_ref',
           'feature_universe_evidence_ref', 'endogenous_gene_ids_file',
           'all_features_endogenous_verified', 'contrasts'}
_CONTRAST = {'contrast_id', 'case_donor_id', 'reference_donor_ids',
             'retinal_region', 'cell_type', 'label_evidence_ref', 'matching_evidence_ref'}
_IDENTITY = ('donor_id', 'section_id', 'retinal_region', 'cell_type')
_FLAGS = dict(donor_id='donor_id_missing', section_id='section_id_missing',
              retinal_region='retinal_region_missing', cell_type='annotation_missing')
CAUTION = (
    'Specimen-level relative-transcript-abundance contrasts only. One affected '
    'donor does not support replicated FTD associations. Inherited cell annotations '
    'are not independent validation. Relative CPM can reflect compositional closure. '
    'Assays remain separate; sections and cells are not independent disease donors.'
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return (isinstance(value, str) and 0 < len(value) <= 4096
            and value == value.strip() and not any(ord(c) < 32 for c in value)
            and 'REPLACE' not in value.upper()
            and not any(marker in value for marker in '<>')
            and value.lower() not in {'false', 'unknown', 'unresolved', 'unverified', 'placeholder', 'todo', 'tbd'})


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _absolute(value, base=None, required=False):
    _require(_text(str(value)), 'Resolved paths are required')
    path = Path(value)
    _require('..' not in path.parts, 'Parent traversal is forbidden')
    _require(not required or path.is_absolute(), 'An absolute path is required')
    return Path(os.path.abspath((base or Path.cwd()) / path))


def _directory(path):
    """Walk with directory descriptors: no symlink ancestor is followed."""
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        pending_writable = False
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
            info = os.fstat(fd)
            # Sticky system temporary directories are safe for owned private children.
            sticky_system = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
            if info.st_mode & 0o022 and not sticky_system:
                pending_writable = True
            # Approved lab storage may have a shared group-writable parent.
            # Require an owned private directory below it, and reject later
            # writable descendants unless another owned private boundary follows.
            if info.st_uid == os.getuid() and not info.st_mode & 0o077:
                pending_writable = False
        _require(not pending_writable, 'An ancestor is writable by untrusted users')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _private(path):
    """A private file or a private owned enclosing directory protects declarations."""
    for item in (path, *path.parents):
        info = item.lstat()
        _require(not stat.S_ISLNK(info.st_mode), 'Symlinks are forbidden')
        if info.st_uid == os.getuid() and not info.st_mode & 0o077:
            return
    raise ValueError('Configuration and universe declarations must be private')


def _read(path, limit):
    directory = _directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                 'Input must be a regular, unlinked-alias-free file')
        _require(before.st_size <= limit, 'Input exceeds the byte limit')
        data = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        _require(len(data) <= limit and before.st_size == len(data)
                 and (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                 == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 'Input changed while reading')
    return data


def _pin(path, data):
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'Duplicate JSON keys')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: _require(False, 'Nonfinite JSON value'))


def _config(cfg, mode, context):
    _require(isinstance(cfg, dict) and set(cfg) <= _FIELDS
             and type(cfg.get('schema_version')) is int and cfg['schema_version'] == 1,
             'Unsupported molecular configuration schema')
    _require(mode in ('synthetic', 'study') and context in ('local', 'pmacs'),
             'Unsupported execution scope')
    # This guard precedes all source-directory, artifact and universe access.
    validate_runtime_scope(cfg, mode=mode, execution_context=context)
    _require(cfg.get('assay') in ('rna', 'spatial'), 'An explicit assay is required')
    for field in ('source_run_dir', 'source_annotation_evidence_ref', 'feature_universe_evidence_ref'):
        _require(_text(cfg.get(field)), 'Resolved declarations are required')
    _require(_sha(cfg.get('source_run_sha256')), 'An exact source-run SHA256 pin is required')
    universe = 'endogenous_gene_ids_file' in cfg
    _require(universe != ('all_features_endogenous_verified' in cfg),
             'Exactly one endogenous feature-universe declaration is required')
    if universe:
        _require(_text(cfg['endogenous_gene_ids_file']), 'Resolved universe path is required')
    else:
        _require(cfg['all_features_endogenous_verified'] is True,
                 'All-feature endogenous verification must be true')
    contrasts = cfg.get('contrasts')
    _require(isinstance(contrasts, list) and 1 <= len(contrasts) <= LIMITS['contrasts'],
             'Contrast count exceeds limits or no contrasts declared')
    ids = set()
    for item in contrasts:
        _require(isinstance(item, dict) and set(item) == _CONTRAST,
                 'Explicit contrast declarations are required')
        _require(all(_text(item[k]) and len(item[k]) <= 256
                     for k in _CONTRAST - {'reference_donor_ids'}), 'Unresolved contrast declaration')
        refs = item['reference_donor_ids']
        _require(isinstance(refs, list) and 1 <= len(refs) <= LIMITS['references']
                 and all(_text(r) and len(r) <= 256 for r in refs)
                 and len(set(refs)) == len(refs) and item['case_donor_id'] not in refs,
                 'References must be distinct and exclude the case')
        _require(item['contrast_id'] not in ids, 'Duplicate contrast ID')
        ids.add(item['contrast_id'])


def _csv(data, max_rows):
    rows = csv.reader(io.StringIO(data.decode('utf-8'), newline=''), strict=True)
    header = next(rows, [])
    _require(header and len(set(header)) == len(header), 'Invalid or duplicate CSV columns')
    result = []
    for row in rows:
        _require(len(result) < max_rows and len(row) == len(header),
                 'CSV dimensions exceed limits or are inconsistent')
        _require(all(len(v) <= 256 and not any(ord(c) < 32 for c in v) for v in row),
                 'CSV values exceed limits or contain control characters')
        result.append(dict(zip(header, row)))
    return header, result


def _genes(data):
    header, rows = _csv(data, LIMITS['genes'])
    values = [r['gene_id'] for r in rows] if header == ['gene_id'] else []
    _require(values and all(v and v == v.strip() for v in values)
             and len(set(values)) == len(values), 'Unique nonempty gene IDs are required')
    return values


def _groups(data, descriptor, assay):
    header, rows = _csv(data, LIMITS['groups'])
    keys = [key for key in _IDENTITY if key in header]
    expected = set(keys) | {_FLAGS[k] for k in keys} | {'n_cells'}
    _require(rows and 'donor_id' in keys and set(header) == expected
             and descriptor.get('grouping_columns') == keys,
             'Unsupported pseudobulk group columns')
    _require(assay != 'spatial' or 'section_id' in keys, 'Spatial sections must be recorded')
    _require(descriptor.get('cell_type_annotation_available') is ('cell_type' in keys),
             'Inconsistent annotation-availability flag')
    seen = set()
    for i, row in enumerate(rows):
        signature = tuple(row[k] for k in keys)
        _require(signature not in seen, 'Duplicate pseudobulk group identity')
        seen.add(signature)
        _require(row['n_cells'].isascii() and row['n_cells'].isdigit()
                 and 0 < int(row['n_cells']) <= LIMITS['count_total'], 'Invalid group cell count')
        row['n_cells'] = int(row['n_cells'])
        for key in keys:
            flag = _FLAGS[key]
            _require(row[flag] in ('True', 'False'), 'Invalid missing-value flag')
            row[flag] = row[flag] == 'True'
            _require(row[flag] == (row[key] == ''), 'Inconsistent missing-value flag')
        row['source_group_index'] = i
    donors = {r['donor_id'] for r in rows if not r['donor_id_missing']}
    _require(0 < len(donors) <= LIMITS['donors'], 'Donor count exceeds limits')
    _require(descriptor.get('n_groups') == len(rows)
             and descriptor.get('n_cells') == sum(r['n_cells'] for r in rows)
             and descriptor.get('n_donors') == len(donors), 'Inconsistent source dimensions')
    return rows, donors


def _matrix(data, shape):
    """Preflight every NPY header and ZIP size before allocating any arrays."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        required = {'format.npy', 'shape.npy', 'data.npy', 'indices.npy', 'indptr.npy'}
        _require(len(names) == len(set(names)) and required <= set(names)
                 and set(names) <= required | {'_is_array.npy'}, 'Unsupported sparse NPZ members')
        _require(sum(i.file_size for i in infos) <= LIMITS['array_bytes'],
                 'Sparse NPZ expanded bytes exceed limit')
        headers = {}
        for info in infos:
            _require(not info.flag_bits & 1 and info.compress_type in (0, 8),
                     'Unsupported sparse NPZ encoding')
            with archive.open(info) as handle:
                fmt = np.lib.format.read_magic(handle)
                _require(fmt in ((1, 0), (2, 0)), 'Unsupported NPY header')
                read_header = (np.lib.format.read_array_header_1_0 if fmt == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                dims, fortran, dtype = read_header(handle, max_header_size=4096)
                _require(not dtype.hasobject and not dtype.fields and dtype.subdtype is None
                         and all(type(n) is int and n >= 0 for n in dims), 'Unsafe NPY dtype or shape')
                size = math.prod(dims) * dtype.itemsize
                _require(size <= LIMITS['array_bytes'] and handle.tell() + size == info.file_size,
                         'NPY payload dimensions or bytes are inconsistent')
                headers[info.filename] = (dims, dtype)
        for name in ('data.npy', 'indices.npy', 'indptr.npy'):
            dims, dtype = headers[name]
            cap = LIMITS['nnz'] if name != 'indptr.npy' else max(shape) + 1
            _require(len(dims) == 1 and dims[0] <= cap and dtype.kind in (
                'iuf' if name == 'data.npy' else 'iu'), 'Sparse array exceeds shape or dtype limits')
        _require(headers['shape.npy'][0] == (2,) and headers['shape.npy'][1].kind in 'iu'
                 and headers['format.npy'][0] == () and headers['format.npy'][1].kind == 'S'
                 and headers['format.npy'][1].itemsize <= 3, 'Invalid sparse format metadata')
        if '_is_array.npy' in headers:
            _require(headers['_is_array.npy'] == ((), np.dtype('bool')), 'Invalid sparse array marker')
        arrays = {name[:-4]: np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
                  for name in names}
    _require(tuple(int(v) for v in arrays['shape']) == shape, 'Sparse matrix dimensions disagree')
    fmt = arrays['format'].item()
    _require(fmt in (b'csr', b'csc'), 'Only CSR and CSC sparse matrices are supported')
    counts, indices, ptr = arrays['data'], arrays['indices'], arrays['indptr']
    axis = 0 if fmt == b'csr' else 1
    _require(len(ptr) == shape[axis] + 1 and len(indices) == len(counts)
             and ptr[0] == 0 and ptr[-1] == len(counts)
             and np.all(ptr[1:] >= ptr[:-1]) and np.all(ptr <= len(counts))
             and np.all(indices >= 0) and np.all(indices < shape[1 - axis]),
             'Invalid sparse indices or pointers')
    _require(np.all(np.isfinite(counts)) and np.all(counts >= 0)
             and np.all(counts <= LIMITS['count_total']) and np.all(counts == np.floor(counts)),
             'Counts must be finite nonnegative integers')
    _require(sum(int(v) for v in counts) <= LIMITS['count_total'], 'Count total exceeds exact arithmetic limit')
    constructor = sparse.csr_matrix if axis == 0 else sparse.csc_matrix
    matrix = constructor((counts.astype(np.int64), indices.astype(np.int64), ptr.astype(np.int64)), shape=shape)
    matrix.check_format(full_check=True)
    matrix.sort_indices()  # The accepted descriptive sparse product need not be sorted.
    _require(matrix.has_canonical_format, 'Sparse coordinates must be unique')
    return matrix.tocsr()


def _source(cfg, root, pins):
    path = root / 'provenance' / 'run.json'
    data = _read(path, LIMITS['run_bytes'])
    pin = _pin(path, data)
    _require(pin['sha256'] == cfg['source_run_sha256'], 'Source run does not match its SHA256 pin')
    pins.append((path, LIMITS['run_bytes'], pin))
    record = _json(data)
    _require(isinstance(record, dict) and type(record.get('schema_version')) is int
             and record['schema_version'] == 1 and record.get('status') == 'completed'
             and record.get('data_kind') == cfg['data_kind']
             and record.get('training_enabled') is False
             and record.get('disease_inference_enabled') is False,
             'A completed descriptive-only source run is required')
    source_cfg = record.get('configuration', {})
    _require(isinstance(source_cfg, dict) and source_cfg.get('assay') == cfg['assay']
             and source_cfg.get('data_kind') == cfg['data_kind']
             and type(source_cfg.get('schema_version')) is int and source_cfg['schema_version'] == 1,
             'Source configuration assay or scope disagrees')
    if cfg['assay'] == 'spatial':
        _require(_text(source_cfg.get('segmentation_ref')), 'A resolved source segmentation declaration is required')
    descriptor = record.get('descriptive', {})
    _require(isinstance(descriptor, dict) and descriptor.get('disease_effect_inference') is False,
             'Descriptive source dimensions and scope are required')
    for field, cap in (('n_groups', LIMITS['groups']), ('n_genes', LIMITS['genes']),
                       ('n_donors', LIMITS['donors']), ('n_cells', LIMITS['count_total'])):
        _require(type(descriptor.get(field)) is int and 0 < descriptor[field] <= cap,
                 'Source dimensions exceed limits')
    artifacts = record.get('artifacts')
    _require(isinstance(artifacts, list), 'Source artifact hashes are required')
    declared = {}
    for artifact in artifacts:
        _require(isinstance(artifact, dict), 'Invalid artifact declaration')
        name = artifact.get('path')
        _require(_text(name) and not Path(name).is_absolute() and '..' not in Path(name).parts
                 and '\\' not in name and name == Path(name).as_posix()
                 and name not in declared and _sha(artifact.get('sha256'))
                 and type(artifact.get('bytes')) is int and artifact['bytes'] >= 0,
                 'Invalid or duplicate source artifact pin')
        declared[name] = artifact
    loaded = {}
    for name in ARTIFACTS:
        _require(name in declared, 'A required pseudobulk artifact is missing')
        limit = LIMITS['npz_bytes'] if name.endswith('.npz') else LIMITS['csv_bytes']
        _require(declared[name]['bytes'] <= limit, 'Declared artifact exceeds byte limit')
        blob = _read(root / name, limit)
        actual = _pin(root / name, blob)
        _require(all(actual[k] == declared[name][k] for k in ('bytes', 'sha256')),
                 'Source artifact hash or byte size disagrees')
        pins.append((root / name, limit, actual))
        loaded[name] = blob
    genes = _genes(loaded[ARTIFACTS[2]])
    _require(len(genes) == descriptor['n_genes'], 'Source gene dimensions disagree')
    groups, donors = _groups(loaded[ARTIFACTS[1]], descriptor, cfg['assay'])
    matrix = _matrix(loaded[ARTIFACTS[0]], (len(groups), len(genes)))
    return record, genes, groups, donors, matrix


def _unchanged(pins, private_paths):
    for path, limit, expected in pins:
        _require(_pin(path, _read(path, limit)) == expected, 'A pinned input changed during execution')
    for path in private_paths:
        _private(path)


def _new_output(destination, inputs):
    for path in inputs:
        _require(not destination.is_relative_to(path) and not path.is_relative_to(destination),
                 'Output overlaps a source or declaration')
    fd = _directory(destination.parent)
    try:
        os.mkdir(destination.name, 0o700, dir_fd=fd)
    finally:
        os.close(fd)


def _open(path, replace=False):
    fd = _directory(path.parent)
    try:
        flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK | (0 if replace else os.O_CREAT | os.O_EXCL)
        handle = os.open(path.name, flags, 0o600, dir_fd=fd)
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o077:
            os.close(handle)
            raise ValueError('Unsafe private output file')
        if replace:
            os.ftruncate(handle, 0)
    finally:
        os.close(fd)
    return os.fdopen(handle, 'w', encoding='utf-8', newline='')


def _write_json(path, value, replace=False):
    with _open(path, replace) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _outputs(destination):
    """Stream bounded output hashes rather than reading the entire CSV into RAM."""
    result = []
    for path in sorted(destination.iterdir()):
        if path.name == 'provenance':
            continue
        fd = _directory(destination)
        try:
            opened = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        finally:
            os.close(fd)
        with os.fdopen(opened, 'rb') as handle:
            info = os.fstat(handle.fileno())
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not info.st_mode & 0o077,
                     'Unsafe output artifact')
            _require(info.st_size <= 6000 * (LIMITS['output_rows'] + LIMITS['matched_rows']),
                     'Output artifact exceeds byte limit')
            digest = hashlib.sha256()
            observed = 0
            for chunk in iter(lambda: handle.read(1024**2), b''):
                observed += len(chunk)
                _require(observed <= info.st_size, 'Output artifact grew while hashing')
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            _require((info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                     == (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Output artifact changed')
        result.append(dict(path=path.name, bytes=info.st_size, sha256=digest.hexdigest()))
    return result


def _population(matrix, groups, donor, contrast, columns, assay):
    selected, coverage = [], []
    for row in groups:
        if row['donor_id_missing'] or row['donor_id'] != donor:
            continue
        reason = ''
        for key, absent in (('cell_type', 'annotation_not_recorded'), ('retinal_region', 'region_not_recorded')):
            if key not in row:
                reason = absent
                break
            if row[_FLAGS[key]]:
                reason = 'annotation_missing' if key == 'cell_type' else 'region_missing'
                break
        if not reason and assay == 'spatial' and row['section_id_missing']:
            reason = 'section_missing'
        if not reason and row['retinal_region'] != contrast['retinal_region']:
            reason = 'other_region'
        if not reason and row['cell_type'] != contrast['cell_type']:
            reason = 'other_cell_type'
        if not reason:
            selected.append(row['source_group_index'])
        coverage.append((row, reason))
    cells = sum(groups[i]['n_cells'] for i in selected)
    counts = None
    library = None
    reason = 'missing_population'
    if selected:
        # Raw section sums precede any library normalization; never average sections.
        counts = np.asarray(matrix[selected][:, columns].sum(axis=0)).ravel()
        library = int(counts.sum())
        reason = 'low_cells' if cells < PARAMETERS['min_cells'] else ('zero_library' if library == 0 else '')
    sections = (len({groups[i]['section_id'] for i in selected if not groups[i]['section_id_missing']})
                if 'section_id' in groups[0] else None)
    return dict(counts=counts, library=library, cells=cells, reason=reason,
                coverage=coverage, sections=sections)


def _normalized(pop):
    if pop['reason']:
        return None, None
    cpm = pop['counts'].astype(np.float64) * (1e6 / pop['library'])
    return cpm, np.log2(cpm + PARAMETERS['pseudocount_cpm'])


def _analyse(cfg, genes, groups, donors, matrix, columns, destination):
    contrasts = sorted(cfg['contrasts'], key=lambda c: c['contrast_id'])
    aliases = {d: f'D{i + 1:03d}' for i, d in enumerate(sorted(donors))}
    _require(all(set([c['case_donor_id'], *c['reference_donor_ids']]) <= donors for c in contrasts),
             'A declared donor ID is absent from the source')
    row_count = len(columns) * sum(1 + len(c['reference_donor_ids']) for c in contrasts)
    _require(row_count <= LIMITS['output_rows'], 'Molecular output rows exceed limit')
    coverage_count = sum(sum(r['donor_id'] in {c['case_donor_id'], *c['reference_donor_ids']}
                             for r in groups) + len(c['reference_donor_ids']) + 1 for c in contrasts)
    _require(coverage_count <= LIMITS['matched_rows'], 'Matched-group output rows exceed limit')
    _require(max(len(c['reference_donor_ids']) + 1 for c in contrasts) * len(columns)
             <= LIMITS['dense_elements'], 'Population dense allocation exceeds limit')
    fields = ['contrast_alias', 'comparison', 'case_alias', 'reference_alias', 'reference_aliases',
              'gene_id', 'case_counts', 'reference_counts', 'case_library_size', 'reference_library_sizes',
              'case_n_cells', 'reference_n_cells', 'n_references', 'n_eligible_references',
              'gene_total_counts', 'case_cpm', 'case_logcpm', 'reference_cpms', 'reference_logcpms',
              'reference_logcpm', 'logcpm_difference', 'status', 'reason']
    coverage_fields = ['contrast_alias', 'donor_alias', 'donor_id', 'source_group_index',
                       'section_id', 'section_id_missing', 'retinal_region', 'cell_type', 'annotation_missing',
                       'retinal_region_missing', 'n_cells', 'selected', 'exclusion_reason',
                       'population_status', 'population_reason', 'selected_n_cells', 'selected_sections']
    summaries, correspondence = [], []
    with _open(destination / 'molecular_contrasts.csv') as result, _open(destination / 'matched_groups.csv') as matched:
        writer, coverage_writer = csv.DictWriter(result, fields), csv.DictWriter(matched, coverage_fields)
        writer.writeheader()
        coverage_writer.writeheader()
        for index, contrast in enumerate(contrasts):
            alias = f'C{index + 1:03d}'
            case_id = contrast['case_donor_id']
            refs = sorted(contrast['reference_donor_ids'])
            populations = {d: _population(matrix, groups, d, contrast, columns, cfg['assay'])
                           for d in [case_id, *refs]}
            normalized = {d: _normalized(p) for d, p in populations.items()}
            correspondence.append(dict(contrast_alias=alias, declaration=contrast))
            for donor, pop in populations.items():
                for row, exclusion in pop['coverage'] or [({}, 'missing_population')]:
                    coverage_writer.writerow(dict(contrast_alias=alias, donor_alias=aliases[donor], donor_id=donor,
                        **{k: row.get(k, '') for k in coverage_fields[3:10]},
                        n_cells=row.get('n_cells', ''), selected=not exclusion, exclusion_reason=exclusion,
                        population_status='unavailable' if pop['reason'] else 'available',
                        population_reason=pop['reason'], selected_n_cells=pop['cells'], selected_sections=pop['sections']))
            outcomes = []
            for used_refs, kind in [([r], 'individual_reference') for r in refs] + [(refs, 'mean_reference')]:
                reasons = [f'{aliases[d]}:{populations[d]["reason"]}' for d in [case_id, *used_refs]
                           if populations[d]['reason']]
                available = 0
                low_counts = 0
                for j, col in enumerate(columns):
                    case = populations[case_id]
                    ref_pops = [populations[d] for d in used_refs]
                    raw_case = int(case['counts'][j]) if case['counts'] is not None else None
                    raw_refs = [int(p['counts'][j]) if p['counts'] is not None else None for p in ref_pops]
                    total = None if raw_case is None or None in raw_refs else raw_case + sum(raw_refs)
                    reason = ';'.join(reasons)
                    if not reason and total < PARAMETERS['min_gene_counts']:
                        reason = 'low_counts'
                        low_counts += 1
                    values = dict(contrast_alias=alias, comparison=kind, case_alias=aliases[case_id],
                        reference_alias=aliases[used_refs[0]] if kind == 'individual_reference' else 'mean_reference',
                        reference_aliases=json.dumps([aliases[d] for d in used_refs]), gene_id=genes[col],
                        case_counts=raw_case, reference_counts=json.dumps(raw_refs), case_library_size=case['library'],
                        reference_library_sizes=json.dumps([p['library'] for p in ref_pops]), case_n_cells=case['cells'],
                        reference_n_cells=json.dumps([p['cells'] for p in ref_pops]), n_references=len(used_refs),
                        n_eligible_references=sum(not p['reason'] for p in ref_pops), gene_total_counts=total,
                        status='unavailable' if reason else 'available', reason=reason)
                    if not reason:
                        case_cpm, case_log = normalized[case_id]
                        ref_cpms = [float(normalized[d][0][j]) for d in used_refs]
                        ref_logs = [float(normalized[d][1][j]) for d in used_refs]
                        mean_log = sum(ref_logs) / len(ref_logs)
                        values.update(case_cpm=float(case_cpm[j]), case_logcpm=float(case_log[j]),
                                      reference_cpms=json.dumps(ref_cpms), reference_logcpms=json.dumps(ref_logs),
                                      reference_logcpm=mean_log, logcpm_difference=float(case_log[j]) - mean_log)
                        available += 1
                    writer.writerow(values)
                outcomes.append(dict(comparison=kind, reference_aliases=[aliases[d] for d in used_refs],
                                     available_genes=available, low_count_genes=low_counts,
                                     unavailable_genes=len(columns) - available, population_reasons=reasons))
            summaries.append(dict(contrast_alias=alias, case_alias=aliases[case_id],
                                  reference_aliases=[aliases[d] for d in refs], comparisons=outcomes))
    return dict(label='specimen-level relative-transcript-abundance contrasts',
                caution=CAUTION, parameters=PARAMETERS, n_endogenous_genes=len(columns),
                n_output_rows=row_count, contrasts=summaries), dict(donors=aliases, contrasts=correspondence)


def _code():
    return [_pin(p, _read(p, LIMITS['config_bytes'])) for p in
            (Path(__file__).absolute(), Path(__file__).with_name('cli.py').absolute(),
             Path(__file__).with_name('provenance.py').absolute(), Path(__file__).with_name('__init__.py').absolute())]


def execute(config_path, output, mode='study', execution_context='pmacs'):
    """Consume only pinned descriptive artifacts; all resulting files remain private."""
    config_path = _absolute(config_path)
    config_bytes = _read(config_path, LIMITS['config_bytes'])
    _private(config_path)
    cfg = _json(config_bytes)
    _config(cfg, mode, execution_context)
    root = _absolute(cfg['source_run_dir'], config_path.parent, required=mode == 'study')
    universe = (_absolute(cfg['endogenous_gene_ids_file'], required=True)
                if 'endogenous_gene_ids_file' in cfg else None)
    destination = _absolute(output)
    _new_output(destination, [root, config_path] + ([universe] if universe else []))
    record = dict(schema_version=1, status='running', started_at=utc_now(), data_kind=mode,
                  assay=cfg['assay'], execution_context=execution_context, lsf_job_id=os.environ.get('LSB_JOBID'),
                  configuration=cfg, configuration_path=str(config_path), output_directory=str(destination),
                  configuration_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  parameters=PARAMETERS, limits=LIMITS, training_enabled=False, disease_inference_enabled=False)
    record_path = destination / 'provenance' / 'run.json'
    pins = [(config_path, LIMITS['config_bytes'], _pin(config_path, config_bytes))]
    private_paths = [config_path] + ([universe] if universe else [])
    try:
        fd = _directory(destination)
        try:
            os.mkdir('provenance', 0o700, dir_fd=fd)
        finally:
            os.close(fd)
        _write_json(record_path, record)
        record['code'] = _code()
        record['environment'] = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                     packages={p: version(p) for p in ('numpy', 'scipy')})
        source, genes, groups, donors, matrix = _source(cfg, root, pins)
        record['inherited_not_reverified'] = {k: source[k] for k in (
            'code', 'configuration', 'configuration_sha256', 'input_files', 'input_sha256', 'input_selection') if k in source}
        if universe:
            blob = _read(universe, LIMITS['csv_bytes'])
            _private(universe)
            pins.append((universe, LIMITS['csv_bytes'], _pin(universe, blob)))
            endogenous = _genes(blob)
            _require(set(endogenous) <= set(genes), 'An endogenous universe gene is absent from the source')
        else:
            endogenous = genes
        locations = {g: i for i, g in enumerate(genes)}
        columns = [locations[g] for g in sorted(endogenous)]
        _unchanged(pins, private_paths)
        record['input_pins_before'] = [p[2] for p in pins]
        summary, aliases = _analyse(cfg, genes, groups, donors, matrix, columns, destination)
        record['private_source_correspondence'] = aliases
        summary.update(schema_version=1, data_kind=mode, assay=cfg['assay'],
                       training_enabled=False, disease_inference_enabled=False,
                       evidence_scope='synthetic_pipeline_check' if mode == 'synthetic' else 'specimen_descriptive_only')
        _write_json(destination / 'summary.json', summary)
        with _open(destination / 'report.md') as report:
            report.write('Specimen-level relative-transcript-abundance contrasts\n\n' + CAUTION + '\n\n')
            report.write(f'Evidence: {summary["evidence_scope"]}. Endogenous genes: {len(columns)}. '
                         'All sections of each exact declared donor × region × cell type are summed first. '
                         'CPM = 1e6 × counts / endogenous library; logCPM = log2(CPM + 1). '
                         'Reference means weight donor logCPM equally after normalization. '
                         'Minimum cells: 20; minimum combined gene counts: 10. No reference is silently dropped.\n\n')
            report.write('| Contrast | Comparison | Reference aliases | Available genes | Unavailable genes |\n'
                         '|---|---|---|---:|---:|\n')
            for item in summary['contrasts']:
                for row in item['comparisons']:
                    report.write(f'| {item["contrast_alias"]} | {row["comparison"]} | '
                                 f'{", ".join(row["reference_aliases"])} | {row["available_genes"]} | '
                                 f'{row["unavailable_genes"]} |\n')
        record['artifacts'] = _outputs(destination)
        _unchanged(pins, private_paths)
        _require(_code() == record['code'], 'Executing code changed during execution')
        record['input_pins_after'] = [p[2] for p in pins]
        record['status'] = 'completed'
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__)
        with _open(destination / 'private_error.log') as log:
            traceback.print_exc(file=log)
        # Preserve the original error even when an unexpected output is unsafe.
        try:
            record['artifacts'] = _outputs(destination)
        except Exception:
            record['output_hash_status'] = 'incomplete_due_to_unsafe_output'
        raise
    finally:
        record['finished_at'] = utc_now()
        record['observed_input_pins'] = [p[2] for p in pins]
        _write_json(record_path, record, replace=record_path.exists())
    return destination


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command arguments')


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('study', 'synthetic'), default='study')
    parser.add_argument('--execution-context', choices=('local', 'pmacs'), default='pmacs')
    try:
        args = parser.parse_args(argv)
        execute(args.config, args.output, args.mode, args.execution_context)
    except Exception:
        print('Molecular run failed. Inspect the private error log if an output directory was created.')
        return 1
    print('Molecular run completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
