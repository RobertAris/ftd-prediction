"""Explicit sparse H5AD input and metadata-only schema inspection.

Study callers must establish PMACS/runtime scope before calling these functions.
This module never guesses count semantics, donor identities or section mappings.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _hard_node(root, path):
    import h5py
    node = root
    for part in path.strip('/').split('/'):
        if not part or part in ('.', '..') or not isinstance(node, h5py.Group):
            raise ValueError('Unsupported HDF5 path')
        if not isinstance(node.get(part, getlink=True), h5py.HardLink):
            raise ValueError('HDF5 external or soft links are unsupported')
        node = node[part]
    return node


def _matrix_schema(node):
    import h5py
    result = {'encoding': _text(node.attrs.get('encoding-type', 'missing'))}
    if isinstance(node, h5py.Dataset):
        result.update(shape=list(node.shape), dtype=str(node.dtype),
                      external_storage=bool(node.external or node.is_virtual))
    elif result['encoding'] in ('csr_matrix', 'csc_matrix'):
        result['shape'] = [int(x) for x in node.attrs['shape']]
        data = _hard_node(node, 'data')
        result.update(dtype=str(data.dtype), stored_entries=int(data.shape[0]))
    else:
        result['supported_matrix'] = False
    return result


def inspect_schema(path):
    """Read HDF5 keys/whitelisted attributes/shapes only, never dataset values."""
    import h5py
    with h5py.File(path, 'r') as f:
        for key in ('X', 'raw', 'layers', 'obs', 'var', 'obsm'):
            if key in f:
                _hard_node(f, key)
        result = {'format': 'h5ad', 'matrices': {}, 'obs_columns': [],
                  'var_columns': [], 'obsm_keys': []}
        for key in ('X', 'raw/X'):
            if key in f:
                result['matrices'][key] = _matrix_schema(_hard_node(f, key))
        if 'layers' in f:
            layers = _hard_node(f, 'layers')
            for key in layers.keys():
                result['matrices']['layers/' + key] = _matrix_schema(_hard_node(layers, key))
        for key in ('obs', 'var'):
            group = _hard_node(f, key)
            result[key + '_columns'] = [_text(x) for x in group.attrs.get('column-order', [])]
        if 'raw/var' in f:
            result['raw_var_columns'] = [_text(x) for x in _hard_node(f, 'raw/var').attrs.get('column-order', [])]
        if 'obsm' in f:
            obsm = _hard_node(f, 'obsm')
            result['obsm_keys'] = list(obsm.keys())
            if 'spatial' in obsm:
                result['spatial_schema'] = _matrix_schema(_hard_node(obsm, 'spatial'))
        return result


def _safe_read_elem(node):
    """Reject external HDF5 storage before reading an explicitly chosen field."""
    import h5py
    from anndata.io import read_elem
    if isinstance(node, h5py.Dataset):
        if node.external or node.is_virtual:
            raise ValueError('External dataset storage is unsupported')
    else:
        for key in node.keys():
            child = _hard_node(node, key)
            # These groups are shallow AnnData categorical/nullable or sparse encodings.
            if isinstance(child, h5py.Dataset) and (child.external or child.is_virtual):
                raise ValueError('External dataset storage is unsupported')
            if isinstance(child, h5py.Group):
                raise ValueError('Unsupported nested field encoding')
    return read_elem(node)


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'Invalid {name} bound')
    return value


def _apply_mapping(frame, config):
    """Join a private, explicitly verified sample/region crosswalk without guessing."""
    import pandas as pd
    path = config.get('mapping_file')
    if path is None:
        if config.get('mapping_keys') is not None:
            raise ValueError('Mapping keys require a mapping file')
        return frame
    keys = config.get('mapping_keys')
    if (not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys)
            or len(keys) != len(set(keys)) or not set(keys).issubset(frame.columns)
            or set(keys) - {'sample_id', 'retinal_region', 'acquisition_region'}):
        raise ValueError('Mapping keys must explicitly reference selected sample/region columns')
    path = Path(path)
    if not path.is_absolute() or path.stat().st_size > 10 * 1024**2:
        raise ValueError('Mapping file must be an absolute bounded private CSV')
    mapping = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=100001)
    targets = set(mapping.columns) - set(keys)
    if (len(mapping) > 100000 or not targets or not set(keys).issubset(mapping.columns)
            or targets - {'donor_id', 'section_id', 'tissue_component'}
            or targets & set(frame.columns)):
        raise ValueError('Mapping columns are missing, unsupported or would replace existing columns')
    for column in mapping:
        values = mapping[column]
        if values.str.strip().eq('').any() or not values.eq(values.str.strip()).all():
            raise ValueError('Mapping fields must be explicit nonempty strings')
    if mapping.duplicated(keys).any():
        raise ValueError('Mapping keys are ambiguous')
    if frame[keys].isna().any().any() or any(frame[k].astype(str).str.strip().eq('').any() for k in keys):
        raise ValueError('Observations contain missing mapping keys')
    lookup = mapping.set_index(keys)
    query = pd.MultiIndex.from_frame(frame[keys].astype(str)) if len(keys) > 1 else pd.Index(frame[keys[0]].astype(str))
    joined = lookup.reindex(query)
    if joined.isna().any().any():
        raise ValueError('Mapping does not cover every observation key')
    for column in sorted(targets):
        frame[column] = joined[column].to_numpy()
    return frame


def load_h5ad(path, config):
    """Load an explicit counts source and selected metadata, keeping counts sparse.

    Optional per-donor sampling is deterministic and recorded, for bounded pilots.
    It is sampling within the existing donors, never additional replication.
    """
    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    from anndata.io import sparse_dataset
    from scipy import sparse

    count_source = config.get('count_source')
    if not isinstance(count_source, str) or not (
        count_source in ('X', 'raw/X') or
        (count_source.startswith('layers/') and count_source.count('/') == 1 and count_source[7:])
    ):
        raise ValueError('An explicit X, raw/X or exact layers/name count source is required')
    provenance = config.get('counts_provenance', {})
    if provenance.get('kind') not in ('measured_umi', 'background_corrected_umi') or not provenance.get('evidence_ref'):
        raise ValueError('Verified unnormalized count provenance is required')
    if config.get('mapping_verified') is not True or not config.get('mapping_evidence_ref'):
        raise ValueError('Verified donor/section mapping evidence is required')
    columns = config.get('obs_columns')
    if not isinstance(columns, dict) or ('donor_id' not in columns and config.get('mapping_file') is None):
        raise ValueError('An explicit donor metadata column is required')
    permitted = {'donor_id', 'sample_id', 'section_id', 'retinal_region', 'acquisition_region', 'cell_type', 'layer', 'tissue_component'}
    if set(columns) - permitted or any(not isinstance(v, str) or not v or '/' in v for v in columns.values()):
        raise ValueError('Unsupported observation-column mapping')
    spatial = config.get('assay') == 'spatial'
    if config.get('assay') not in ('rna', 'spatial'):
        raise ValueError('Unsupported assay')
    if spatial and (('section_id' not in columns and config.get('mapping_file') is None) or config.get('coordinate_units') != 'um'):
        raise ValueError('Spatial input requires verified sections and micron coordinates')
    max_source_cells = _positive_int(config.get('max_source_cells', 2_000_000), 'source-cell')
    max_cells = _positive_int(config.get('max_cells', 500_000), 'cell')
    max_genes = _positive_int(config.get('max_genes', 100_000), 'gene')
    max_bytes = _positive_int(config.get('max_matrix_bytes', 4 * 1024**3), 'matrix-byte')
    cap = config.get('cell_cap_per_donor')
    if cap is not None:
        _positive_int(cap, 'per-donor cell')
        if spatial:
            raise ValueError('Random cell thinning is unsupported for spatial graphs; use complete sections')
    seed = config.get('seed', 11)
    if type(seed) is not int or seed < 0:
        raise ValueError('Invalid sampling seed')

    with h5py.File(path, 'r') as f:
        matrix = _hard_node(f, count_source)
        schema = _matrix_schema(matrix)
        shape = schema.get('shape', [])
        if len(shape) != 2 or not 0 < shape[0] <= max_source_cells or not 0 < shape[1] <= max_genes:
            raise ValueError('Source matrix exceeds the declared shape bounds')
        obs = _hard_node(f, 'obs')
        index_key = _text(obs.attrs.get('_index', '_index'))
        cell_ids = pd.Index(_safe_read_elem(_hard_node(obs, index_key)).astype(str))
        if len(cell_ids) != shape[0] or not cell_ids.is_unique:
            raise ValueError('Cell identifiers are missing, duplicated or misaligned')
        frame = pd.DataFrame(index=cell_ids)
        for target, source in columns.items():
            values = _safe_read_elem(_hard_node(obs, source))
            if len(values) != shape[0]:
                raise ValueError('Observation column length does not match counts')
            frame[target] = values
        frame = _apply_mapping(frame, config)
        for required in ('donor_id', 'section_id') if spatial else ('donor_id',):
            if required not in frame:
                raise ValueError('Biological mapping is incomplete')
            values = frame[required]
            if values.isna().any() or values.astype(str).str.strip().isin(['', 'unknown', 'unresolved', 'nan', 'None']).any():
                raise ValueError('Required biological mapping contains unresolved values')
            frame[required] = values.astype(str)
        if spatial and frame.groupby('section_id', observed=True)['donor_id'].nunique().gt(1).any():
            raise ValueError('A physical section maps to multiple donors')
        rows = np.arange(shape[0], dtype=np.int64)
        if cap is not None:
            chunks = []
            for donor in sorted(frame['donor_id'].unique()):
                # Query donors must not change the sampled training cells.
                donor_seed = hashlib.sha256(f'{seed}:{donor}'.encode('utf-8')).digest()
                rng = np.random.default_rng(int.from_bytes(donor_seed[:16], 'big'))
                indices = np.flatnonzero(frame['donor_id'].to_numpy() == donor)
                chunks.append(rng.choice(indices, min(cap, len(indices)), replace=False))
            rows = np.sort(np.concatenate(chunks))
        if len(rows) > max_cells:
            raise ValueError('Selected cell count exceeds the declared bound')

        var = _hard_node(f, 'raw/var' if count_source == 'raw/X' else 'var')
        genes = pd.Index(_safe_read_elem(_hard_node(var, _text(var.attrs.get('_index', '_index')))).astype(str))
        if len(genes) != shape[1] or not genes.is_unique or (genes.str.len() == 0).any():
            raise ValueError('Gene identifiers are empty, duplicated or misaligned')
        encoding = schema['encoding']
        if encoding in ('csr_matrix', 'csc_matrix'):
            for key in ('data', 'indices', 'indptr'):
                node = _hard_node(matrix, key)
                if node.external or node.is_virtual:
                    raise ValueError('External sparse storage is unsupported')
            data, indices, indptr = (matrix[key] for key in ('data', 'indices', 'indptr'))
            if data.dtype.kind not in 'iuf' or indices.dtype.kind not in 'iu' or indptr.dtype.kind not in 'iu':
                raise ValueError('Unsupported sparse matrix dtypes')
            axis = shape[0] if encoding == 'csr_matrix' else shape[1]
            if len(indptr) != axis + 1 or len(data) != len(indices):
                raise ValueError('Malformed sparse matrix structure')
            entries = len(data)
            if encoding == 'csr_matrix':
                starts, ends = indptr[rows], indptr[rows + 1]
                if np.any(starts < 0) or np.any(ends < starts) or np.any(ends > len(data)):
                    raise ValueError('Malformed sparse row pointers')
                entries = int(np.sum(ends - starts, dtype=np.int64))
            estimated = entries * (data.dtype.itemsize + indices.dtype.itemsize) + (len(rows) + 1) * indptr.dtype.itemsize
            if estimated > max_bytes:
                raise ValueError('Selected sparse matrix exceeds the declared memory bound')
            # For CSC, the conservative estimate above bounds its full stored arrays.
            counts = sparse_dataset(matrix, should_cache_indptr=False)[rows, :].tocsr()
        elif isinstance(matrix, h5py.Dataset) and matrix.dtype.kind in 'iuf':
            if matrix.external or matrix.is_virtual:
                raise ValueError('External dense storage is unsupported')
            if len(rows) * shape[1] * matrix.dtype.itemsize > max_bytes:
                raise ValueError('Selected dense matrix exceeds the declared memory bound')
            counts = sparse.csr_matrix(matrix[rows, :])
        else:
            raise ValueError('Unsupported counts matrix encoding')
        counts.check_format(full_check=True)
        if not np.isfinite(counts.data).all() or np.any(counts.data < 0) or not np.equal(counts.data, np.floor(counts.data)).all():
            raise ValueError('Selected counts are not finite nonnegative integers')
        counts.sum_duplicates()
        counts.eliminate_zeros()
        if np.any(np.asarray(counts.sum(axis=1)).ravel() <= 0):
            raise ValueError('Zero-library cells require explicit preprocessing before training')
        a = ad.AnnData(X=counts.copy(), obs=frame.iloc[rows].copy(), var=pd.DataFrame(index=genes))
        a.layers['counts'] = counts
        if spatial:
            key = config.get('spatial_key')
            if not isinstance(key, str) or not key or '/' in key:
                raise ValueError('An explicit spatial coordinate key is required')
            node = _hard_node(f, 'obsm/' + key)
            if not isinstance(node, h5py.Dataset) or node.shape != (shape[0], 2) or node.dtype.kind not in 'iuf' or node.external or node.is_virtual:
                raise ValueError('Coordinates must be a numeric cells-by-two local dataset')
            xy = node[rows, :]
            if not np.isfinite(xy).all():
                raise ValueError('Coordinates contain nonfinite values')
            a.obsm['spatial'] = xy
        a.uns['ftd_input'] = {'source_shape': list(shape), 'selected_cells': len(rows),
            'sampling_seed': seed, 'cell_cap_per_donor': cap,
            'sampling_algorithm': 'independent-donor-sha256-pcg64-v1',
            'count_source': count_source, 'counts_provenance': dict(provenance),
            'mapping_evidence_ref': config['mapping_evidence_ref']}
        return a
