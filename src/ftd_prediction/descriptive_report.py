"""Private, offline figures from completed descriptive artifacts only."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import colorsys
import contextlib
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import traceback
import zipfile

import numpy as np

from .neural_report import _hash, _inside, _json
from .spatial import _labels


LAB_ORDER = ('RGC', 'AC', 'HC', 'BC', 'MG', 'Cone', 'Rod', 'Endothelial',
             'Astrocytes', 'Microglia')
_BASE = {'summary.json', 'private_cell_mapping.csv', 'pseudobulk_counts.npz',
         'pseudobulk_groups.csv', 'pseudobulk_genes.csv', 'private_execution.log'}
_GRAPH = {'spatial_edges.npz', 'neighborhood_fractions.npz', 'neighborhood_fractions.json'}
_OBS = {'cell_id', 'donor_id', 'sample_id', 'section_id', 'retinal_region',
        'acquisition_region', 'cell_type', 'layer', 'tissue_component', 'x_um', 'y_um'}
_NOTICE = ('Private descriptive report; assays remain separate. Cells/sections are not independent '
           'disease donors. No diagnoses, disease tests/CIs, expression/program analysis or density. '
           'This report grants no sharing permission.')
_NEIGHBOR = 'mean outgoing-neighbor fraction among non-isolated cells'


def _require(ok, message='Invalid descriptive artifacts'):
    if not ok:
        raise ValueError(message)


def _path(value):
    path = Path(value).expanduser()
    _require('..' not in path.parts, 'Unsafe path')
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        _require(not part.is_symlink(), 'Symlinks are forbidden')
    return path


def _read(path, limit=256 * 1024**2):
    _require(path.stat().st_size <= limit, 'Artifact exceeds reader byte budget')
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    _require(len(raw) <= limit, 'Artifact exceeds reader byte budget')
    return raw


def _csv(raw):
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8')), strict=True))
    _require(rows and rows[0] and len(set(rows[0])) == len(rows[0]), 'Invalid CSV header')
    _require(all(len(row) == len(rows[0]) for row in rows[1:]), 'Malformed CSV rows')
    return rows[0], [dict(zip(rows[0], row)) for row in rows[1:]]


def _positive(value):
    return type(value) is int and value > 0


def _label(value):
    _require(isinstance(value, str) and len(value) <= 256
             and not any(ord(c) < 32 for c in value), 'Invalid metadata label')
    return _labels([value])[0]


def _load(root):
    record_raw = _read(_inside(root, 'provenance/run.json'), 32 * 1024**2)
    record = _json(record_raw)
    _require(type(record.get('schema_version')) is int and record['schema_version'] == 1
             and record.get('status') == 'completed', 'Completed supported run required')
    kind = record.get('data_kind')
    _require(kind in ('synthetic', 'study'), 'Unknown data kind')
    _require(record.get('training_enabled') is False
             and record.get('disease_inference_enabled') is False, 'Descriptive-only run required')
    context = record.get('execution_context')
    _require(context in ('local', 'pmacs'), 'Unknown execution context')
    if kind == 'study':
        job = os.environ.get('LSB_JOBID', '')
        source_job = record.get('lsf_job_id', '')
        _require(context == 'pmacs' and isinstance(source_job, str)
                 and re.fullmatch('[0-9]+', source_job) and int(source_job) > 0
                 and re.fullmatch('[0-9]+', job) and int(job) > 0,
                 'Study reports require an active positive LSB_JOBID')
    config = record.get('configuration')
    _require(isinstance(config, dict) and config.get('data_kind') == kind
             and config.get('assay') in ('rna', 'spatial'), 'Invalid run configuration')
    for digest in (record.get('input_sha256'), record.get('configuration_sha256'),
                   record.get('code', {}).get('source_sha256')):
        _require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                 'Required provenance digest absent')
    graph = record.get('spatial_graph')
    composition = record.get('composition')
    _require(isinstance(graph, dict) and graph.get('status') in ('completed', 'not_requested'))
    _require(isinstance(composition, dict) and composition.get('status') in ('completed', 'unavailable'))
    has_graph = graph['status'] == 'completed'
    _require(has_graph == ('spatial_graph' in config))
    if has_graph:
        _require(config['assay'] == 'spatial' and config.get('coordinate_units') == 'um')
    expected = _BASE | (_GRAPH if has_graph else set())
    if composition['status'] == 'completed':
        expected |= {'composition.csv'}
    artifacts = record.get('artifacts')
    _require(isinstance(artifacts, list) and len(artifacts) == len(expected), 'Artifact inventory mismatch')
    snapshots, sources = {}, {}
    # Verify the ENTIRE inventory before interpreting any table/descriptor.
    for entry in artifacts:
        _require(isinstance(entry, dict) and set(entry) == {'path', 'sha256', 'bytes'})
        name, digest, size = entry['path'], entry['sha256'], entry['bytes']
        path = _inside(root, name)
        _require(name in expected and name not in sources, 'Unknown or duplicate artifact')
        _require(type(size) is int and size >= 0 and isinstance(digest, str)
                 and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid artifact integrity metadata')
        _require(path.stat().st_size == size and _hash(path) == digest, 'Artifact integrity failure')
        sources[name] = dict(entry)
        if name in {'summary.json', 'private_cell_mapping.csv', 'composition.csv'} | _GRAPH:
            raw = _read(path)
            _require(hashlib.sha256(raw).hexdigest() == digest, 'Artifact changed during verification')
            snapshots[name] = raw
    _require(set(sources) == expected, 'Missing required artifact')
    for path in root.rglob('*'):
        _path(path)
        name = path.relative_to(root).as_posix()
        _require(name in expected | {'provenance', 'provenance/run.json'}, 'Unlisted run entry')
    summary = _json(snapshots['summary.json'])
    _require(set(summary) == {'data_kind', 'evidence_scope', 'descriptive', 'composition',
                            'spatial_graph', 'disease_inference_enabled', 'training_enabled'})
    _require(summary['data_kind'] == kind and summary['training_enabled'] is False
             and summary['disease_inference_enabled'] is False
             and summary['evidence_scope'] == ('synthetic_pipeline_check' if kind == 'synthetic' else 'descriptive_only'))
    for key in ('descriptive', 'composition', 'spatial_graph'):
        _require(record.get(key) == summary[key], 'Run/summary mismatch')
    desc = summary['descriptive']
    _require(isinstance(desc, dict) and desc.get('disease_effect_inference') is False)
    _require(all(_positive(desc.get(k)) for k in ('n_cells', 'n_genes', 'n_donors', 'n_groups')))
    columns, rows = _csv(snapshots['private_cell_mapping.csv'])
    _require(set(columns) <= _OBS and {'cell_id', 'donor_id'} <= set(columns), 'Unknown mapping metadata')
    _require(len(rows) == desc['n_cells'] and len({r['cell_id'] for r in rows}) == len(rows))
    _require(all(r['cell_id'] and _label(r['donor_id']) is not None for r in rows))
    grouping = ['donor_id'] + [k for k in ('section_id', 'retinal_region', 'cell_type') if k in columns]
    _require(desc.get('grouping_columns') == grouping)
    normalized = [{k: _label(row[k]) for k in grouping} for row in rows]
    _require(len({r['donor_id'] for r in normalized}) == desc['n_donors'])
    _require(len({tuple(r[k] for k in grouping) for r in normalized}) == desc['n_groups'])
    annotated = 'cell_type' in columns
    _require(desc.get('cell_type_annotation_available') is annotated
             and (composition['status'] == 'completed') is annotated)
    labels = [r.get('cell_type') for r in normalized]
    if annotated:
        _composition(snapshots['composition.csv'], normalized, grouping, composition)
    coords = None
    if config['assay'] == 'spatial':
        _require({'section_id', 'x_um', 'y_um'} <= set(columns) and config.get('coordinate_units') == 'um')
        _require(all(r['section_id'] is not None for r in normalized), 'Missing physical section')
        assignments = {}
        for row in normalized:
            section, donor = row['section_id'], row['donor_id']
            _require(section not in assignments or assignments[section] == donor, 'Section crosses donors')
            assignments[section] = donor
        coords = np.asarray([[float(r['x_um']), float(r['y_um'])] for r in rows])
        _require(np.isfinite(coords).all(), 'Nonfinite coordinates')
    arrays = _graph(snapshots, graph, config, rows, normalized, labels, coords) if has_graph else None
    sources['provenance/run.json'] = {'path': 'provenance/run.json', 'bytes': len(record_raw),
                                     'sha256': hashlib.sha256(record_raw).hexdigest()}
    return dict(record=record, sources=sources, rows=normalized, labels=labels,
                coordinates=coords, arrays=arrays, annotated=annotated)


def _composition(raw, mapping, grouping, info):
    keys = [k for k in grouping if k != 'cell_type']
    flags = [('annotation_missing' if k == 'cell_type' else k + '_missing') for k in grouping]
    columns, rows = _csv(raw)
    _require(set(columns) == set(grouping + flags + ['n_cells', 'group_cells', 'fraction']))
    _require(info.get('grouping_columns') == keys and info.get('denominator') == 'all_loaded_cells_in_group'
             and info.get('missing_annotation_cells') == sum(r['cell_type'] is None for r in mapping))
    expected = Counter(tuple(r[k] for k in grouping) for r in mapping)
    totals = Counter(tuple(r[k] for k in keys) for r in mapping)
    seen = set()
    for row in rows:
        values = tuple(_label(row[k]) for k in grouping)
        _require(values not in seen and values in expected, 'Composition grouping mismatch')
        seen.add(values)
        for value, flag in zip(values, flags):
            _require(row[flag] == str(value is None), 'Missing-label flag mismatch')
        _require(re.fullmatch('[0-9]+', row['n_cells']) and re.fullmatch('[0-9]+', row['group_cells']))
        count, total = int(row['n_cells']), int(row['group_cells'])
        _require(count == expected[values] and total == totals[tuple(_label(row[k]) for k in keys)])
        fraction = float(row['fraction'])
        _require(math.isfinite(fraction) and math.isclose(fraction, count / total, rel_tol=1e-12, abs_tol=1e-12))
    _require(seen == set(expected), 'Composition omits loaded cells')


def _npz(raw, names):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        _require(len(archive.namelist()) == len(names)
                 and set(archive.namelist()) == {name + '.npy' for name in names}
                 and sum(z.file_size for z in archive.infolist()) <= 512 * 1024**2,
                 'Invalid or oversized NPZ archive')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in names}
    _require(all(a.dtype.kind in 'bifu' for a in arrays.values()), 'Numeric NPZ arrays required')
    return arrays


def _graph(raw, info, config, original, rows, labels, coords):
    _require(_json(raw['neighborhood_fractions.json']) == info, 'Graph JSON mismatch')
    categories = info.get('categories')
    expected = sorted({v for v in labels if v is not None}) + ([None] if None in labels else [])
    _require(categories == expected, 'Invalid graph category labels/order')
    _require(info.get('coordinate_units') == 'um' and info.get('directed') is True
             and info.get('fraction_denominator') == 'all_outgoing_neighbors_including_missing_labels'
             and info.get('mapping') == 'private_cell_mapping.csv; zero-based row order for both NPZ files'
             and info.get('n_unknown_cells') == labels.count(None)
             and info.get('cell_type_annotation_available') is ('cell_type' in original[0]))
    boundary = ['donor_id', 'section_id'] + (['tissue_component'] if 'tissue_component' in original[0] else [])
    _require(info.get('boundary_columns') == boundary and info.get('settings') == config['spatial_graph'])
    settings = info['settings']
    radius, cap = settings.get('radius_um'), settings.get('max_neighbors')
    _require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0 and _positive(cap))
    n, k = len(rows), len(categories)
    edges = _npz(raw['spatial_edges.npz'], {'edges'})['edges']
    _require(edges.dtype.kind in 'iu' and edges.ndim == 2 and edges.shape[0] == 2)
    _require(info.get('n_edges') == edges.shape[1] and (not edges.size or (edges.min() >= 0 and edges.max() < n)))
    _require(not np.any(edges[0] == edges[1]) and np.unique(edges, axis=1).shape[1] == edges.shape[1])
    for a, b in edges.T:
        _require(all(original[a][key] == original[b][key] for key in boundary), 'Cross-boundary edge')
    _require(np.all(np.linalg.norm(coords[edges[0]] - coords[edges[1]], axis=1) <= radius + 1e-9))
    names = {'counts', 'fractions', 'degree', 'is_isolate', 'unknown_label_count', 'annotation_missing'}
    arrays = _npz(raw['neighborhood_fractions.npz'], names)
    for name, array in arrays.items():
        _require(array.shape == ((n, k) if name in ('counts', 'fractions') else (n,)), 'NPZ row alignment mismatch')
        _require(array.dtype.kind in ('b' if name in ('is_isolate', 'annotation_missing') else
                                     'f' if name == 'fractions' else 'iu'))
    counts = np.zeros((n, k), dtype=np.int64)
    indices = {v: i for i, v in enumerate(categories)}
    if edges.shape[1]:
        target = np.asarray([indices[labels[b]] for b in edges[1]])
        np.add.at(counts, (edges[0], target), 1)
    degree = counts.sum(axis=1)
    fractions = np.divide(counts, degree[:, None], out=np.zeros((n, k)), where=degree[:, None] != 0)
    unknown = counts[:, indices[None]] if None in indices else np.zeros(n, dtype=np.int64)
    _require(np.all(degree <= cap) and np.array_equal(counts, arrays['counts'])
             and np.array_equal(degree, arrays['degree'])
             and np.array_equal(degree == 0, arrays['is_isolate'])
             and np.array_equal(unknown, arrays['unknown_label_count'])
             and np.array_equal([v is None for v in labels], arrays['annotation_missing'])
             and np.allclose(fractions, arrays['fractions'], rtol=1e-12, atol=1e-12),
             'Graph descriptors disagree with ordered mapping/edges')
    return arrays


def _ordered(values):
    observed = set(values)
    return [v for v in LAB_ORDER if v in observed] + sorted(observed - set(LAB_ORDER) - {None}) + ([None] if None in observed else [])


def _color(label):
    if label is None:
        return '#999999'
    fixed = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
             '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
    if label in LAB_ORDER:
        return fixed[LAB_ORDER.index(label)]
    hue = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) / 2**32
    return '#%02x%02x%02x' % tuple(round(c * 255) for c in colorsys.hsv_to_rgb(hue, .65, .65))


def _analyze(data):
    rows, labels = data['rows'], data['labels']
    donors = sorted({r['donor_id'] for r in rows})
    # Optional RNA section metadata may contain nulls. Physical-section panels
    # apply only to spatial runs; all RNA rows still enter donor/region pooling.
    sections = (sorted({(r['donor_id'], r['section_id']) for r in rows})
                if data['coordinates'] is not None else [])
    donor_alias = {v: f'Donor {i + 1:03d}' for i, v in enumerate(donors)}
    section_alias = {v: f'Section {i + 1:03d}' for i, v in enumerate(sections)}
    regions = sorted({r.get('retinal_region') for r in rows} - {None})
    region_alias = {v: f'Region {i + 1:03d}' for i, v in enumerate(regions)}
    region_alias[None] = 'Region missing / unavailable'
    categories = _ordered(labels)
    missing_name = 'Missing annotation (null)'
    while missing_name in categories:
        missing_name += ' [missing]'
    names = {v: v if v is not None else missing_name for v in categories}
    pooled = []
    # Count every loaded cell once; equivalently sum section n_cells, then divide.
    for donor in donors:
        donor_regions = {r.get('retinal_region') for r in rows if r['donor_id'] == donor}
        for region in sorted(donor_regions, key=lambda v: (v is None, v or '')):
            selected = [i for i, r in enumerate(rows) if r['donor_id'] == donor and r.get('retinal_region') == region]
            counts = Counter(labels[i] for i in selected)
            pooled.append(dict(donor=donor_alias[donor], region=region_alias[region], n_cells=len(selected),
                               counts=[counts[v] for v in categories],
                               fractions=[counts[v] / len(selected) for v in categories]))
    spatial = []
    for key in sections:
        indices = np.asarray([i for i, r in enumerate(rows) if (r['donor_id'], r['section_id']) == key], dtype=int)
        item = dict(donor=donor_alias[key[0]], section=section_alias[key], n_cells=len(indices), rows=indices.tolist())
        if data['arrays'] is not None:
            a = data['arrays']
            nonisolated = indices[~a['is_isolate'][indices]]
            means = []
            source_categories = data['record']['spatial_graph']['categories']
            for label in categories:
                focal = nonisolated[np.asarray([labels[i] == label for i in nonisolated], dtype=bool)]
                means.append([float(a['fractions'][focal, source_categories.index(v)].mean()) for v in categories]
                             if len(focal) else [None] * len(categories))
            item.update(n_nonisolated=len(nonisolated), n_isolates=len(indices) - len(nonisolated),
                        isolate_fraction=float(a['is_isolate'][indices].mean()),
                        mean_out_degree=float(a['degree'][indices].mean()),
                        max_out_degree=int(a['degree'][indices].max()), neighbor_means=means)
        spatial.append(item)
    aliases = dict(donors=[dict(alias=donor_alias[v], value=v) for v in donors],
                   sections=[dict(alias=section_alias[v], donor=donor_alias[v[0]], value=v[1]) for v in sections],
                   regions=[dict(alias=region_alias[v], value=v) for v in regions])
    return dict(categories=categories, category_names=[names[v] for v in categories],
                colors=[_color(v) for v in categories], pooled=pooled, sections=spatial, aliases=aliases)


def _write(path, value):
    with path.open('xb') as handle:
        handle.write(value.encode('utf-8') if isinstance(value, str) else value)


def _render(data, analysis, destination, config):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.patches import Patch

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = 'Arial' if 'Arial' in fonts else 'DejaVu Sans'
    categories, names = analysis['categories'], analysis['category_names']
    keep = list(range(len(categories)))
    omitted = []
    if len(keep) > config['max_categories']:
        keep = list(range(config['max_categories'] - 1 - (None in categories)))
        if None in categories:
            keep.append(categories.index(None))
        omitted = [i for i in range(len(categories)) if i not in keep]
    display_groups = [[i] for i in keep if categories[i] is not None] + ([omitted] if omitted else [])
    if None in categories:
        display_groups.append([categories.index(None)])
    aggregate_name = 'Combined omitted display categories'
    while aggregate_name in names:
        aggregate_name += ' [display aggregate]'
    shown_names = [aggregate_name if group == omitted else names[group[0]] for group in display_groups]
    colors = ['#444444' if group == omitted else analysis['colors'][group[0]] for group in display_groups]
    figures, display = [], dict(font=font, pdf_fonttype=42, aspect='equal',
                               displayed_category_names=shown_names,
                               category_groups=[[categories[i] for i in g] for g in display_groups],
                               omitted_category_count=len(omitted), maps=[], skipped_panels=0,
                               omitted_section_rows=max(0, len(analysis['sections']) - config['max_groups']),
                               omitted_pooled_groups=max(0, len(analysis['pooled']) - config['max_groups']))

    def save(fig, stem, caption):
        try:
            fig.tight_layout()
            for suffix in ('png', 'pdf'):
                with (destination / (stem + '.' + suffix)).open('xb') as handle:
                    fig.savefig(handle, format=suffix, dpi=150, bbox_inches='tight', pad_inches=.15,
                                metadata={'Creator': 'Private descriptive reporter'})
            figures.append(dict(stem=stem, caption=caption))
        finally:
            plt.close(fig)

    style = {'font.family': font, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'text.usetex': False, 'text.parse_math': False, 'font.size': 9}
    with matplotlib.rc_context(style):
        groups = analysis['pooled'][:config['max_groups']]
        group_names = [r['donor'] + ' / ' + r['region'] for r in groups]
        fig, ax = plt.subplots(figsize=(8, max(3, .3 * len(groups) + 1)))
        ax.barh(group_names, [r['n_cells'] for r in groups], color='#356e9d')
        ax.invert_yaxis()
        ax.set(xlabel='Loaded cells (all annotations, including missing)', title='Loaded-cell counts by donor / region')
        save(fig, 'loaded_cells', 'All loaded cells, including missing annotations/regions; sections pooled by cell count.')
        if data['annotated']:
            fig, ax = plt.subplots(figsize=(10, max(3.5, .3 * len(groups) + 1)))
            left = np.zeros(len(groups))
            for indices, name, color in zip(display_groups, shown_names, colors):
                widths = np.asarray([sum(r['fractions'][i] for i in indices) for r in groups])
                ax.barh(group_names, widths, left=left, color=color, label=name)
                left += widths
            ax.invert_yaxis()
            ax.set(xlim=(0, 1), xlabel='Fraction of all loaded cells', title='Composition by donor / region')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
            save(fig, 'composition', 'Denominator: all loaded cells, including missing. Pool sections by summing n_cells '
                 'then recomputing fractions; never average section fractions or sum repeated group_cells.')
        section_panels = analysis['sections'] if data['coordinates'] is not None else []
        for item in section_panels:
            if len(figures) >= config['max_panels']:
                display['skipped_panels'] += 1 + int(data['arrays'] is not None and data['annotated'])
                continue
            indices = np.asarray(item['rows'])
            selected = indices[np.linspace(0, len(indices) - 1, min(len(indices), config['max_cells_per_map']), dtype=int)]
            display['maps'].append(dict(section=item['section'], total_cells=len(indices),
                                        displayed_cells=len(selected), selected_mapping_rows=selected.tolist(),
                                        selection='evenly spaced ordered mapping rows; display only'))
            fig, ax = plt.subplots(figsize=(9, 5))
            for group, name, color in zip(display_groups, shown_names, colors):
                members = {categories[i] for i in group}
                cell_rows = [i for i in selected if data['labels'][i] in members]
                if cell_rows:
                    xy = data['coordinates'][cell_rows]
                    ax.scatter(xy[:, 0], xy[:, 1], s=3, c=color, linewidths=0)
            # Limits use all section cells, even if display has been sampled.
            xy = data['coordinates'][indices]
            for setter, values in ((ax.set_xlim, xy[:, 0]), (ax.set_ylim, xy[:, 1])):
                lo, hi = values.min(), values.max()
                pad = max((hi - lo) * .03, .5)
                setter(lo - pad, hi + pad)
            ax.set_aspect('equal', adjustable='box')
            ax.set(xlabel='x (um)', ylabel='y (um)', title=f"{item['donor']} / {item['section']} — {len(selected)}/{len(indices)} cells displayed")
            ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, shown_names)],
                      loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
            save(fig, item['section'].lower().replace(' ', '_') + '_map',
                 'Equal x/y scale in um, separate section. '
                 f'{len(selected)} of {len(indices)} cells displayed using deterministic ordered-row selection. '
                 'Analyses use all cells; no density.')
            if data['arrays'] is None or not data['annotated']:
                continue
            if item['n_nonisolated'] == 0:
                display['skipped_panels'] += 1
                continue
            if len(figures) >= config['max_panels']:
                display['skipped_panels'] += 1
                continue
            # Aggregate display categories from full per-cell fractions, conditioning on non-isolates.
            valid = indices[~data['arrays']['is_isolate'][indices]]
            source = data['record']['spatial_graph']['categories']
            matrix = np.full((len(display_groups), len(display_groups)), np.nan)
            for i, focal_group in enumerate(display_groups):
                focal = [v for v in valid if data['labels'][v] in {categories[j] for j in focal_group}]
                if focal:
                    for j, target_group in enumerate(display_groups):
                        columns = [source.index(categories[v]) for v in target_group]
                        matrix[i, j] = data['arrays']['fractions'][np.ix_(focal, columns)].sum(axis=1).mean()
            fig, ax = plt.subplots(figsize=(max(5, .28 * len(shown_names) + 3), max(4, .28 * len(shown_names) + 2)))
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#dddddd')
            mesh = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect='equal')
            ax.set_xticks(range(len(shown_names)), shown_names, rotation=90)
            ax.set_yticks(range(len(shown_names)), shown_names)
            ax.set(xlabel='Outgoing-neighbor annotation', ylabel='Focal annotation', title=item['section'])
            fig.colorbar(mesh, ax=ax, label='Mean fraction')
            save(fig, item['section'].lower().replace(' ', '_') + '_neighbors',
                 _NEIGHBOR + '. Rows: focal; columns: neighbor (missing retained). Gray: no eligible focal cells. '
                 'Equal focal-cell weight; isolates excluded. Directed section/component graph; no tissue masks.')
    return figures, display


def _documents(data, analysis, figures, display, config, destination):
    escape = lambda value: html.escape(str(value), quote=True)
    md = lambda value: re.sub(r'([\\`*\[\]{}()_])', r'\\\1', escape(value)).replace('|', '&#124;')
    notices = [_NOTICE, f"Data kind: {data['record']['data_kind']}; assay: {data['record']['configuration']['assay']}.",
               f"PDF editable text: {display['font']} (Arial preferred, DejaVu Sans fallback), pdf.fonttype=42.",
               'Annotation missing cells: ' + str(data['labels'].count(None)),
               'Retinal region metadata absent.' if 'retinal_region' not in data['rows'][0] else 'Missing regions retain their cell mass.',
               'Graph not requested; neighbor panels unavailable.' if data['arrays'] is None else _NEIGHBOR + '.',
               'Cell-type annotation absent; composition/neighbor heatmaps unavailable.' if not data['annotated'] else 'Observed names retained; no synonym mapping.',
               'Spatial maps unavailable for RNA.' if data['coordinates'] is None else 'Physical sections are rendered separately; axes have equal scale in um.',
               f"Rendering limits: {json.dumps(config, sort_keys=True)}. Omitted pooled display groups: {display['omitted_pooled_groups']}; "
               f"combined categories: {display['omitted_category_count']}; skipped panels: {display['skipped_panels']}. "
               f"Omitted section table rows: {display['omitted_section_rows']}. "
               'Display limits only; full analyses/selections: report_provenance.json.',
               'Ordinal donor/section/region aliases are stable within this run; private provenance retains raw mappings.']
    markdown = ['# Private descriptive report', ''] + [md(v) + '\n' for v in notices]
    body = ['<h1>Private descriptive report</h1>'] + ['<p>' + escape(v) + '</p>' for v in notices]
    body.append('<p>Display categories: ' + ', '.join(escape(v) for v in display['displayed_category_names']) + '</p>')
    markdown.append('Display categories: ' + ', '.join(md(v) for v in display['displayed_category_names']))
    for figure in figures:
        stem, caption = figure['stem'], figure['caption']
        encoded = base64.b64encode((destination / (stem + '.png')).read_bytes()).decode('ascii')
        body.append(f'<figure><img alt="{escape(caption)}" src="data:image/png;base64,{encoded}">'
                    f'<figcaption>{escape(caption)} <a href="{stem}.pdf">Editable PDF</a></figcaption></figure>')
        markdown.extend(['', f'![{md(caption)}]({stem}.png)', '', md(caption), f'[Editable PDF]({stem}.pdf)'])
    if data['arrays'] is not None:
        unavailable = sum(s['n_nonisolated'] == 0 for s in analysis['sections'])
        note = f'{unavailable} sections have no non-isolated cells; their neighbor heatmaps are unavailable.'
        body.append('<p>' + note + '</p>')
        markdown.extend(['', note])
        body.append('<h2>Outgoing degree and isolates</h2><p>Mean outgoing degree includes isolates; no area normalization.</p>')
        markdown.extend(['', '## Outgoing degree and isolates', '', 'Mean outgoing degree includes isolates; no area normalization.', '',
                         '| Donor | Section | Cells | Non-isolated | Isolates | Isolate fraction | Mean / max outgoing degree |',
                         '| --- | --- | --- | --- | --- | --- | --- |'])
        body.append('<table><tr><th>Donor / section</th><th>Cells</th><th>Non-isolated</th><th>Isolates</th><th>Isolate fraction</th><th>Mean / max outgoing degree</th></tr>')
        for item in analysis['sections'][:config['max_groups']]:
            values = [item['donor'], item['section'], item['n_cells'], item['n_nonisolated'], item['n_isolates'],
                      f"{item['isolate_fraction']:.4f}", f"{item['mean_out_degree']:.4f} / {item['max_out_degree']}"]
            markdown.append('| ' + ' | '.join(md(v) for v in values) + ' |')
            body.append('<tr>' + ''.join('<td>' + escape(v) + '</td>' for v in [values[0] + ' / ' + values[1]] + values[2:]) + '</tr>')
        body.append('</table>')
    _write(destination / 'report.md', '\n'.join(markdown) + '\n')
    _write(destination / 'report.html', '<!doctype html><html lang="en"><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Private descriptive report</title>'
           '<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:1rem;line-height:1.5}'
           'img{max-width:100%}td,th{padding:.4rem;border:1px solid #bbb}table{border-collapse:collapse}'
           'p,figcaption{overflow-wrap:anywhere}</style><body>' + ''.join(body) + '</body></html>')


def build_report(run, output, *, max_cells_per_map=5000, max_panels=24, max_categories=24, max_groups=30):
    """Create a new private report. Study artifacts require the real LSF environment guard."""
    config = dict(max_cells_per_map=max_cells_per_map, max_panels=max_panels,
                  max_categories=max_categories, max_groups=max_groups)
    for key, bound in (('max_cells_per_map', 50000), ('max_panels', 100), ('max_categories', 64), ('max_groups', 100)):
        _require(_positive(config[key]) and config[key] <= bound, 'Invalid rendering budget')
    _require(max_panels >= 2 and max_categories >= 3, 'Rendering budgets too small')
    root, destination = _path(run), _path(output)
    _require(root.is_dir() and destination.parent.is_dir(), 'Existing run and output parent required')
    _require(not destination.is_relative_to(root) and not root.is_relative_to(destination), 'Output overlaps source run')
    _require(not os.path.lexists(destination), 'Output must be new')
    old_umask = os.umask(0o077)
    try:
        destination.mkdir(mode=0o700, exist_ok=False)
        with (destination / 'private_error.log').open('x', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    data = _load(root)
                    analysis = _analyze(data)
                    # Keep matplotlib caches within this private output, not a shared home.
                    with contextlib.ExitStack() as stack:
                        previous = os.environ.get('MPLCONFIGDIR')
                        os.environ['MPLCONFIGDIR'] = str(destination / '.matplotlib')
                        def restore():
                            if previous is None:
                                os.environ.pop('MPLCONFIGDIR', None)
                            else:
                                os.environ['MPLCONFIGDIR'] = previous
                        stack.callback(restore)
                        figures, display = _render(data, analysis, destination, config)
                    _documents(data, analysis, figures, display, config, destination)
                    # Catch source mutation during rendering before declaring completion.
                    for name, entry in data['sources'].items():
                        path = _inside(root, name)
                        _require(path.stat().st_size == entry['bytes'] and _hash(path) == entry['sha256'], 'Source changed during report')
                    log.flush()
                    outputs = [{'path': p.relative_to(destination).as_posix(), 'sha256': _hash(p), 'bytes': p.stat().st_size}
                               for p in sorted(destination.rglob('*')) if p.is_file()]
                    provenance = dict(schema_version=1, status='completed', data_kind=data['record']['data_kind'],
                                      assay=data['record']['configuration']['assay'], training_enabled=False,
                                      disease_inference_enabled=False, configuration=config,
                                      source_artifacts=list(data['sources'].values()),
                                      source_configuration=data['record']['configuration'],
                                      source_code=data['record']['code'], reporter_sha256=_hash(Path(__file__)),
                                      spatial_helper_sha256=_hash(Path(__file__).with_name('spatial.py')),
                                      integrity_helper_sha256=_hash(Path(__file__).with_name('neural_report.py')),
                                      analysis=analysis, rendering=display, figures=figures, outputs=outputs)
                    _write(destination / 'report_provenance.json', json.dumps(provenance, indent=2, allow_nan=False) + '\n')
                except Exception:
                    traceback.print_exc(file=log)
                    raise
    finally:
        os.umask(old_umask)
    return destination


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid report arguments')


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-cells-per-map', type=int, default=5000)
    parser.add_argument('--max-panels', type=int, default=24)
    parser.add_argument('--max-categories', type=int, default=24)
    parser.add_argument('--max-groups', type=int, default=30)
    try:
        args = vars(parser.parse_args(argv))
        build_report(**args)
    except Exception:
        print('Descriptive report failed. Inspect the private error log if a report directory was created.')
        return 1
    print('Private descriptive report created.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
