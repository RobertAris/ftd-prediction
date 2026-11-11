"""Invented sparse data only; descriptive arithmetic, locality and private scope."""
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ftd_prediction import descriptive as runner
from ftd_prediction.provenance import file_sha256


class DescriptiveTests(unittest.TestCase):
    def fixture(self, root, spatial=False, crosswalk=False):
        inputs = root / 'inputs'
        inputs.mkdir()
        obs = pd.DataFrame({'donor': ['d1'] * 6 + ['d2'] * 2,
            'section': ['s1'] * 4 + ['s2'] * 2 + ['s3'] * 2,
            'region': ['central'] * 4 + ['peripheral'] * 2 + [None] * 2,
            'type': ['A', None, 'B', 'A', 'A', 'A', 'B', 'B'],
            'component': ['u', 'u', 'u', 'v', 'u', 'u', 'u', 'u']},
            index=[f'c{i}' for i in range(8)])
        matrix = sparse.csr_matrix([[1, 0], [0, 2], [3, 4], [5, 0],
                                    [6, 7], [8, 9], [10, 11], [12, 13]])
        a = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=['g1', 'g2']))
        a.layers['counts'] = matrix.copy()
        a.obsm['spatial'] = np.array([[0, 0], [.1, 0], [.2, 0], [.05, 0],
                                      [0, 0], [1, 0], [0, 0], [1, 0]])
        source = inputs / 'invented.h5ad'
        a.write_h5ad(source)
        cfg = dict(schema_version=1, data_kind='synthetic', assay='spatial' if spatial else 'rna',
            input_file=str(source), canonical_input_roots=[str(inputs)], count_source='layers/counts',
            counts_provenance=dict(kind='measured_umi', evidence_ref='invented-fixture'),
            mapping_verified=True, mapping_evidence_ref='invented-fixture', seed=11,
            obs_columns=dict(donor_id='donor', section_id='section', retinal_region='region', cell_type='type'))
        if spatial:
            cfg.update(coordinate_units='um', spatial_key='spatial', spatial_graph=dict(radius_um=2, max_neighbors=1))
            cfg['obs_columns']['tissue_component'] = 'component'
        if crosswalk:
            cfg['obs_columns'].pop('donor_id')
            cfg['obs_columns']['sample_id'] = 'donor'
            mapping = inputs / 'mapping.csv'
            mapping.write_text('sample_id,donor_id\nd1,d1\nd2,d2\n')
            cfg.update(mapping_file=str(mapping), mapping_keys=['sample_id'])
        path = root / 'config.json'
        path.write_text(json.dumps(cfg))
        return path, cfg

    def run_config(self, path, cfg, output, **kwargs):
        path.write_text(json.dumps(cfg))
        return runner.execute(path, output, **dict(mode='synthetic', execution_context='local', **kwargs))

    def record(self, output):
        return json.loads((output / 'provenance/run.json').read_text())

    def test_decoded_transcript_provenance_survives_spatial_description(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path, cfg = self.fixture(root, spatial=True)
            cfg['counts_provenance'] = dict(
                kind='measured_transcripts', evidence_ref='invented-decoding-and-assignment')
            self.run_config(path, cfg, root / 'out')
            self.assertEqual(self.record(root / 'out')['status'], 'completed')
            self.assertEqual(self.record(root / 'out')['input_selection']
                             ['counts_provenance']['kind'], 'measured_transcripts')
            cfg['assay'] = 'rna'
            cfg.pop('spatial_graph')
            with patch.object(runner, 'load_h5ad') as load:
                with self.assertRaisesRegex(ValueError, 'count provenance'):
                    self.run_config(path, cfg, root / 'wrong-assay')
                load.assert_not_called()

    def test_sparse_rna_arithmetic_provenance_and_no_backends(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root, crosswalk=True)
            hashes = {p: file_sha256(p) for p in (path, Path(cfg['input_file']), Path(cfg['mapping_file']))}
            original_import = __import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('scvi', 'novae') or name.endswith(('rna_training', 'spatial_training')):
                    raise AssertionError('Training backend imported')
                return original_import(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=guarded), \
                    patch('ftd_prediction.neural._validate_environment', side_effect=AssertionError('backend versions')), \
                    patch.object(sparse.csr_matrix, 'toarray', side_effect=AssertionError('densification')), \
                    patch.object(sparse.csc_matrix, 'toarray', side_effect=AssertionError('densification')):
                out = runner.execute(path, root / 'out', mode='synthetic', execution_context='local')
            groups = pd.read_csv(out / 'pseudobulk_groups.csv')
            bulk = sparse.load_npz(out / 'pseudobulk_counts.npz').toarray()
            expected = {('s1', 'A'): [6, 0], ('s1', 'B'): [3, 4],
                        ('s1', None): [0, 2], ('s2', 'A'): [14, 16], ('s3', 'B'): [22, 24]}
            for row, values in zip(groups.to_dict('records'), bulk):
                label = None if row['annotation_missing'] else row['cell_type']
                np.testing.assert_array_equal(values, expected[(row['section_id'], label)])
            comp = pd.read_csv(out / 'composition.csv')
            first = comp[comp.section_id == 's1']
            self.assertEqual(sorted(first.n_cells), [1, 1, 2])
            self.assertEqual(sorted(first.fraction), [.25, .25, .5])
            self.assertTrue(first.group_cells.eq(4).all())
            self.assertEqual(comp.loc[comp.annotation_missing, 'n_cells'].tolist(), [1])
            self.assertEqual(comp.groupby('donor_id').n_cells.sum().to_dict(), {'d1': 6, 'd2': 2})
            self.assertTrue(comp.loc[comp.section_id == 's3', 'retinal_region'].isna().all())
            self.assertEqual(pd.read_csv(out / 'pseudobulk_genes.csv').gene_id.tolist(), ['g1', 'g2'])
            self.assertEqual(pd.read_csv(out / 'private_cell_mapping.csv').cell_id.tolist(), [f'c{i}' for i in range(8)])
            record = self.record(out)
            self.assertEqual((record['status'], record['seed'], record['spatial_graph']['status']), ('completed', 11, 'not_requested'))
            self.assertFalse((out / 'spatial_edges.npz').exists())
            self.assertEqual(record['configuration_sha256'], hashes[path])
            self.assertEqual({v['path']: v['sha256'] for v in record['input_files']},
                             {str(p.resolve()): h for p, h in hashes.items() if p != path})
            self.assertTrue(record['code']['source_sha256'])
            self.assertIn('anndata', record['environment']['packages'])
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            for artifact in record['artifacts']:
                self.assertEqual(artifact['sha256'], file_sha256(out / artifact['path']))
            self.assertEqual(hashes, {p: file_sha256(p) for p in hashes})

    def test_null_literal_sentinel_and_unresolved_annotations_share_one_policy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root, spatial=True)
            source = Path(cfg['input_file'])
            a = ad.read_h5ad(source)
            a.obs['type'] = pd.Categorical([None, '__unavailable__', 'UNKNOWN', ' Unresolved ', 'A', 'A', 'B', 'B'])
            a.obs['region'] = pd.Categorical(['central'] * 4 + ['__unavailable__'] * 2 + [None, '__unavailable__'])
            a.write_h5ad(source)
            original_hash = file_sha256(source)
            with patch.object(sparse.csr_matrix, 'toarray', side_effect=AssertionError('densification')), \
                    patch.object(sparse.csc_matrix, 'toarray', side_effect=AssertionError('densification')):
                out = self.run_config(path, cfg, root / 'out')
            groups = pd.read_csv(out / 'pseudobulk_groups.csv', keep_default_na=False)
            comp = pd.read_csv(out / 'composition.csv', keep_default_na=False)
            bulk = sparse.load_npz(out / 'pseudobulk_counts.npz')
            self.assertTrue(sparse.isspmatrix_csr(bulk))
            signature = ['donor_id', 'section_id', 'retinal_region', 'retinal_region_missing',
                         'cell_type', 'annotation_missing']
            expected = {
                ('d1', 's1', 'central', False, '', True): ([9, 4], 3),
                ('d1', 's1', 'central', False, '__unavailable__', False): ([0, 2], 1),
                ('d1', 's2', '__unavailable__', False, 'A', False): ([14, 16], 2),
                ('d2', 's3', '', True, 'B', False): ([10, 11], 1),
                ('d2', 's3', '__unavailable__', False, 'B', False): ([12, 13], 1),
            }
            self.assertEqual(len(groups), len(expected))
            self.assertEqual(set(groups[signature].itertuples(index=False, name=None)), set(expected))
            for index, row in groups.iterrows():
                values, n_cells = expected[tuple(row[signature])]
                np.testing.assert_array_equal(bulk.getrow(index).toarray().ravel(), values)
                self.assertEqual(row.n_cells, n_cells)
            self.assertEqual(comp.set_index(signature).n_cells.to_dict(),
                             {key: value[1] for key, value in expected.items()})
            first = comp[comp.section_id == 's1']
            self.assertEqual(sorted(first.fraction), [.25, .75])
            self.assertEqual(self.record(out)['composition']['missing_annotation_cells'], 3)
            info = json.loads((out / 'neighborhood_fractions.json').read_text())
            self.assertEqual(info['categories'], ['A', 'B', '__unavailable__', None])
            self.assertEqual(info['n_unknown_cells'], 3)
            with np.load(out / 'neighborhood_fractions.npz', allow_pickle=False) as z:
                self.assertEqual(z['annotation_missing'].tolist(), [True, False, True, True, False, False, False, False])
                np.testing.assert_array_equal(z['counts'][:3], [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            raw = pd.read_csv(out / 'private_cell_mapping.csv', keep_default_na=False)
            self.assertEqual(raw.cell_type.tolist()[:4], ['', '__unavailable__', 'UNKNOWN', ' Unresolved '])
            self.assertEqual(file_sha256(source), original_hash)

    def test_donor_only_composition_uses_unequal_loaded_cell_totals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root)
            for key in ('section_id', 'retinal_region'):
                cfg['obs_columns'].pop(key)
            out = self.run_config(path, cfg, root / 'out')
            comp = pd.read_csv(out / 'composition.csv')
            first = comp[comp.donor_id == 'd1']
            self.assertEqual(sorted(first.n_cells), [1, 1, 4])
            np.testing.assert_allclose(sorted(first.fraction), [1/6, 1/6, 4/6])
            self.assertEqual(comp.loc[comp.donor_id == 'd2', 'fraction'].tolist(), [1.])

    def test_spatial_edges_and_missing_neighbors(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root, spatial=True)
            out = self.run_config(path, cfg, root / 'out')
            with np.load(out / 'spatial_edges.npz', allow_pickle=False) as z:
                self.assertEqual(z['edges'].T.tolist(), [[0, 1], [1, 0], [2, 1], [4, 5], [5, 4], [6, 7], [7, 6]])
            with np.load(out / 'neighborhood_fractions.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['degree'], [1, 1, 1, 0, 1, 1, 1, 1])
                np.testing.assert_array_equal(z['counts'], [[0, 0, 1], [1, 0, 0], [0, 0, 1],
                    [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
                np.testing.assert_array_equal(z['fractions'], z['counts'])
                self.assertEqual(z['is_isolate'].tolist(), [False, False, False, True, False, False, False, False])
                self.assertEqual(z['annotation_missing'].sum(), 1)
                self.assertEqual(z['unknown_label_count'].tolist(), [1, 0, 1, 0, 0, 0, 0, 0])
            info = json.loads((out / 'neighborhood_fractions.json').read_text())
            self.assertEqual(info['categories'], ['A', 'B', None])
            self.assertEqual(info['descriptor_memory']['max_descriptor_bytes'], 128 * 1024**2)
            self.assertEqual(info['descriptor_memory']['matrix_bytes'], 8 * 3 * 16)
            self.assertIn('tissue masks are absent', ' '.join(info['limitations']))
            self.assertIn('No tissue-area density', ' '.join(info['limitations']))
            cfg.pop('spatial_graph')
            cfg['obs_columns'].pop('cell_type')
            out = self.run_config(path, cfg, root / 'without')
            self.assertEqual(self.record(out)['composition']['status'], 'unavailable')
            self.assertFalse((out / 'composition.csv').exists())
            self.assertEqual(self.record(out)['spatial_graph']['status'], 'not_requested')
            cfg['spatial_graph'] = dict(radius_um=2, max_neighbors=1)
            out = self.run_config(path, cfg, root / 'unannotated')
            with np.load(out / 'neighborhood_fractions.npz', allow_pickle=False) as z:
                self.assertTrue(z['annotation_missing'].all())
                np.testing.assert_array_equal(z['unknown_label_count'], z['degree'])

    def test_study_preflight_before_any_assay_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, base = self.fixture(root)
            base['data_kind'] = 'study'
            cases = [({'mapping_verified': False}, '42', root / 'out'),
                ({'counts_provenance': {}}, '42', root / 'out'), ({}, '', root / 'out'),
                ({'count_source': 'normalized'}, '42', root / 'out'),
                ({'obs_columns': {}}, '42', root / 'out'),
                ({'input_file': str(root / 'outside.h5ad')}, '42', root / 'out'),
                ({'mapping_file': str(root / 'outside.csv')}, '42', root / 'out'),
                ({'canonical_input_roots': []}, '42', root / 'out'),
                ({}, '42', root / 'inputs' / 'out'), ({}, '42', root),
                ({'model': {}}, '42', root / 'out'), ({'split_strategy': 'x'}, '42', root / 'out'),
                ({'fold_indices': [0]}, '42', root / 'out'), ({'schema_version': True}, '42', root / 'out')]
            for changes, job, output in cases:
                with self.subTest(changes=changes, job=job, output=output):
                    cfg = dict(base, **changes); path.write_text(json.dumps(cfg))
                    with patch.dict(os.environ, {'LSB_JOBID': job}), patch.object(runner, 'load_h5ad') as load, \
                            patch.object(runner, 'file_sha256') as hashed:
                        with self.assertRaises((ValueError, FileExistsError)):
                            runner.execute(path, output)
                        load.assert_not_called(); hashed.assert_not_called()
            path.write_text(json.dumps(base))
            with patch.dict(os.environ, {'LSB_JOBID': '42'}):
                with self.assertRaises(ValueError):
                    runner.execute(path, root / 'local', execution_context='local')
                out = runner.execute(path, root / 'accepted')
            self.assertEqual(self.record(out)['lsf_job_id'], '42')

    def test_invalid_graph_settings_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root, spatial=True)
            settings = [None, {}, {'radius_um': 2}, {'radius_um': 2, 'max_neighbors': 1, 'masks': {} }]
            settings += [dict(radius_um=r, max_neighbors=1) for r in (0, -1, True, '2', float('inf'), float('nan'))]
            settings += [dict(radius_um=2, max_neighbors=n) for n in (0, -1, True, 1.5)]
            settings += [dict(radius_um=2, max_neighbors=1, max_descriptor_bytes=b)
                         for b in (None, 0, -1, True, 1.5, '128', float('inf'), {})]
            for graph in settings:
                with self.subTest(graph=graph), patch.object(runner, 'load_h5ad') as load:
                    with self.assertRaises(ValueError):
                        self.run_config(path, dict(cfg, spatial_graph=graph), root / 'out')
                    load.assert_not_called()

    def test_high_cardinality_descriptor_budget_rejected_before_graph_or_dense_allocation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root, spatial=True)
            source = Path(cfg['input_file'])
            a = ad.read_h5ad(source)
            a.obs['type'] = pd.Categorical([f'label{i}' for i in range(7)] + [None])
            a.write_h5ad(source)
            required_bytes = 8 * 8 * (8 + 8)  # Includes the missing-label category.
            cfg['spatial_graph']['max_descriptor_bytes'] = required_bytes - 1
            with patch.object(runner, 'build_radius_graph') as graph, \
                    patch.object(runner, 'neighborhood_fractions') as fractions:
                with self.assertRaisesRegex(ValueError, 'max_descriptor_bytes'):
                    self.run_config(path, cfg, root / 'failed')
                graph.assert_not_called(); fractions.assert_not_called()
            self.assertEqual(self.record(root / 'failed')['status'], 'failed')
            self.assertFalse((root / 'failed/neighborhood_fractions.npz').exists())
            self.assertFalse((root / 'failed/spatial_edges.npz').exists())
            cfg['spatial_graph']['max_descriptor_bytes'] = required_bytes
            out = self.run_config(path, cfg, root / 'exact')
            info = self.record(out)['spatial_graph']['descriptor_memory']
            self.assertEqual((info['n_categories'], info['matrix_bytes'], info['max_descriptor_bytes']),
                             (8, required_bytes, required_bytes))
            with np.load(out / 'neighborhood_fractions.npz', allow_pickle=False) as z:
                self.assertEqual(z['counts'].nbytes + z['fractions'].nbytes, required_bytes)

    def test_unresolved_identities_rejected_with_or_without_requested_graph(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root, spatial=True)
            source = Path(cfg['input_file'])
            a = ad.read_h5ad(source)
            a.obs['section'] = pd.Categorical([' UnAvAiLaBlE '] + a.obs['section'].tolist()[1:])
            a.write_h5ad(source)
            for requested in (False, True):
                candidate = dict(cfg)
                if not requested:
                    candidate.pop('spatial_graph')
                with self.subTest(graph=requested), patch.object(runner, 'build_radius_graph') as graph:
                    with self.assertRaisesRegex(ValueError, 'unresolved'):
                        self.run_config(path, candidate, root / f'failed-{requested}')
                    graph.assert_not_called()
                self.assertFalse((root / f'failed-{requested}/pseudobulk_counts.npz').exists())

    def test_mutations_fail_and_preserve_private_artifact_evidence(self):
        for target in ('input_file', 'mapping_file', 'configuration'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d:
                root = Path(d); path, cfg = self.fixture(root, crosswalk=True)
                original = runner.load_h5ad
                def changed(*args):
                    a = original(*args)
                    p = path if target == 'configuration' else Path(cfg[target])
                    with p.open('ab') as f:
                        f.write(b' ')
                    return a
                with patch.object(runner, 'load_h5ad', side_effect=changed):
                    with self.assertRaises(ValueError):
                        self.run_config(path, cfg, root / 'out')
                record = self.record(root / 'out')
                self.assertEqual((record['status'], record['error_type']), ('failed', 'ValueError'))
                self.assertTrue(record['artifacts'])
                self.assertFalse((root / 'out' / 'pseudobulk_counts.npz').exists())

    def test_late_mutation_records_hashes_of_partial_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root)
            original = runner._describe
            def changed(*args):
                result = original(*args)
                path.write_text(path.read_text() + ' ')
                return result
            with patch.object(runner, '_describe', side_effect=changed), self.assertRaises(ValueError):
                self.run_config(path, cfg, root / 'out')
            record = self.record(root / 'out')
            self.assertEqual(record['status'], 'failed')
            for artifact in record['artifacts']:
                self.assertEqual(artifact['sha256'], file_sha256(root / 'out' / artifact['path']))
            self.assertIn('pseudobulk_counts.npz', [a['path'] for a in record['artifacts']])

    def test_cli_generic_messages_and_private_failure_log(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root)
            result = subprocess.run([sys.executable, '-B', '-m', 'ftd_prediction.descriptive',
                '--config', str(path), '--output', str(root / 'out'), '--mode', 'synthetic',
                '--execution-context', 'local'], capture_output=True, text=True, timeout=30)
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, 'Descriptive run completed.\n', ''))
            def failure(*args):
                print('private-marker-17'); logging.warning('private-marker-17')
                raise RuntimeError('private-marker-17')
            stream, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors), \
                    patch.object(runner, 'load_h5ad', side_effect=failure):
                code = runner.main(['--config', str(path), '--output', str(root / 'failed'),
                                    '--mode', 'synthetic', '--execution-context', 'local'])
                bad = runner.main(['--mode', 'private-marker-17'])
            self.assertEqual((code, bad), (1, 1))
            self.assertNotIn('private-marker-17', stream.getvalue() + errors.getvalue())
            self.assertIn('private-marker-17', (root / 'failed/private_execution.log').read_text())
            self.assertEqual(self.record(root / 'failed')['status'], 'failed')

    def test_unexpected_artifact_symlink_preserves_failed_record(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, cfg = self.fixture(root)
            original = runner._describe
            def linked(adata, output):
                result = original(adata, output)
                (output / 'unexpected-link').symlink_to(cfg['input_file'])
                return result
            with patch.object(runner, '_describe', side_effect=linked), self.assertRaises(ValueError):
                self.run_config(path, cfg, root / 'out')
            record = self.record(root / 'out')
            self.assertEqual(record['status'], 'failed')
            self.assertNotIn('unexpected-link', [a['path'] for a in record['artifacts']])
            for artifact in record['artifacts']:
                self.assertEqual(artifact['sha256'], file_sha256(root / 'out' / artifact['path']))


if __name__ == '__main__':
    unittest.main()
