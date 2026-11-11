"""Invented data only; exercise the real descriptive writer and report renderer."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ftd_prediction import descriptive as runner
from ftd_prediction import descriptive_report as report


def fixture(root, *, assay='spatial', graph=True, annotated=True, missing_section=False):
    """Unequal 4+2 sections in one region; third section is a different donor."""
    root.mkdir()
    obs = pd.DataFrame(dict(donor=['PLANTED_DONOR_A'] * 6 + ['PLANTED_DONOR_B'] * 2,
                            section=['PRIVATE_S1'] * 4 + ['PRIVATE_S2'] * 2 + ['PRIVATE_S3'] * 2,
                            region=['central'] * 6 + [None] * 2,
                            annotation=['RGC', None, 'AC', 'RGC', 'AC', 'AC', '<tag>&', '<tag>&'],
                            component=['u', 'u', 'u', 'v', 'u', 'u', 'u', 'u']),
                       index=[f'INVENTED_CELL_{i}' for i in range(8)])
    if missing_section:
        obs.loc['INVENTED_CELL_1', 'section'] = None
    matrix = sparse.csr_matrix(np.arange(16).reshape(8, 2))
    a = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=['gene1', 'gene2']))
    a.layers['counts'] = matrix.copy()
    a.obsm['spatial'] = np.array([[0, 0], [.1, 0], [.2, 0], [.05, 0],
                                [0, 0], [1, 0], [0, 0], [1, 0]])
    source = root / 'invented.h5ad'
    a.write_h5ad(source)
    columns = dict(donor_id='donor', section_id='section', retinal_region='region')
    if annotated:
        columns['cell_type'] = 'annotation'
    cfg = dict(schema_version=1, data_kind='synthetic', assay=assay, input_file=str(source),
               canonical_input_roots=[str(root / 'unused_raw_root')], count_source='layers/counts',
               counts_provenance=dict(kind='measured_umi', evidence_ref='invented'),
               mapping_verified=True, mapping_evidence_ref='invented', obs_columns=columns, seed=11)
    if assay == 'spatial':
        cfg.update(coordinate_units='um', spatial_key='spatial')
        columns['tissue_component'] = 'component'
        if graph:
            cfg['spatial_graph'] = dict(radius_um=2, max_neighbors=1)
    config_path = root / 'config.json'
    config_path.write_text(json.dumps(cfg))
    with patch.object(runner, '_environment', return_value={'python': 'synthetic'}), \
            patch.object(runner, 'source_snapshot', return_value={'source_sha256': 'a' * 64}):
        return runner.execute(config_path, root / 'run', mode='synthetic', execution_context='local')


def rewrite(run, mutate):
    path = run / 'provenance/run.json'
    record = json.loads(path.read_text())
    mutate(record)
    path.write_text(json.dumps(record))


def rehash(run, name):
    raw = (run / name).read_bytes()
    def update(record):
        for entry in record['artifacts']:
            if entry['path'] == name:
                entry.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    rewrite(run, update)


class DescriptiveReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory(prefix='descriptive-report-synthetic-')
        cls.root = Path(cls.storage.name).resolve()
        cls.source_run = fixture(cls.root / 'spatial')
        cls.output = cls.root / 'report'
        cls.saved_axes = []
        # Initialize matplotlib in private scratch, then inspect actual map axes at save time.
        with patch.dict(os.environ, {'MPLCONFIGDIR': str(cls.root / 'mpl')}):
            from matplotlib.figure import Figure
            original = Figure.savefig
            def inspect(fig, *args, **kwargs):
                for ax in fig.axes:
                    if ax.get_xlabel() == 'x (um)':
                        cls.saved_axes.append((ax.get_title(), ax.get_aspect(), ax.get_ylabel(), ax.get_xlim()))
                return original(fig, *args, **kwargs)
            with patch.object(Figure, 'savefig', inspect):
                report.build_report(cls.source_run, cls.output, max_cells_per_map=2)
        cls.result = json.loads((cls.output / 'report_provenance.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.temp.cleanup)
        self.scratch = Path(self.temp.name)

    def clone(self):
        return Path(shutil.copytree(self.source_run, self.scratch / 'run'))

    def rejected(self, run, *, output=None):
        output = output or self.scratch / 'rejected'
        existing = (output / 'report_provenance.json').read_bytes() if (output / 'report_provenance.json').exists() else None
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = report.main(['--run', str(run), '--output', str(output)])
        self.assertEqual(status, 1)
        self.assertNotIn('PLANTED', stdout.getvalue() + stderr.getvalue())
        self.assertNotIn('PRIVATE', stdout.getvalue() + stderr.getvalue())
        self.assertNotIn(str(run), stdout.getvalue() + stderr.getvalue())
        self.assertEqual(stderr.getvalue(), '')
        if existing is None:
            self.assertFalse((output / 'report_provenance.json').exists())
        else:
            self.assertEqual((output / 'report_provenance.json').read_bytes(), existing)
        return output

    def test_figures_hashes_permissions(self):
        self.assertEqual(len(self.result['figures']), 8)
        for figure in self.result['figures']:
            pdf = (self.output / (figure['stem'] + '.pdf')).read_bytes()
            png = (self.output / (figure['stem'] + '.png')).read_bytes()
            self.assertTrue(pdf.startswith(b'%PDF-'))
            self.assertIn(b'/FontFile2', pdf)  # Embedded TrueType text, not outlined glyphs.
            self.assertNotIn(b'/Subtype /Type3', pdf)
            self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertGreater(len(pdf), 4000)
            self.assertGreater(len(png), 4000)
        for entry in self.result['outputs']:
            path = self.output / entry['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'])
            self.assertEqual(path.stat().st_size, entry['bytes'])
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        for path in self.output.rglob('*'):
            if path.is_file():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(self.result['rendering']['pdf_fonttype'], 42)
        self.assertIn(self.result['rendering']['font'], ('Arial', 'DejaVu Sans'))

    def test_weighted_pooling_missing_mass(self):
        a = self.result['analysis']
        self.assertEqual(a['categories'], ['RGC', 'AC', '<tag>&', None])
        first, second = a['pooled']
        self.assertEqual(first['n_cells'], 6)
        np.testing.assert_allclose(first['fractions'], [2/6, 3/6, 0, 1/6])
        self.assertNotAlmostEqual(first['fractions'][0], (.5 + 0) / 2)
        self.assertEqual(sum(r['n_cells'] for r in a['pooled']), 8)
        self.assertEqual(second['n_cells'], 2)
        self.assertIn('missing', second['region'])
        self.assertAlmostEqual(sum(first['fractions']), 1)

    def test_rna_optional_mixed_sections_preserve_composition_and_stable_aliases(self):
        run = fixture(self.scratch / 'rna_mixed', assay='rna', missing_section=True)
        output = report.build_report(run, self.scratch / 'rna_mixed_report')
        result = json.loads((output / 'report_provenance.json').read_text())
        analysis = result['analysis']
        self.assertEqual(sum(g['n_cells'] for g in analysis['pooled']), 8)
        self.assertEqual(analysis['pooled'][0]['n_cells'], 6)
        np.testing.assert_allclose(analysis['pooled'][0]['fractions'], [2/6, 3/6, 0, 1/6])
        self.assertEqual(analysis['sections'], [])
        self.assertEqual(analysis['aliases']['sections'], [])
        data = report._load(run)
        reversed_data = dict(data, rows=list(reversed(data['rows'])), labels=list(reversed(data['labels'])))
        self.assertEqual(analysis['aliases'], report._analyze(reversed_data)['aliases'])
        self.assertIn('Spatial maps unavailable for RNA.', (output / 'report.md').read_text())
        with self.assertRaises(ValueError):
            fixture(self.scratch / 'invalid_spatial_section', assay='spatial', missing_section=True)

    def test_neighbors_missing_and_isolates(self):
        s = self.result['analysis']['sections'][0]
        self.assertEqual((s['n_cells'], s['n_nonisolated'], s['n_isolates']), (4, 3, 1))
        self.assertEqual(s['isolate_fraction'], .25)
        self.assertEqual(s['mean_out_degree'], .75)
        self.assertEqual(s['max_out_degree'], 1)
        # RGC non-isolate and AC both neighbor an unannotated cell; the RGC isolate is excluded.
        np.testing.assert_allclose(s['neighbor_means'][0], [0, 0, 0, 1])
        np.testing.assert_allclose(s['neighbor_means'][1], [0, 0, 0, 1])
        np.testing.assert_allclose(s['neighbor_means'][3], [1, 0, 0, 0])
        self.assertIn(report._NEIGHBOR, (self.output / 'report.md').read_text())

    def test_sections_aspect_and_sampling(self):
        self.assertEqual(len(self.saved_axes), 6)  # One map per section, each exported twice.
        for title, aspect, ylabel, _ in self.saved_axes:
            self.assertEqual(aspect, 1.0)
            self.assertEqual(ylabel, 'y (um)')
            self.assertIn('Section', title)
            self.assertNotIn('PRIVATE', title)
        maps = self.result['rendering']['maps']
        self.assertEqual(maps[0]['selected_mapping_rows'], [0, 3])
        self.assertEqual(maps[1]['selected_mapping_rows'], [4, 5])
        self.assertEqual(maps[0]['total_cells'], 4)
        self.assertEqual(maps[0]['displayed_cells'], 2)
        self.assertEqual(self.result['analysis']['pooled'][0]['n_cells'], 6)

    def test_offline_escaping_aliases(self):
        document = (self.output / 'report.html').read_text()
        self.assertIn('&lt;tag&gt;&amp;', document)
        self.assertNotIn('<tag>', document)
        self.assertIn('data:image/png;base64,', document)
        self.assertNotIn('https://', document)
        self.assertNotIn('PLANTED_DONOR', document)
        self.assertNotIn('PRIVATE_S', document)
        self.assertNotIn('INVENTED_CELL', document)
        data = report._load(self.source_run)
        reversed_data = dict(data, rows=list(reversed(data['rows'])), labels=list(reversed(data['labels'])))
        self.assertEqual(report._analyze(data)['aliases'], report._analyze(reversed_data)['aliases'])

    def test_no_source_reopening(self):
        # Remove every synthetic assay/config source; completed artifacts alone are sufficient.
        run = self.clone()
        record = json.loads((run / 'provenance/run.json').read_text())
        record['configuration']['input_file'] = '/UNAVAILABLE/PRIVATE_INPUT.h5ad'
        record['source_file'] = record['configuration']['input_file']
        (run / 'provenance/run.json').write_text(json.dumps(record))
        original = np.load
        def safe_load(file, *args, **kwargs):
            self.assertIsInstance(file, io.BytesIO)
            self.assertIs(kwargs.get('allow_pickle'), False)
            return original(file, *args, **kwargs)
        with patch.object(ad, 'read_h5ad', side_effect=AssertionError('assay read')), \
                patch.object(sparse, 'load_npz', side_effect=AssertionError('expression read')), \
                patch.object(np, 'load', safe_load):
            report._load(run)

    def test_incomplete_unknown_metadata(self):
        for key, value in [('status', 'running'), ('status', 'failed'), ('data_kind', 'UNKNOWN'),
                           ('execution_context', 'UNKNOWN'), ('training_enabled', True),
                           ('disease_inference_enabled', True), ('schema_version', True)]:
            with self.subTest(key=key, value=value):
                run = self.scratch / f'run_{key}_{value}'
                shutil.copytree(self.source_run, run)
                rewrite(run, lambda r: r.update({key: value}))
                self.rejected(run, output=self.scratch / f'out_{key}_{value}')

    def test_inventory_integrity(self):
        for fault in ('hash', 'size', 'missing', 'duplicate'):
            with self.subTest(fault=fault):
                run = Path(shutil.copytree(self.source_run, self.scratch / fault))
                def corrupt(r):
                    if fault == 'hash':
                        r['artifacts'][0]['sha256'] = '0' * 64
                    elif fault == 'size':
                        r['artifacts'][0]['bytes'] += 1
                    elif fault == 'missing':
                        r['artifacts'].pop()
                    else:
                        r['artifacts'][-1] = r['artifacts'][0]
                rewrite(run, corrupt)
                self.rejected(run, output=self.scratch / ('out_' + fault))

    def test_unsafe_artifacts(self):
        for i, path in enumerate(('../PLANTED_SECRET', '/PLANTED_SECRET', 'a/../summary.json',
                                  'a\\summary.json', 'weights.pt', 'counts.h5ad')):
            run = Path(shutil.copytree(self.source_run, self.scratch / f'run{i}'))
            rewrite(run, lambda r: r['artifacts'][0].update(path=path))
            self.rejected(run, output=self.scratch / f'out{i}')
        run = self.clone()
        target = self.scratch / 'PLANTED_SECRET'
        (run / 'summary.json').rename(target)
        (run / 'summary.json').symlink_to(target)
        self.rejected(run)

    def test_npz_schema_alignment(self):
        cases = ('object', 'negative', 'outside', 'cross_section', 'fraction', 'alignment', 'category')
        for case in cases:
            with self.subTest(case=case):
                run = Path(shutil.copytree(self.source_run, self.scratch / case))
                name = 'spatial_edges.npz' if case in ('negative', 'outside', 'cross_section') else 'neighborhood_fractions.npz'
                with np.load(run / name, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                if case == 'object':
                    arrays['degree'] = arrays['degree'].astype(object)
                elif case in ('negative', 'outside', 'cross_section'):
                    arrays['edges'][1, 0] = {'negative': -1, 'outside': 8, 'cross_section': 4}[case]
                elif case == 'fraction':
                    arrays['fractions'][0] = 0
                elif case == 'alignment':
                    arrays['annotation_missing'] = arrays['annotation_missing'][::-1]
                elif case == 'category':
                    info = json.loads((run / 'neighborhood_fractions.json').read_text())
                    info['categories'][0] = 'UNKNOWN'
                    (run / 'neighborhood_fractions.json').write_text(json.dumps(info))
                    rehash(run, 'neighborhood_fractions.json')
                np.savez_compressed(run / name, **arrays)
                rehash(run, name)
                self.rejected(run, output=self.scratch / ('out_' + case))

    def test_table_json_schemas(self):
        for case in ('mapping', 'composition', 'json', 'donor'):
            run = Path(shutil.copytree(self.source_run, self.scratch / case))
            name = 'summary.json' if case == 'json' else 'composition.csv' if case == 'composition' else 'private_cell_mapping.csv'
            if case == 'json':
                (run / name).write_text('{"x": 1, "x": 2}')
            else:
                frame = pd.read_csv(run / name, keep_default_na=False)
                if case == 'mapping':
                    frame['unexpected_metadata'] = 'PLANTED'
                elif case == 'composition':
                    frame.loc[0, 'group_cells'] = 900
                else:
                    frame.loc[0, 'donor_id'] = 'UNKNOWN'
                frame.to_csv(run / name, index=False)
            rehash(run, name)
            self.rejected(run, output=self.scratch / ('out_' + case))

    def test_output_paths_and_arguments(self):
        self.rejected(self.source_run, output=self.output)
        for output in (self.source_run / 'nested', self.source_run.parent):
            with self.assertRaises(ValueError):
                report.build_report(self.source_run, output)
        link = self.scratch / 'linked'
        link.symlink_to(self.scratch, target_is_directory=True)
        self.rejected(self.source_run, output=link / 'out')
        for args in (['--PLANTED_SECRET'], ['--run', 'PLANTED', '--output', 'PLANTED', '--max-panels', 'PRIVATE']):
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(report.main(args), 1)
            self.assertNotIn('PLANTED', stdout.getvalue() + stderr.getvalue())
            self.assertNotIn('PRIVATE', stdout.getvalue() + stderr.getvalue())

    def test_study_guard(self):
        run = self.clone()
        record = json.loads((run / 'provenance/run.json').read_text())
        record.update(data_kind='study', execution_context='pmacs', lsf_job_id='123')
        record['configuration']['data_kind'] = 'study'
        summary = json.loads((run / 'summary.json').read_text())
        summary.update(data_kind='study', evidence_scope='descriptive_only')
        (run / 'summary.json').write_text(json.dumps(summary))
        (run / 'provenance/run.json').write_text(json.dumps(record))
        rehash(run, 'summary.json')
        for i, job in enumerate(('', '0', '-1', 'abc', '1.0')):
            with patch.dict(os.environ, {'LSB_JOBID': job}):
                self.rejected(run, output=self.scratch / f'out{i}')
        with patch.dict(os.environ, {'LSB_JOBID': '124'}):
            self.assertEqual(report._load(run)['record']['data_kind'], 'study')

    def test_unavailable_panels_and_bounds(self):
        run = fixture(self.scratch / 'missing', graph=False, annotated=False)
        output = report.build_report(run, self.scratch / 'missing_report', max_panels=2)
        text = (output / 'report.md').read_text()
        self.assertIn('Graph not requested', text)
        self.assertIn('Cell-type annotation absent', text)
        result = json.loads((output / 'report_provenance.json').read_text())
        self.assertEqual([v['stem'] for v in result['figures']], ['loaded_cells', 'section_001_map'])
        self.assertGreater(result['rendering']['skipped_panels'], 0)
        rna = fixture(self.scratch / 'rna', assay='rna')
        output = report.build_report(rna, self.scratch / 'rna_report', max_categories=3, max_groups=1)
        result = json.loads((output / 'report_provenance.json').read_text())
        self.assertEqual(len(result['figures']), 2)
        self.assertIn('Spatial maps unavailable for RNA', (output / 'report.md').read_text())
        self.assertEqual(result['rendering']['omitted_category_count'], 2)
        self.assertIn([None], result['rendering']['category_groups'])
        self.assertEqual(result['rendering']['omitted_pooled_groups'], 1)
        self.assertEqual(sum(r['n_cells'] for r in result['analysis']['pooled']), 8)
        for kwargs in ({'max_panels': 1}, {'max_categories': 2}, {'max_cells_per_map': 0}, {'max_groups': True}):
            with self.assertRaises(ValueError):
                report.build_report(rna, self.scratch / 'bad', **kwargs)

    def test_private_failure_log(self):
        with patch.object(report, '_load', side_effect=ValueError('PLANTED_PRIVATE_ID')):
            output = self.rejected(self.source_run)
        self.assertIn('PLANTED_PRIVATE_ID', (output / 'private_error.log').read_text())


if __name__ == '__main__':
    unittest.main()
