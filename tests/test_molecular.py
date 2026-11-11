"""Invented accepted-artifact fixtures; no H5AD, model or study-data access."""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy import sparse

from ftd_prediction import molecular as m


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, header, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def csv_read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


class MolecularTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.root.chmod(0o700)
        self.source = self.root / 'source'
        (self.source / 'provenance').mkdir(parents=True)
        self.config = self.root / 'config.json'
        self.universe = self.root / 'universe.csv'
        self.out = self.root / 'out'
        self.ids = ['plantedCase', 'plantedRef1', 'plantedRef2']
        self.genes = ['g1', 'g2', 'low', 'not_a_prefix_control']
        self.header = ['donor_id', 'section_id', 'retinal_region', 'cell_type', 'n_cells',
                       'donor_id_missing', 'section_id_missing', 'retinal_region_missing', 'annotation_missing']
        self.groups = [[self.ids[0], 's1', 'central', 'A', 5, False, False, False, False],
                       [self.ids[0], 's2', 'central', 'A', 25, False, False, False, False],
                       [self.ids[1], 's3', 'central', 'A', 20, False, False, False, False],
                       [self.ids[2], 's4', 'central', 'A', 20, False, False, False, False]]
        self.counts = np.array([[10, 90, 0, 9000], [90, 10, 1, 9000],
                                [10, 90, 0, 9000], [900, 100, 0, 9000]], dtype=np.int64)
        csv_write(self.universe, ['gene_id'], [[g] for g in self.genes[:3]])
        self.cfg = dict(schema_version=1, data_kind='synthetic', assay='rna',
                        source_run_dir=str(self.source), source_run_sha256='0' * 64,
                        source_annotation_evidence_ref='invented annotation audit',
                        feature_universe_evidence_ref='invented endogenous audit',
                        endogenous_gene_ids_file=str(self.universe), contrasts=[dict(
                            contrast_id='plantedContrast', case_donor_id=self.ids[0],
                            reference_donor_ids=self.ids[1:], retinal_region='central', cell_type='A',
                            label_evidence_ref='invented labels', matching_evidence_ref='invented matching')])
        self.record = dict(schema_version=1, status='completed', data_kind='synthetic',
                           training_enabled=False, disease_inference_enabled=False,
                           configuration=dict(schema_version=1, data_kind='synthetic', assay='rna'),
                           input_files=[dict(path='/never/reopen/raw.h5ad', sha256='a' * 64)],
                           code=dict(source_sha256='b' * 64))
        self.refresh()

    def refresh(self, matrix=True):
        if matrix:
            sparse.save_npz(self.source / m.ARTIFACTS[0], sparse.csr_matrix(self.counts))
        csv_write(self.source / m.ARTIFACTS[1], self.header, self.groups)
        csv_write(self.source / m.ARTIFACTS[2], ['gene_id'], [[g] for g in self.genes])
        present = [k for k in m._IDENTITY if k in self.header]
        self.record['descriptive'] = dict(n_cells=sum(int(r[self.header.index('n_cells')]) for r in self.groups),
            n_genes=len(self.genes), n_groups=len(self.groups), n_donors=len({r[0] for r in self.groups}),
            grouping_columns=present, cell_type_annotation_available='cell_type' in present,
            disease_effect_inference=False)
        self.repin()

    def repin(self):
        self.record['artifacts'] = [dict(path=name, bytes=(self.source / name).stat().st_size,
                                         sha256=digest(self.source / name)) for name in m.ARTIFACTS]
        self.save_record()

    def save_record(self):
        run = self.source / 'provenance/run.json'
        run.write_text(json.dumps(self.record))
        self.cfg['source_run_sha256'] = digest(run)
        self.save_config()

    def save_config(self):
        self.config.write_text(json.dumps(self.cfg))

    def run_fixture(self, **kwargs):
        self.save_config()
        return m.execute(self.config, self.out, mode=kwargs.get('mode', 'synthetic'),
                         execution_context=kwargs.get('execution_context', 'local'))

    def rows(self):
        return csv_read(self.out / 'molecular_contrasts.csv')

    def test_weighted_section_pooling_equal_donor_mean_and_control_exclusion(self):
        self.run_fixture()
        rows = self.rows()
        self.assertEqual(len(rows), 9)
        self.assertEqual({r['gene_id'] for r in rows}, set(self.genes[:3]))
        first = next(r for r in rows if r['gene_id'] == 'g1' and r['reference_alias'] == 'D002')
        self.assertEqual((int(first['case_counts']), int(first['case_library_size']), int(first['case_n_cells'])),
                         (100, 201, 30))
        self.assertEqual(json.loads(first['reference_library_sizes']), [100])
        expected_case = np.log2(1e6 * 100 / 201 + 1)
        ref1, ref2 = np.log2(1e6 * .1 + 1), np.log2(1e6 * .9 + 1)
        self.assertAlmostEqual(float(first['case_logcpm']), expected_case)
        mean = next(r for r in rows if r['gene_id'] == 'g1' and r['comparison'] == 'mean_reference')
        self.assertAlmostEqual(float(mean['reference_logcpm']), (ref1 + ref2) / 2)
        self.assertAlmostEqual(float(mean['logcpm_difference']), expected_case - (ref1 + ref2) / 2)
        self.assertGreater(abs(float(mean['reference_logcpm']) - np.log2(1e6 * 910 / 1100 + 1)), .1)
        self.assertGreater(abs(float(mean['reference_logcpm']) - np.log2(1e6 * .5 + 1)), .1)
        self.assertEqual(json.loads(mean['reference_counts']), [10, 900])
        selected = csv_read(self.out / 'matched_groups.csv')
        self.assertEqual([int(r['selected_n_cells']) for r in selected[:2]], [30, 30])
        record = json.loads((self.out / 'provenance/run.json').read_text())
        self.assertEqual(record['input_pins_before'], record['input_pins_after'])
        self.assertIn('input_files', record['inherited_not_reverified'])
        for artifact in record['artifacts']:
            self.assertEqual(artifact['sha256'], digest(self.out / artifact['path']))
            self.assertEqual(artifact['bytes'], (self.out / artifact['path']).stat().st_size)
        self.assertEqual(record['parameters'], dict(min_cells=20, min_gene_counts=10, pseudocount_cpm=1))

    def test_low_gene_counts_and_exact_threshold(self):
        self.counts[0, 2] = 9
        self.counts[1, 2] = 0
        self.counts[3, 2] = 1
        self.refresh()
        self.run_fixture()
        low = [r for r in self.rows() if r['gene_id'] == 'low']
        self.assertEqual([(r['reference_alias'], r['status']) for r in low],
                         [('D002', 'unavailable'), ('D003', 'available'), ('mean_reference', 'available')])
        self.assertEqual(low[0]['reason'], 'low_counts')
        self.assertEqual(low[0]['logcpm_difference'], '')

    def test_low_cell_reference_invalidates_all_reference_mean(self):
        self.groups[3][4] = 19
        self.refresh()
        self.run_fixture()
        for row in self.rows():
            if row['reference_alias'] in ('D003', 'mean_reference'):
                self.assertEqual(row['status'], 'unavailable')
                self.assertIn('low_cells', row['reason'])
                self.assertEqual(row['logcpm_difference'], '')
        mean = next(r for r in self.rows() if r['comparison'] == 'mean_reference')
        self.assertEqual((mean['n_references'], mean['n_eligible_references']), ('2', '1'))

    def test_absent_population_missing_annotations_regions_and_no_other_region_pooling(self):
        self.groups[1][2] = 'peripheral'
        self.groups[2][3], self.groups[2][8] = '', True
        self.groups[3][2], self.groups[3][7] = '', True
        self.refresh()
        self.run_fixture()
        coverage = csv_read(self.out / 'matched_groups.csv')
        self.assertEqual({r['exclusion_reason'] for r in coverage},
                         {'', 'other_region', 'annotation_missing', 'region_missing'})
        self.assertTrue(all(r['status'] == 'unavailable' for r in self.rows()))
        self.assertTrue(all(int(r['case_n_cells']) == 5 for r in self.rows()))
        mean = self.rows()[-1]
        self.assertIn('low_cells', mean['reason'])
        self.assertIn('missing_population', mean['reason'])
        self.assertEqual(json.loads(mean['reference_counts']), [None, None])

    def test_unrecorded_annotation_or_region_is_unavailable(self):
        for field in ('cell_type', 'retinal_region'):
            with self.subTest(field=field):
                header, groups = self.header[:], [r[:] for r in self.groups]
                remove = sorted([self.header.index(field), self.header.index(m._FLAGS[field])], reverse=True)
                for i in remove:
                    self.header.pop(i)
                    for row in self.groups:
                        row.pop(i)
                self.refresh()
                self.out = self.root / ('out-' + field)
                self.run_fixture()
                self.assertTrue(all(r['status'] == 'unavailable' for r in self.rows()))
                self.header, self.groups = header, groups

    def test_rna_without_section_column_does_not_invent_section_count(self):
        for i in sorted([self.header.index('section_id'), self.header.index('section_id_missing')], reverse=True):
            self.header.pop(i)
            for row in self.groups:
                row.pop(i)
        # The descriptive source has already summed its two case rows in this schema.
        self.groups[0][self.header.index('n_cells')] = 30
        self.groups.pop(1)
        self.counts = np.vstack([self.counts[:2].sum(axis=0), self.counts[2:]])
        self.refresh()
        self.run_fixture()
        self.assertTrue(all(r['selected_sections'] == '' for r in csv_read(self.out / 'matched_groups.csv')))
        self.assertEqual(int(self.rows()[0]['case_counts']), 100)

    def test_missing_case_population_zero_library_and_absent_donor(self):
        self.cfg['contrasts'][0]['cell_type'] = 'other'
        self.run_fixture()
        self.assertTrue(all('missing_population' in r['reason'] for r in self.rows()))
        self.out = self.root / 'zero'
        self.cfg['contrasts'][0]['cell_type'] = 'A'
        self.counts[:2, :3] = 0
        self.refresh()
        self.run_fixture()
        self.assertTrue(all('zero_library' in r['reason'] for r in self.rows()))
        self.out = self.root / 'absent'
        self.cfg['contrasts'][0]['case_donor_id'] = 'plantedAbsent'
        with self.assertRaisesRegex(ValueError, 'donor ID is absent'):
            self.run_fixture()

    def test_guard_precedes_any_source_or_universe_access(self):
        self.cfg['data_kind'] = 'study'
        self.save_config()
        original = m._read
        opened = []
        def spy(path, limit):
            opened.append(path)
            return original(path, limit)
        for context, job in [('local', '12'), ('pmacs', ''), ('pmacs', '0'), ('pmacs', '-1'), ('pmacs', 'x')]:
            with self.subTest(context=context, job=job), patch.dict(os.environ, LSB_JOBID=job), \
                    patch.object(m, '_read', side_effect=spy), patch.object(m, 'validate_runtime_scope', wraps=m.validate_runtime_scope) as guard:
                with self.assertRaises(ValueError):
                    m.execute(self.config, self.out, mode='study', execution_context=context)
                guard.assert_called_once()
                self.assertEqual(opened, [self.config])
                self.assertFalse(self.out.exists())
                opened.clear()

    def test_study_synthetic_scope_and_source_record_fail_closed(self):
        variants = [('status', 'failed'), ('training_enabled', True), ('disease_inference_enabled', True),
                    ('schema_version', True), ('data_kind', 'study')]
        for field, value in variants:
            with self.subTest(field=field):
                prior = self.record[field]
                self.record[field] = value
                self.save_record()
                self.out = self.root / ('bad-' + field)
                with self.assertRaises(ValueError):
                    self.run_fixture()
                self.record[field] = prior
        self.record['configuration']['assay'] = 'spatial'
        self.save_record()
        self.out = self.root / 'assay'
        with self.assertRaisesRegex(ValueError, 'assay or scope'):
            self.run_fixture()

    def test_guarded_study_defaults_with_invented_artifacts_and_spatial_segmentation(self):
        self.cfg.update(data_kind='study', assay='spatial')
        self.record.update(data_kind='study')
        self.record['configuration'].update(data_kind='study', assay='spatial', segmentation_ref='invented variant')
        self.save_record()
        with patch.dict(os.environ, LSB_JOBID='42'):
            m.execute(self.config, self.out)
        record = json.loads((self.out / 'provenance/run.json').read_text())
        self.assertEqual((record['lsf_job_id'], record['execution_context']), ('42', 'pmacs'))
        self.record['configuration']['segmentation_ref'] = 'REPLACE_ME'
        self.save_record()
        with patch.dict(os.environ, LSB_JOBID='42'), self.assertRaisesRegex(ValueError, 'segmentation'):
            m.execute(self.config, self.root / 'bad-segmentation')

    def test_source_artifact_and_config_tamper_and_failure_record(self):
        # Modification without repinning is rejected before analysis.
        (self.source / m.ARTIFACTS[2]).write_text('gene_id\ntampered\n')
        with self.assertRaisesRegex(ValueError, 'hash or byte'):
            self.run_fixture()
        record = json.loads((self.out / 'provenance/run.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertTrue((self.out / 'private_error.log').exists())
        self.refresh()
        original = m._analyse
        for target in ('config', 'run', 'artifact', 'universe'):
            with self.subTest(target=target):
                self.out = self.root / ('tamper-' + target)
                def tamper(*args):
                    result = original(*args)
                    paths = dict(config=self.config, run=self.source / 'provenance/run.json',
                                 artifact=self.source / m.ARTIFACTS[1], universe=self.universe)
                    with paths[target].open('a') as handle:
                        handle.write(' ')
                    return result
                with patch.object(m, '_analyse', side_effect=tamper), self.assertRaisesRegex(ValueError, 'pinned input changed'):
                    self.run_fixture()
                csv_write(self.universe, ['gene_id'], [[g] for g in self.genes[:3]])
                self.refresh()

    def test_missing_hash_artifact_and_bad_source_pin(self):
        self.record['artifacts'][0].pop('sha256')
        self.save_record()
        with self.assertRaises(ValueError):
            self.run_fixture()
        self.repin()
        self.record['artifacts'].pop()
        self.save_record()
        self.out = self.root / 'missing'
        with self.assertRaisesRegex(ValueError, 'artifact is missing'):
            self.run_fixture()
        self.out = self.root / 'pin'
        self.cfg['source_run_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256 pin'):
            self.run_fixture()

    def test_configuration_rejects_placeholders_overrides_duplicates_and_ambiguous_universe(self):
        original = json.loads(json.dumps(self.cfg))
        mutations = [dict(all_features_endogenous_verified=True), dict(all_features_endogenous_verified=False),
                     dict(source_annotation_evidence_ref=False), dict(feature_universe_evidence_ref='REPLACE_ME'),
                     dict(source_annotation_evidence_ref='unverified'),
                     dict(feature_universe_evidence_ref='placeholder'),
                     dict(feature_universe_evidence_ref='<verified_record>'),
                     dict(source_run_dir='/REPLACE_ME'), dict(min_cells=1), dict(schema_version=True),
                     dict(contrasts=[]), dict(contrasts=self.cfg['contrasts'] * 2)]
        for change in mutations:
            with self.subTest(change=change):
                self.cfg = {**original, **change}
                with self.assertRaises(ValueError):
                    self.run_fixture()
                self.assertFalse(self.out.exists())
        self.cfg = original
        contrast = self.cfg['contrasts'][0]
        contrast['reference_donor_ids'] = [self.ids[0]]
        with self.assertRaises(ValueError):
            self.run_fixture()

    def test_explicit_all_features_and_invalid_universe_csv(self):
        for content in ('gene_id\ng1\ng1\n', 'gene_id\nabsent\n', 'gene_id,symbol\ng1,x\n', 'gene_id\n""\n'):
            with self.subTest(content=content):
                self.universe.write_text(content)
                self.out = self.root / ('bad-universe-' + hashlib.sha256(content.encode()).hexdigest()[:8])
                with self.assertRaises(ValueError):
                    self.run_fixture()
        self.cfg.pop('endogenous_gene_ids_file')
        self.cfg['all_features_endogenous_verified'] = True
        self.out = self.root / 'all-features'
        self.run_fixture()
        self.assertIn('not_a_prefix_control', {r['gene_id'] for r in self.rows()})
        self.assertEqual(int(self.rows()[0]['case_library_size']), 18201)

    def test_duplicate_gene_groups_dimensions_cells_and_flags_fail(self):
        cases = ('genes', 'groups', 'cells', 'flag', 'dimensions')
        for case in cases:
            with self.subTest(case=case):
                genes, groups = self.genes[:], [r[:] for r in self.groups]
                if case == 'genes':
                    self.genes[1] = self.genes[0]
                elif case == 'groups':
                    self.groups[1][:4] = self.groups[0][:4]
                elif case == 'flag':
                    self.groups[2][8] = True
                self.refresh()
                if case == 'cells':
                    self.record['descriptive']['n_cells'] += 1
                    self.save_record()
                if case == 'dimensions':
                    sparse.save_npz(self.source / m.ARTIFACTS[0], sparse.csr_matrix(self.counts[:, :2]))
                    self.repin()
                self.out = self.root / ('bad-' + case)
                with self.assertRaises(ValueError):
                    self.run_fixture()
                self.genes, self.groups = genes, groups

    def test_invalid_count_values(self):
        for number in (-1, .5, np.nan, np.inf):
            with self.subTest(number=number):
                values = self.counts.astype(float)
                values[0, 0] = number
                sparse.save_npz(self.source / m.ARTIFACTS[0], sparse.csr_matrix(values))
                self.repin()
                self.out = self.root / ('count-' + str(number))
                with self.assertRaisesRegex(ValueError, 'finite nonnegative integers'):
                    self.run_fixture()

    def test_malformed_object_oversize_and_bad_sparse_indices(self):
        npz = self.source / m.ARTIFACTS[0]
        payloads = {
            'malformed': dict(data=np.array([1]), format=np.array(b'csr'), shape=np.array([4, 4]),
                              indices=np.array([99]), indptr=np.array([0, 1, 1, 1, 1])),
            'object': dict(data=np.array([object()], dtype=object), format=np.array(b'csr'), shape=np.array([4, 4]),
                           indices=np.array([0]), indptr=np.array([0, 1, 1, 1, 1])),
            'duplicate': dict(data=np.array([1, 2]), format=np.array(b'csr'), shape=np.array([4, 4]),
                              indices=np.array([0, 0]), indptr=np.array([0, 2, 2, 2, 2]))}
        for name, payload in payloads.items():
            with self.subTest(name=name):
                np.savez(npz, **payload)
                self.repin()
                self.out = self.root / name
                with self.assertRaises(ValueError):
                    self.run_fixture()
        self.refresh()
        with patch.dict(m.LIMITS, array_bytes=16), self.assertRaisesRegex(ValueError, 'expanded bytes'):
            self.out = self.root / 'oversize'
            self.run_fixture()
        # A lying NPY header is rejected without loading/densifying its claimed array.
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, dict(descr='<f8', fortran_order=False, shape=(10**10,)))
        with zipfile.ZipFile(npz, 'w') as z:
            z.writestr('data.npy', stream.getvalue())
            for key, array in payloads['malformed'].items():
                if key != 'data':
                    b = io.BytesIO()
                    np.save(b, array)
                    z.writestr(key + '.npy', b.getvalue())
        self.repin()
        self.out = self.root / 'lying-header'
        with patch.object(np, 'load', side_effect=AssertionError('allocation before header validation')), self.assertRaises(ValueError):
            self.run_fixture()

    def test_sparse_csc_and_no_full_matrix_densification(self):
        sparse.save_npz(self.source / m.ARTIFACTS[0], sparse.csc_matrix(self.counts))
        self.repin()
        with patch.object(sparse.csr_matrix, 'toarray', side_effect=AssertionError('full densification')):
            self.run_fixture()
        self.assertEqual(len(self.rows()), 9)

    def test_accepts_unsorted_output_of_descriptive_sparse_product(self):
        product = (sparse.eye(4, format='csr', dtype=np.int64) @ sparse.csr_matrix(self.counts)).tocsr()
        self.assertFalse(product.has_sorted_indices)
        sparse.save_npz(self.source / m.ARTIFACTS[0], product)
        self.repin()
        self.run_fixture()
        self.assertEqual(int(self.rows()[0]['case_counts']), 100)
        self.assertEqual(int(self.rows()[0]['case_library_size']), 201)

    def test_resource_limits_fail_without_truncating_rows(self):
        for key, cap in [('output_rows', 8), ('matched_rows', 1), ('dense_elements', 1), ('genes', 3)]:
            self.out = self.root / ('limit-' + key)
            with self.subTest(key=key), patch.dict(m.LIMITS, {key: cap}), self.assertRaises(ValueError):
                self.run_fixture()
            self.assertFalse((self.out / 'molecular_contrasts.csv').exists())

    def test_private_new_outputs_no_overwrite_no_input_overlap(self):
        self.run_fixture()
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o700)
        for path in self.out.rglob('*'):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)
        original = digest(self.out / 'molecular_contrasts.csv')
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(digest(self.out / 'molecular_contrasts.csv'), original)
        for target in (self.source / 'nested', self.config, self.universe, self.root):
            with self.subTest(target=target):
                self.out = target
                with self.assertRaises(ValueError):
                    self.run_fixture()

    def test_symlinks_and_writable_ancestors_rejected(self):
        linked = self.root / 'linked'
        linked.symlink_to(self.source, target_is_directory=True)
        self.cfg['source_run_dir'] = str(linked)
        with self.assertRaises(OSError):
            self.run_fixture()
        self.cfg['source_run_dir'] = str(self.source)
        self.out = self.root / 'leaf-link'
        original = self.source / m.ARTIFACTS[2]
        original.rename(self.source / 'real-genes.csv')
        original.symlink_to(self.source / 'real-genes.csv')
        with self.assertRaises(OSError):
            self.run_fixture()
        original.unlink()
        (self.source / 'real-genes.csv').rename(original)
        self.source.chmod(0o777)
        self.out = self.root / 'untrusted'
        with self.assertRaisesRegex(ValueError, 'untrusted'):
            self.run_fixture()
        self.source.chmod(0o755)

    def test_owned_private_workspace_below_shared_lab_parent_accepts_complete_run(self):
        shared = self.root / 'shared-lab'
        shared.mkdir()
        shared.chmod(0o2770)
        private = shared / 'owned-private'
        private.mkdir(mode=0o700)
        for attribute in ('source', 'config', 'universe'):
            old = getattr(self, attribute)
            new = private / old.name
            old.rename(new)
            setattr(self, attribute, new)
        self.out = private / 'out'
        self.cfg.update(source_run_dir=str(self.source), endogenous_gene_ids_file=str(self.universe))
        self.run_fixture()
        record = json.loads((self.out / 'provenance/run.json').read_text())
        self.assertEqual(record['status'], 'completed')
        self.assertTrue(self.rows())
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)

    def test_shared_ancestry_requires_owned_fully_private_boundary(self):
        shared = self.root / 'shared-lab'
        shared.mkdir()
        shared.chmod(0o2770)
        boundary = shared / 'boundary'
        boundary.mkdir()
        boundary.chmod(0o750)
        with self.assertRaisesRegex(ValueError, 'untrusted'):
            m._directory(boundary)
        boundary.chmod(0o700)
        inode = boundary.stat().st_ino
        original = os.fstat
        def foreign_owner(fd):
            info = original(fd)
            if info.st_ino == inode:
                fields = list(info)
                fields[4] = os.getuid() + 1
                return os.stat_result(fields)
            return info
        with patch.object(m.os, 'fstat', side_effect=foreign_owner):
            with self.assertRaisesRegex(ValueError, 'untrusted'):
                m._directory(boundary)

    def test_cli_never_emits_planted_identifiers_on_success_error_or_parse_failure(self):
        command = [sys.executable, '-m', 'ftd_prediction.molecular', '--config', str(self.config),
                   '--output', str(self.out), '--mode', 'synthetic', '--execution-context', 'local']
        for suffix in ([], ['--plantedCLI'], []):
            result = subprocess.run(command + suffix, text=True, capture_output=True, timeout=30)
            self.assertNotIn('planted', result.stdout + result.stderr)
            self.assertEqual(result.stderr, '')
        for name in ('summary.json', 'report.md'):
            self.assertNotIn('planted', (self.out / name).read_text())

    def test_no_model_raw_or_mapping_reads(self):
        original = m._read
        observed = []
        def guarded(path, limit):
            self.assertNotIn('raw.h5ad', str(path))
            self.assertNotIn('private_cell_mapping', str(path))
            observed.append(path)
            return original(path, limit)
        original_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'scvi', 'novae', 'anndata', 'h5py'):
                raise AssertionError('Model or H5AD import')
            return original_import(name, *args, **kwargs)
        with patch.object(m, '_read', side_effect=guarded), patch('builtins.__import__', side_effect=guarded_import):
            self.run_fixture()
        self.assertTrue(observed)


if __name__ == '__main__':
    unittest.main()
