"""Invented records test report privacy, scope and integrity."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ftd_prediction.neural_report import build_report, main


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.run = self.base / 'run'
        self.run.mkdir()
        self.fold = dict(method='scvi', status='completed', train_donors=['PRIVATE_A'],
            test_donors=['PRIVATE_B'], selected_genes=['PRIVATE_GENE'], training_steps=8,
            encoder_updated=True, reload_verified=True, config={'checkpoint_path': '/PRIVATE_MODEL'},
            metrics={'pca_train_log_expression_mse': .25, 'disease_auroc': .99})
        self.record = dict(schema_version=1, status='completed', data_kind='synthetic',
            disease_inference_enabled=False, configuration={'data_kind': 'synthetic', 'assay': 'rna',
            'input_file': '/PRIVATE_INPUT'}, input_sha256='a'*64, configuration_sha256='b'*64,
            code={'source_sha256': 'c'*64}, environment={'machine': '<b>unsafe<b>',
            'python_executable': '/PRIVATE_ENV'}, descriptive={'n_cells': 20, 'n_genes': 6,
            'n_donors': 2, 'disease_effect_inference': False})
        self.refresh()

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def save(self):
        self.write('run.json', self.record)

    def refresh(self):
        self.write('summary.json', dict(data_kind=self.record['data_kind'], disease_inference_enabled=False,
            descriptive=self.record['descriptive'], folds=[self.fold]))
        self.write('fold_00_summary.json', self.fold)
        (self.run / 'opaque.bin').write_bytes(b'invented-opaque\x00\xff')
        self.record['artifacts'] = [dict(path=p.name, bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.run.iterdir() if p.name != 'run.json']
        self.save()

    def build(self):
        return build_report([self.run], self.base / 'report')

    def reject(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.build()
        self.assertFalse((self.base / 'report').exists())

    def test_private_escaped_immutable(self):
        before = {p: p.read_bytes() for p in self.run.iterdir()}
        output = self.build()
        text = (output / 'report.html').read_text()
        for token in ('PRIVATE_', 'disease_auroc', '<b>', 'href=', 'src='):
            self.assertNotIn(token, text + (output / 'report.md').read_text())
        for token in ('0.25', 'unavailable', '&lt;b&gt;', 'Disease inference is disabled',
                      'test-section adaptation', 'not independent biological truth'):
            self.assertIn(token, text)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual((output / 'report.html').stat().st_mode & 0o777, 0o600)

    def test_tamper(self):
        (self.run / 'opaque.bin').write_bytes(b'changed')
        self.reject()

    def annotation(self, values):
        self.fold['metrics']['annotation_agreement'] = {'per_representation': {
            'scvi': {'heldout_per_donor': values,
                     'training_resubstitution': {'balanced_accuracy': .95}}}}
        self.refresh()

    def test_heldout_aliases_and_real_scores(self):
        self.annotation({'PRIVATE_B': dict(balanced_accuracy=.625, macro_f1=.6, n_labelled=8,
                                           unsupported_classes=['PRIVATE_CLASS'])})
        output = self.build()
        for name in ('report.md', 'report.html'):
            text = (output / name).read_text()
            for token in ('PRIVATE_B', 'PRIVATE_CLASS'):
                self.assertNotIn(token, text)
            for token in ('scvi / held-out donor 1 / balanced_accuracy', '0.625', '0.6',
                          'unsupported class count', 'training resubstitution annotation agreement (not held out)'):
                self.assertIn(token, text)

    def test_heldout_missing_and_zero_annotations(self):
        self.annotation({'PRIVATE_B': dict(n_labelled=0, balanced_accuracy=None, macro_f1=None,
                                           unsupported_classes=[])})
        text = (self.build() / 'report.md').read_text()
        self.assertIn('| scvi / held-out donor 1 / balanced_accuracy | unavailable |', text)
        self.assertIn('| scvi / held-out donor 1 / n_labelled | 0 |', text)
        self.assertIn('| expression / held-out donor 1 / macro_f1 | unavailable |', text)

    def test_heldout_unknown_donor_and_invalid_scalars(self):
        for values in ({'UNDECLARED_SECRET': {}}, {'PRIVATE_B': {'macro_f1': 1.2}},
                       {'PRIVATE_B': {'n_labelled': -1}}, {'PRIVATE_B': {'n_labelled': True}},
                       {'PRIVATE_B': {'unsupported_classes': 'secret'}}, {'PRIVATE_B': 3}):
            with self.subTest(values=values):
                self.annotation(values)
                self.reject()

    def test_summary_consistency(self):
        self.record['descriptive']['n_cells'] = 99
        self.save()
        self.reject()
        self.refresh()
        self.fold['training_steps'] = 99
        self.write('fold_00_summary.json', self.fold)
        p = self.run / 'fold_00_summary.json'
        for a in self.record['artifacts']:
            if a['path'] == p.name:
                a.update(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        self.save()
        self.reject()

    def test_paths(self):
        for bad in ('../outside', '/absolute', 'a/../../outside', 'a\\escape', './summary.json'):
            self.refresh()
            self.record['artifacts'][0]['path'] = bad
            self.save()
            self.reject()
        self.refresh()
        (self.run / 'opaque.bin').unlink()
        (self.base / 'target').write_bytes(b'opaque')
        (self.run / 'opaque.bin').symlink_to(self.base / 'target')
        self.reject()

    def test_directory_symlink(self):
        other = self.base / 'other'
        other.mkdir()
        (other / 'opaque.bin').write_bytes(b'opaque')
        (self.run / 'nested').symlink_to(other, target_is_directory=True)
        self.record['artifacts'][0]['path'] = 'nested/opaque.bin'
        self.save()
        self.reject()

    def test_inventory_and_overlap(self):
        self.record['artifacts'] += self.record['artifacts'][:1]
        self.save()
        self.reject()
        self.refresh()
        self.record['artifacts'] = [a for a in self.record['artifacts'] if a['path'] != 'summary.json']
        self.save()
        self.reject()
        self.fold['test_donors'] = self.fold['train_donors'][:]
        self.refresh()
        self.reject()

    def test_scope_gate_and_kinds(self):
        for kind in ('study', 'reference'):
            self.record['data_kind'] = self.record['configuration']['data_kind'] = kind
            self.refresh()
            for job in ('', '0', '-1', '123x'):
                with patch.dict(os.environ, LSB_JOBID=job):
                    self.reject()
            with patch.dict(os.environ, LSB_JOBID='123'):
                build_report([self.run], self.base / kind)
        other = self.base / 'synthetic'
        other.mkdir()
        (other / 'run.json').write_text(json.dumps(dict(self.record, status='failed', data_kind='synthetic')))
        with patch.dict(os.environ, LSB_JOBID='123'), self.assertRaises(ValueError):
            build_report([self.run, other], self.base / 'report')

    def test_failed_and_multiple_cli(self):
        other = self.base / 'failed'
        other.mkdir()
        failed = dict(self.record, status='failed', error_type='RuntimeError', artifacts=[{'path': '../forbidden'}])
        (other / 'run.json').write_text(json.dumps(failed))
        with patch('ftd_prediction.neural_report._hash', side_effect=AssertionError('partial artifacts read')):
            output = build_report([other], self.base / 'failed-report')
        text = (output / 'report.md').read_text()
        self.assertIn('RuntimeError', text)
        self.assertNotIn('training_steps', text)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            code = main(['--run', str(self.run), '--run', str(other), '--output', str(self.base / 'report')])
        self.assertEqual(code, 0)
        self.assertNotIn(str(self.base), log.getvalue())
        self.assertIn('Run 2: incomplete', (self.base / 'report/report.md').read_text())

    def test_new_output_and_disabled_disease(self):
        for output in (self.run, self.run / 'report'):
            with self.assertRaises(ValueError):
                build_report([self.run], output)
        self.record['disease_inference_enabled'] = True
        self.save()
        self.reject()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(main(['--run', str(self.run), '--output', str(self.run)]), 1)
        self.assertNotIn(str(self.base), log.getvalue())


if __name__ == '__main__':
    unittest.main()
