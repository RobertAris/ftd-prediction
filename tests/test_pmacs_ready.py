import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ftd_prediction.pmacs_ready import (
    PreparationError, main, prepare, validate_study_config, verify_manifest,
)


class ReadyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / 'personal'
        self.code = self.workspace / 'code'
        self.package = self.code / 'src' / 'ftd_prediction'
        self.package.mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / 'src' / 'ftd_prediction'
        for name in ('__init__.py', 'pmacs_ready.py', 'cluster.py'):
            (self.package / name).write_bytes((source / name).read_bytes())
        for name in ('descriptive.py', 'neural.py'):
            (self.package / name).write_text('raise RuntimeError("stub must never execute")\n')
        self.manifest = self.code / 'MANIFEST.json'
        self.make_manifest()
        self.run = self.workspace / 'runs'
        self.run.mkdir(mode=0o700)
        self.request, self.output = self.run / 'request', self.run / 'output'
        self.raw = self.root / 'canonical'
        self.raw.mkdir()
        self.assay, self.mapping = self.raw / 'source.h5ad', self.raw / 'crosswalk.csv'
        self.assay.write_bytes(b'invented assay sentinel; not a real H5AD')
        self.mapping.write_bytes(b'invented crosswalk sentinel')
        self.study = {
            'schema_version': 1, 'data_kind': 'study', 'assay': 'rna',
            'input_file': str(self.assay), 'canonical_input_roots': [str(self.raw)],
            'count_source': 'layers/counts',
            'counts_provenance': {'kind': 'measured_umi', 'evidence_ref': 'synthetic-record'},
            'mapping_verified': True, 'mapping_evidence_ref': 'synthetic-mapping-record',
            'obs_columns': {'sample_id': 'sample'}, 'mapping_file': str(self.mapping),
            'mapping_keys': ['sample_id'], 'shared_preprocessing_scope': 'synthetic-only',
            'source_version': 'synthetic-v1', 'seed': 11,
        }
        self.study_path, self.cluster_path = self.workspace / 'study.json', self.workspace / 'cluster.json'
        self.cluster = {
            'workspace': str(self.workspace), 'code_directory': str(self.code),
            'run_directory': str(self.run), 'python_executable': sys.executable,
            'queue': 'synthetic', 'job_name': 'synthetic_ready', 'cpu_slots': 2,
            'wall_time': '00:05', 'memory_semantics_verified': True,
            'run_directory_precreated': True, 'memory_units': 'MB', 'memory_scope': 'per_job',
            'resource_expression': 'rusage[mem=256] span[hosts=1]',
            'arguments': ['preflight', '--output', str(self.output)],
        }
        self.save_configs()

    def make_manifest(self):
        record = {'schema_version': 1, 'files': [
            {'path': p.relative_to(self.code).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
             'size_bytes': p.stat().st_size} for p in sorted(self.code.rglob('*.py'))]}
        self.manifest.write_text(json.dumps(record))
        return record

    def save_configs(self):
        for path, data in ((self.study_path, self.study), (self.cluster_path, self.cluster)):
            path.write_text(json.dumps(data))
            path.chmod(0o600)

    def prepare(self, **kwargs):
        return prepare(self.cluster_path, self.study_path, self.manifest, sys.executable,
                       self.request, self.output, **kwargs)

    def cli_args(self):
        return ['prepare', '--cluster-config', str(self.cluster_path), '--study-config', str(self.study_path),
                '--manifest', str(self.manifest), '--interpreter', sys.executable,
                '--request-directory', str(self.request), '--output', str(self.output)]

    def startup_verify(self):
        receipt = json.loads((self.request / 'receipt.json').read_text())
        return subprocess.run(receipt['verification_argv'], capture_output=True, text=True, timeout=10)

    def test_private_preparation_never_reads_assay_or_crosswalk(self):
        real_open = Path.open
        touched = []

        def guarded_open(path, *args, **kwargs):
            if path in (self.assay, self.mapping):
                touched.append(path)
                raise AssertionError('Assay/crosswalk content access is forbidden')
            return real_open(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded_open):
            receipt_path = self.prepare()
        self.assertEqual(touched, [])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.request.stat().st_mode & 0o777, 0o700)
        for file in self.request.iterdir():
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
        record = json.loads(receipt_path.read_text())
        self.assertEqual(record['lsf_job_id'], 'pending')
        self.assertEqual(record['config_sha256'], hashlib.sha256(self.study_path.read_bytes()).hexdigest())
        self.assertEqual(record['resources']['resource_expression'], self.cluster['resource_expression'])
        self.assertEqual(record['argv'][2], 'ftd_prediction.descriptive')
        script = (self.request / 'job.lsf').read_text()
        for value in ('umask 077', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=2',
                      'LSB_JOBID', str(self.request / 'cache'), 'exec ' + sys.executable):
            self.assertIn(value, script)
        self.assertNotIn('bsub ', script)
        self.assertNotIn('ssh ', script)
        self.assertEqual(self.startup_verify().returncode, 0)

    def test_unresolved_or_missing_provenance_rejected_before_input_path_checks(self):
        changes = [
            {'source_version': marker} for marker in ('REPLACE_VERSION', 'UNVERIFIED', 'placeholder', '<version>')
        ] + [
            {'counts_provenance': {'kind': 'measured_umi', 'evidence_ref': ''}},
            {'counts_provenance': {'kind': 'measured_transcripts', 'evidence_ref': 'record'}},
            {'mapping_verified': False}, {'mapping_evidence_ref': ''}, {'source_version': ''},
            {'shared_preprocessing_scope': ''}, {'counts_provenance': {'kind': 'measured_umi',
                'evidence_ref': 'record', 'nested': {'status': ['UNVERIFIED']}}},
        ]
        for change in changes:
            config = {**self.study, **change}
            with self.subTest(change=change), patch.object(Path, 'is_file', side_effect=AssertionError('path access')):
                with self.assertRaises(PreparationError):
                    validate_study_config(config, 'ftd_prediction.descriptive')

    def test_descriptive_cannot_request_training(self):
        for key, value in (('model', {'max_epochs': 1}), ('split_strategy', 'leave_one_donor_out'),
                           ('fold_indices', [0]), ('train', True)):
            with self.subTest(key=key), self.assertRaises(PreparationError):
                validate_study_config({**self.study, key: value}, 'ftd_prediction.descriptive')

    def test_count_contract_matches_loader_and_unknown_preprocessing_is_truthful(self):
        from ftd_prediction.h5ad_io import allowed_count_kinds
        for assay in ('rna', 'spatial'):
            base = self.study if assay == 'rna' else self.spatial_config()
            for kind in allowed_count_kinds(assay):
                config = {**base, 'counts_provenance': {'kind': kind, 'evidence_ref': 'verified-source-record'},
                          'shared_preprocessing_scope': 'unverified'}
                validate_study_config(config, 'ftd_prediction.descriptive')
        for field in ('mapping_evidence_ref', 'source_version'):
            with self.subTest(field=field), self.assertRaises(PreparationError):
                validate_study_config({**self.study, field: 'unverified'}, 'ftd_prediction.descriptive')
        with self.assertRaises(PreparationError):
            validate_study_config({**self.study, 'shared_preprocessing_scope': 'unverified',
                                   'mapping_verified': False}, 'ftd_prediction.descriptive')

    def test_spatial_descriptor_budget_is_optional_but_positive_integer(self):
        base = {**self.spatial_config(), 'spatial_graph': {'radius_um': 18.0, 'max_neighbors': 16}}
        validate_study_config(base, 'ftd_prediction.descriptive')
        for budget in (128 * 1024**2, 1):
            config = {**base, 'spatial_graph': {**base['spatial_graph'], 'max_descriptor_bytes': budget}}
            validate_study_config(config, 'ftd_prediction.descriptive')
        for budget in (0, -1, True, 1.0, '128'):
            config = {**base, 'spatial_graph': {**base['spatial_graph'], 'max_descriptor_bytes': budget}}
            with self.subTest(budget=budget), self.assertRaises(PreparationError):
                validate_study_config(config, 'ftd_prediction.descriptive')

    def spatial_config(self):
        return {**self.study, 'assay': 'spatial', 'coordinate_units': 'um', 'spatial_key': 'spatial',
                'panel_ref': 'synthetic-panel-v1', 'segmentation_ref': 'synthetic-segmentation-v1',
                'coordinate_transform_ref': 'synthetic-identity-transform-record'}

    def test_spatial_requires_coordinate_panel_segmentation_and_transform_records(self):
        config = self.spatial_config()
        validate_study_config(config, 'ftd_prediction.descriptive')
        for key in ('coordinate_units', 'spatial_key', 'panel_ref', 'segmentation_ref', 'coordinate_transform_ref'):
            invalid = dict(config)
            invalid.pop(key)
            with self.subTest(key=key), self.assertRaises(PreparationError):
                validate_study_config(invalid, 'ftd_prediction.descriptive')
        config['cell_cap_per_donor'] = 10
        with self.assertRaises(PreparationError):
            validate_study_config(config, 'ftd_prediction.descriptive')

    def test_neural_checkpoint_is_required_only_for_spatial_and_never_opened(self):
        rna = {**self.study, 'model': {'accelerator': 'cpu'}, 'split_strategy': 'leave_one_donor_out'}
        validate_study_config(rna, 'ftd_prediction.neural')
        spatial = {**rna, **self.spatial_config(), 'model': {'mode': 'frozen'}}
        with self.assertRaises(PreparationError):
            validate_study_config(spatial, 'ftd_prediction.neural')
        checkpoint = self.workspace / 'checkpoint'
        checkpoint.mkdir()
        for name in ('config.json', 'model.safetensors'):
            (checkpoint / name).touch()
        spatial['model']['checkpoint_path'] = str(checkpoint)
        with patch.object(Path, 'open', side_effect=AssertionError('content access')):
            validate_study_config(spatial, 'ftd_prediction.neural')

    def test_output_and_request_reuse_refused_without_overwrite(self):
        self.output.mkdir()
        with self.assertRaises(PreparationError):
            self.prepare()
        self.assertFalse(self.request.exists())
        self.output.rmdir()
        self.prepare()
        receipt = (self.request / 'receipt.json').read_bytes()
        with self.assertRaises(PreparationError):
            self.prepare()
        self.assertEqual((self.request / 'receipt.json').read_bytes(), receipt)

    def test_private_configs_and_memory_semantics_remain_required(self):
        self.study_path.chmod(0o644)
        with self.assertRaises(PreparationError):
            self.prepare()
        self.assertTrue((self.request / 'preparation_error.log').is_file())
        self.assertFalse((self.request / 'job.lsf').exists())
        # Different fresh request, preserving the failure evidence.
        self.request = self.run / 'request2'
        self.study_path.chmod(0o600)
        self.cluster['memory_semantics_verified'] = False
        self.save_configs()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.request / 'job.lsf').exists())

    def test_config_and_code_must_stay_in_personal_workspace(self):
        external = self.root / 'external.json'
        external.write_bytes(self.study_path.read_bytes())
        external.chmod(0o600)
        with self.assertRaises(PreparationError):
            prepare(self.cluster_path, external, self.manifest, sys.executable, self.request, self.output)
        self.assertFalse(self.request.exists())

    def test_manifest_missing_changed_size_hash_and_unlisted_payload_fail(self):
        original = (self.package / 'descriptive.py').read_bytes()
        record = self.make_manifest()
        self.assertEqual(verify_manifest(self.code, self.manifest)['file_count'], 5)
        for mutation in ('changed', 'missing', 'size', 'hash', 'unlisted'):
            (self.package / 'descriptive.py').write_bytes(original)
            altered = copy.deepcopy(record)
            extra = self.code / 'unexpected.py'
            if mutation == 'changed':
                (self.package / 'descriptive.py').write_bytes(b'x' * len(original))
            elif mutation == 'missing':
                (self.package / 'descriptive.py').unlink()
            elif mutation == 'size':
                altered['files'][0]['size_bytes'] += 1
            elif mutation == 'hash':
                altered['files'][0].pop('sha256')
            else:
                extra.touch()
            self.manifest.write_text(json.dumps(altered))
            with self.subTest(mutation=mutation), self.assertRaises(PreparationError):
                verify_manifest(self.code, self.manifest)
            if extra.exists():
                extra.unlink()

    def test_manifest_traversal_absolute_duplicate_and_noncode_paths_fail(self):
        original = self.make_manifest()
        for path in ('../private.py', '/tmp/private.py', 'src/../private.py',
                     'src//ftd_prediction/private.py', 'study.h5ad', 'configs/private.json'):
            record = copy.deepcopy(original)
            record['files'][0]['path'] = path
            self.manifest.write_text(json.dumps(record))
            with self.subTest(path=path), self.assertRaises(PreparationError):
                verify_manifest(self.code, self.manifest)
        original['files'].append(original['files'][0])
        self.manifest.write_text(json.dumps(original))
        with self.assertRaises(PreparationError):
            verify_manifest(self.code, self.manifest)

    def test_manifest_symlink_payload_and_ancestor_rejected(self):
        path = self.package / 'descriptive.py'
        path.unlink()
        path.symlink_to(self.assay)
        with self.assertRaises(PreparationError):
            verify_manifest(self.code, self.manifest)
        path.unlink()
        path.write_text('stub')
        self.make_manifest()
        link = self.workspace / 'code_link'
        link.symlink_to(self.code, target_is_directory=True)
        with self.assertRaises(PreparationError):
            verify_manifest(link, link / 'MANIFEST.json')
        extra = self.code / 'linked_directory'
        extra.symlink_to(self.raw, target_is_directory=True)
        with self.assertRaises(PreparationError):
            verify_manifest(self.code, self.manifest)

    def test_job_startup_refuses_original_frozen_or_cluster_config_mutation(self):
        self.prepare()
        for path in (self.study_path, self.request / 'study.json', self.cluster_path):
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            result = self.startup_verify()
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(str(path), result.stdout + result.stderr)
            path.write_bytes(original)
        self.assertEqual(self.startup_verify().returncode, 0)

    def test_generated_shell_checks_lsf_context_and_new_output_before_runner(self):
        self.prepare()
        script = self.request / 'job.lsf'
        syntax = subprocess.run(['/bin/bash', '-n', str(script)], capture_output=True, timeout=10)
        self.assertEqual(syntax.returncode, 0)
        environ = {k: v for k, v in os.environ.items() if k != 'LSB_JOBID'}
        outside_lsf = subprocess.run(['/bin/bash', str(script)], env=environ, capture_output=True, timeout=10)
        self.assertNotEqual(outside_lsf.returncode, 0)
        self.assertIn(b'lsf_context', outside_lsf.stdout)
        self.assertFalse((self.request / 'cache').exists())
        self.output.mkdir()
        in_lsf = subprocess.run(['/bin/bash', str(script)], env={**environ, 'LSB_JOBID': '123'},
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(in_lsf.returncode, 0)
        self.assertNotIn('stub must never execute', in_lsf.stderr)
        self.assertFalse((self.request / 'cache').exists())

    def test_job_startup_refuses_changed_code_manifest_and_verifier(self):
        self.prepare()
        original_manifest = self.manifest.read_bytes()
        self.manifest.write_bytes(original_manifest + b'\n')
        self.assertNotEqual(self.startup_verify().returncode, 0)
        self.manifest.write_bytes(original_manifest)
        (self.package / 'descriptive.py').write_text('raise RuntimeError("changed")\n')
        self.assertNotEqual(self.startup_verify().returncode, 0)
        self.make_manifest()
        (self.package / 'pmacs_ready.py').write_text('print("BYPASSED")\n')
        result = self.startup_verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('BYPASSED', result.stdout)

    def test_job_startup_refuses_output_ancestor_replaced_by_external_symlink(self):
        parent = self.run / 'targets'
        parent.mkdir(mode=0o700)
        self.output = parent / 'result'
        self.prepare()
        parent.rename(self.run / 'targets_preserved')
        external = self.root / 'outside_workspace'
        external.mkdir()
        parent.symlink_to(external, target_is_directory=True)
        self.assertNotEqual(self.startup_verify().returncode, 0)
        result = subprocess.run(['/bin/bash', str(self.request / 'job.lsf')],
            env={**os.environ, 'LSB_JOBID': '123'}, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('stub must never execute', result.stderr)
        self.assertNotIn(str(external), result.stdout + result.stderr)
        self.assertFalse((self.request / 'cache').exists())
        self.assertFalse((external / 'result').exists())

    def test_optional_environment_lock_must_exist_and_is_pinned(self):
        lock = self.workspace / 'environment-lock.txt'
        with self.assertRaises(FileNotFoundError):
            self.prepare(environment_lock=lock)
        self.assertFalse((self.request / 'job.lsf').exists())
        self.request = self.run / 'request2'
        lock.write_text('invented test environment record')
        self.prepare(environment_lock=lock)
        self.assertEqual(self.startup_verify().returncode, 0)
        lock.write_text('changed test environment record')
        self.assertNotEqual(self.startup_verify().returncode, 0)

    def test_cli_stdout_is_sanitized_and_private_errors_retained(self):
        self.study['input_file'] = str(self.raw / 'SENSITIVE_NOT_PRESENT.h5ad')
        self.save_configs()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(self.cli_args())
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(stdout.getvalue())['status'], 'failed')
        self.assertNotIn(str(self.workspace), stdout.getvalue() + stderr.getvalue())
        self.assertNotIn('SENSITIVE', stdout.getvalue() + stderr.getvalue())
        self.assertTrue((self.request / 'preparation_error.log').is_file())
        self.assertEqual((self.request / 'preparation_error.log').stat().st_mode & 0o777, 0o600)
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(['prepare', '--SENSITIVE-bad-option']), 1)
        self.assertNotIn('SENSITIVE', output.getvalue() + errors.getvalue())

    def test_duplicate_cli_options_and_assay_digest_requests_are_rejected(self):
        for arguments in (self.cli_args() + ['--output', str(self.output)],
                          ['verify', '--code-directory', str(self.code), '--manifest', str(self.manifest),
                           '--file-digest', str(self.assay), '0' * 64]):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(arguments), 1)
            self.assertNotIn(str(self.workspace), stdout.getvalue())

    def test_examples_are_deliberately_blocked_and_omit_spatial_graph(self):
        repo = Path(__file__).resolve().parents[1]
        for assay in ('rna', 'spatial'):
            config = json.loads((repo / 'configs' / f'descriptive-{assay}.example.json').read_text())
            self.assertIs(config['mapping_verified'], False)
            self.assertEqual(config['mapping_evidence_ref'], '')
            self.assertNotIn('model', config)
            self.assertNotIn('spatial_graph', config)
            with self.assertRaises(PreparationError):
                validate_study_config(config, 'ftd_prediction.descriptive')


if __name__ == '__main__':
    unittest.main()
