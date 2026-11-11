import copy
import json
import tempfile
import unittest
from pathlib import Path

from ftd_prediction.cluster import ClusterConfigError, render_lsf, render_lsf_text


def valid_config():
    workspace = "/project/hipaa_ycheng11lab/synthetic_project/synthetic_user"
    return {
        "workspace": workspace, "code_directory": workspace + "/code",
        "python_executable": workspace + "/env/bin/python", "run_directory": workspace + "/runs/unique_001",
        "queue": "synthetic_queue", "job_name": "synthetic_preflight", "cpu_slots": 4,
        "wall_time": "01:00", "memory_semantics_verified": True, "run_directory_precreated": True,
        "memory_units": "MB", "memory_scope": "per_job",
        "resource_expression": "rusage[mem=16000] span[hosts=1]",
        "arguments": ["preflight", "--output", workspace + "/runs/unique_001/artifacts", "--seed", "11"],
    }


class ClusterTests(unittest.TestCase):
    def test_explicit_configuration_renders_without_submission(self):
        text = render_lsf_text(valid_config())
        self.assertIn("#BSUB -q synthetic_queue", text)
        self.assertIn('rusage[mem=16000] span[hosts=1]', text)
        self.assertIn("exec /project/", text)
        self.assertNotIn("ssh ", text)
        self.assertNotIn("bsub <", text)
        self.assertIn("umask 077", text)

    def test_unverified_semantics_and_uncreated_log_directory_fail(self):
        for key in ("memory_semantics_verified", "run_directory_precreated"):
            config = valid_config()
            config[key] = False
            with self.subTest(key=key), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_placeholder_and_injection_fail_before_writing(self):
        cases = {
            "queue": "queue\n#BSUB -q rogue",
            "resource_expression": 'rusage[mem=16]\n#BSUB -o /elsewhere',
            "python_executable": "/tmp/python;touch_bad",
            "workspace": "/project/REPLACE_PROJECT/user",
            "job_name": "job$(touch /tmp/bad)",
            "code_directory": "/project/hipaa_ycheng11lab/p/../shared",
        }
        for key, value in cases.items():
            config = valid_config()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_run_and_code_must_stay_inside_personal_workspace(self):
        for key in ("run_directory", "code_directory"):
            config = valid_config()
            config[key] = "/project/hipaa_ycheng11lab/another_user/work"
            with self.subTest(key=key), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_only_reviewed_pipeline_subcommands(self):
        config = valid_config()
        config["arguments"] = ["-c", "import os"]
        with self.assertRaises(ClusterConfigError):
            render_lsf_text(config)

    def test_shell_control_argument_is_rejected(self):
        for value in ("$(touch /tmp/not_executed)", "value;touch /tmp/not_executed", "value\n#BSUB -q rogue"):
            config = valid_config()
            config["arguments"] = ["preflight", "--output", value]
            with self.subTest(value=value), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_effective_command_output_stays_inside_unique_run_directory(self):
        outside = "/project/hipaa_ycheng11lab/shared_master/new_artifacts"
        for arguments in (["preflight", "--output", outside], ["preflight", "--output=" + outside]):
            config = valid_config()
            config["arguments"] = arguments
            with self.subTest(arguments=arguments), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_duplicate_or_abbreviated_output_options_rejected(self):
        for suffix in (["--output", "/tmp/elsewhere"], ["--output=/tmp/elsewhere"], ["--out", "/tmp/elsewhere"]):
            config = valid_config()
            config["arguments"] += suffix
            with self.subTest(suffix=suffix), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_output_equals_form_inside_run_directory_is_supported(self):
        config = valid_config()
        config["arguments"] = ["preflight", "--output=" + config["run_directory"] + "/artifacts"]
        self.assertIn("--output=", render_lsf_text(config))

    def test_request_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps(valid_config()))
            output = root / "request.lsf"
            render_lsf(config, output)
            original = output.read_text()
            with self.assertRaises(FileExistsError):
                render_lsf(config, output)
            self.assertEqual(original, output.read_text())

    def test_config_object_not_mutated(self):
        config = valid_config()
        original = copy.deepcopy(config)
        render_lsf_text(config)
        self.assertEqual(config, original)

    def test_explicit_h5ad_and_report_modules(self):
        config = valid_config()
        config['arguments'] = ['--config', config['code_directory'] + '/study.json',
                               '--output', config['run_directory'] + '/artifacts',
                               '--mode=study', '--execution-context', 'pmacs']
        for module in ('ftd_prediction.descriptive', 'ftd_prediction.neural', 'ftd_prediction.molecular'):
            config['module'] = module
            with self.subTest(module=module):
                self.assertIn('-m ' + module, render_lsf_text(config))
        config['module'] = 'ftd_prediction.neural_report'
        config['arguments'] = ['--run', config['run_directory'] + '/rna',
                               '--run', config['run_directory'] + '/spatial',
                               '--output', config['run_directory'] + '/report']
        self.assertIn('-m ftd_prediction.neural_report', render_lsf_text(config))

    def test_descriptive_report_single_run_and_molecular_modes(self):
        config = valid_config()
        config['module'] = 'ftd_prediction.descriptive_report'
        config['arguments'] = ['--run', config['run_directory'] + '/completed',
                               '--output', config['run_directory'] + '/report']
        script = render_lsf_text(config)
        self.assertIn('-m ftd_prediction.descriptive_report', script)
        self.assertIn('PYTHONDONTWRITEBYTECODE=1', script)
        self.assertIn('OMP_NUM_THREADS=' + str(config['cpu_slots']), script)
        config['arguments'] += ['--run', config['run_directory'] + '/another']
        with self.assertRaises(ClusterConfigError):
            render_lsf_text(config)
        config['module'] = 'ftd_prediction.molecular'
        config['arguments'] = ['--config', config['run_directory'] + '/contrast.json',
                               '--output', config['run_directory'] + '/molecular', '--mode', 'reference']
        with self.assertRaises(ClusterConfigError):
            render_lsf_text(config)

    def test_module_allowlist_and_strict_options(self):
        for module in ('os', 'ftd_prediction.descriptive;bad', '-c', None):
            config = valid_config()
            config['module'] = module
            with self.subTest(module=module), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)
        config = valid_config()
        config['module'] = 'ftd_prediction.descriptive'
        base = ['--config', config['code_directory'] + '/study.json',
                '--output', config['run_directory'] + '/artifacts']
        for suffix in (['--conf', '/tmp/c'], ['--config=/tmp/c'], ['--mode', 'study', '--mode=synthetic'],
                       ['--mode', 'study;bad'], ['--execution-context', 'local', '--execution-context=pmacs'],
                       ['--train', 'yes'], ['--output', '/tmp/out']):
            config['arguments'] = base + suffix
            with self.subTest(suffix=suffix), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_core_commands_keep_exact_reviewed_options(self):
        config = valid_config()
        for arguments in (['validate', '--manifest', '/tmp/metadata.json'],
                          ['preflight', '--config', 'configs/synthetic-pilot.json',
                           '--output', config['run_directory'] + '/preflight'],
                          ['pilot', '--manifest=/tmp/metadata.json', '--mode', 'synthetic',
                           '--output', config['run_directory'] + '/pilot'],
                          ['report', '--run', config['run_directory'] + '/pilot',
                           '--output', config['run_directory'] + '/report', '--format', 'html']):
            config['arguments'] = arguments
            self.assertIn('-m ftd_prediction ', render_lsf_text(config))
        for suffix in (['--see', '12'], ['--seed', '12'], ['--config', '/tmp/a', '--config=/tmp/b']):
            config = valid_config()
            config['arguments'] += suffix
            with self.subTest(suffix=suffix), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)


if __name__ == "__main__":
    unittest.main()
