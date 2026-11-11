import copy
import json
import tempfile
import unittest
from pathlib import Path

from ftd_prediction.cluster import ClusterConfigError, render_lsf, render_lsf_text


def valid_config():
    workspace = "/project/synthetic_lab/synthetic_project/synthetic_user"
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
            "code_directory": "/project/synthetic_lab/p/../shared",
        }
        for key, value in cases.items():
            config = valid_config()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ClusterConfigError):
                render_lsf_text(config)

    def test_run_and_code_must_stay_inside_personal_workspace(self):
        for key in ("run_directory", "code_directory"):
            config = valid_config()
            config[key] = "/project/synthetic_lab/another_user/work"
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
        outside = "/project/synthetic_lab/shared_master/new_artifacts"
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


if __name__ == "__main__":
    unittest.main()
