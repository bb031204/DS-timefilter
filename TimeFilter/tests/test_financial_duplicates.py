"""Prevent both concurrent and sequential duplicate financial launches."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from utils import financial_runtime as runtime


class DuplicateLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(data='SP500', is_training=1,
                                    root_path=str(self.root / 'data'),
                                    moe_aux_weight=0.005, financial_seed=2021)
        self.patch_root = patch.object(runtime, 'PROJECT_ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def saved_run(self, status='completed'):
        directory = self.root / 'outputs' / 'SP500_previous'
        directory.mkdir(parents=True)
        (directory / 'run_status.json').write_text(json.dumps({'status': status}))
        # Historical runs do not contain the newly introduced force-rerun flag.
        (directory / 'config.yaml').write_text(yaml.safe_dump(dict(vars(self.args),
            financial_output_dir=str(directory), financial_config='old.yaml')))
        return directory

    def test_completed_legacy_run_reused_without_training(self):
        self.saved_run()
        self.args.financial_config = 'new.yaml'
        with patch.object(runtime, '_launch_financial') as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.launch_financial(self.args, []), 0)
            launch.assert_not_called()
        self.assertEqual(len(list((self.root / 'outputs').glob('SP500*'))), 1)

    def test_changed_weight_starts_once(self):
        self.saved_run()
        self.args.moe_aux_weight = 0
        with patch.object(runtime, '_launch_financial', return_value=0) as launch:
            self.assertEqual(runtime.launch_financial(self.args, []), 0)
            launch.assert_called_once()

    def test_explicit_rerun_starts_once(self):
        self.saved_run()
        self.args.financial_force_rerun = True
        with patch.object(runtime, '_launch_financial', return_value=0) as launch:
            self.assertEqual(runtime.launch_financial(self.args, []), 0)
            launch.assert_called_once()

    def test_running_duplicate_blocked_even_with_force_and_lock_released(self):
        self.args.financial_force_rerun = True
        key = runtime.experiment_key(vars(self.args))
        with patch.object(runtime, '_launch_financial', return_value=0) as launch:
            with runtime.experiment_lock(self.root / 'outputs', key), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runtime.launch_financial(self.args, []), 2)
                launch.assert_not_called()
            self.assertEqual(runtime.launch_financial(self.args, []), 0)
            launch.assert_called_once()

    def test_failed_run_does_not_block_retry(self):
        self.saved_run('failed')
        with patch.object(runtime, '_launch_financial', return_value=0) as launch:
            self.assertEqual(runtime.launch_financial(self.args, []), 0)
            launch.assert_called_once()

    def test_evaluation_runs_even_when_same_settings_completed(self):
        self.args.is_training = 0
        self.saved_run()
        with patch.object(runtime, '_launch_financial', return_value=0) as launch:
            self.assertEqual(runtime.launch_financial(self.args, []), 0)
            launch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
