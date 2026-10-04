"""Standalone evaluation must load the matching financial best checkpoint."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import yaml

from utils.financial_checkpoint import resolve_financial_checkpoint
from utils.financial_provenance import collect_provenance


class FinancialCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / 'run.py').write_text('original code', encoding='utf-8')
        data = self.project / 'data'
        data.mkdir()
        (data / 'SP500.npy').write_bytes(b'original data')
        self.run = self.project / 'outputs' / 'SP500_test'
        self.checkpoint = self.run / 'checkpoints' / 'setting' / 'best.pth'
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.touch()
        self.args = SimpleNamespace(data='SP500', root_path=str(data), is_training=0,
                                    patch_len=16, moe_aux_weight=0.005,
                                    checkpoints=str(self.run / 'checkpoints'),
                                    financial_checkpoint=None)
        saved = dict(vars(self.args), is_training=1, financial_checkpoint=None)
        (self.run / 'config.yaml').write_text(yaml.safe_dump(saved))
        self.write_provenance()

    def write_provenance(self):
        value = collect_provenance(self.project, self.args.root_path, self.args.data)
        (self.run / 'provenance.json').write_text(json.dumps(value))

    def test_default_resolves_best_and_verifies_config(self):
        self.assertEqual(resolve_financial_checkpoint(self.args, 'setting', self.project), self.checkpoint)

    def test_diagnostic_settings_do_not_block_independent_evaluation(self):
        saved = yaml.safe_load((self.run / 'config.yaml').read_text())
        saved.update(gradient_diagnostic_epochs=[0, 5, 50], gradient_diagnostic_batch_size=8)
        (self.run / 'config.yaml').write_text(yaml.safe_dump(saved))
        self.args.gradient_diagnostic_epochs = []
        self.args.gradient_diagnostic_batch_size = 1
        self.assertEqual(resolve_financial_checkpoint(self.args, 'setting', self.project), self.checkpoint)

    def test_explicit_last_is_rejected(self):
        self.args.financial_checkpoint = str(self.checkpoint.with_name('last.pth'))
        with self.assertRaisesRegex(ValueError, 'best.pth'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)

    def test_wrong_model_config_rejected(self):
        self.args.patch_len = 8
        with self.assertRaisesRegex(ValueError, 'patch_len'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)

    def test_changed_code_warns_but_changed_data_rejected(self):
        (self.project / 'run.py').write_text('changed code', encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()) as message:
            self.assertEqual(resolve_financial_checkpoint(self.args, 'setting', self.project), self.checkpoint)
        self.assertIn('code_sha256', message.getvalue())
        (self.project / 'run.py').write_text('original code', encoding='utf-8')
        (Path(self.args.root_path) / 'SP500.npy').write_bytes(b'changed data')
        with self.assertRaisesRegex(ValueError, 'data_sha256'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)

    def test_legacy_config_assumes_original_weight(self):
        (self.run / 'provenance.json').unlink()
        config = yaml.safe_load((self.run / 'config.yaml').read_text())
        del config['moe_aux_weight']
        (self.run / 'config.yaml').write_text(yaml.safe_dump(config))
        self.args.moe_aux_weight = 0.05
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(resolve_financial_checkpoint(self.args, 'setting', self.project), self.checkpoint)
        self.args.moe_aux_weight = 0.005
        with self.assertRaisesRegex(ValueError, 'moe_aux_weight'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)

    def test_legacy_config_rejects_new_loss_or_normalization(self):
        self.args.rank_weight = 0.1
        with self.assertRaisesRegex(ValueError, 'rank_weight'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)
        self.args.rank_weight = 0.0
        self.args.financial_norm = 0
        with self.assertRaisesRegex(ValueError, 'financial_norm'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)

    def test_checkpoint_rejects_different_financial_input(self):
        self.args.financial_input_features = 'eod5'
        with self.assertRaisesRegex(ValueError, 'financial_input_features'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)

    def test_missing_training_config_rejected(self):
        (self.run / 'config.yaml').unlink()
        with self.assertRaisesRegex(ValueError, 'configuration'):
            resolve_financial_checkpoint(self.args, 'setting', self.project)


if __name__ == '__main__':
    unittest.main()
