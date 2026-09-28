"""Check that financial gradient probes are reproducible and observational."""

import argparse
import csv
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.TimeFilter import Model
from scripts.run_financial import build_command
from utils.financial_gradient_diagnostic import record_gradient_diagnostic
from utils.financial_losses import stockmixer_rank_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FinancialGradientDiagnosticTests(unittest.TestCase):
    def test_config_passes_rankic_and_diagnostic_settings(self):
        cli = argparse.Namespace(config=str(PROJECT_ROOT / 'config.yaml'), dataset=None,
                                 mode=None, checkpoint=None, batch_size=None,
                                 train_epochs=None, learning_rate=None, moe_aux_weight=None)
        command = build_command(cli)
        self.assertEqual(command[command.index('--financial_selection') + 1], 'RankIC')
        self.assertEqual(command[command.index('--gradient_diagnostic_epochs') + 1:
                                 command.index('--gradient_diagnostic_batch_size')], ['0', '5', '50'])
        self.assertEqual(command[command.index('--gradient_diagnostic_batch_size') + 1], '8')

    def test_one_forward_preserves_parameters_grads_mode_and_rng(self):
        settings = dict(task_name='long_term_forecast', seq_len=4, pred_len=1,
                        c_out=3, enc_in=3, d_model=8, d_ff=16, patch_len=2,
                        alpha=0.3, top_p=0.5, pos=0, n_heads=2, e_layers=1,
                        dropout=0.1, financial_norm=0)
        model = Model(SimpleNamespace(**settings))
        model.eval()
        inputs = torch.tensor([[[.01, -.02, .03], [.02, .01, -.01],
                                [.03, -.02, .02], [.01, .03, -.01]]] * 2)
        targets = torch.tensor([[[.03, -.02, .01]], [[-.01, .02, .03]]])
        calls = []

        def losses(batch):
            calls.append(1)
            predictions, moe = model(batch[0], None, is_training=True)
            mse = (predictions - batch[1]).square().mean()
            rank = stockmixer_rank_loss(predictions, batch[1], torch.ones_like(batch[1]))
            return {'MSE': (mse, 1.0), 'Rank': (rank, 0.1), 'MoE': (moe, 0.005)}

        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        for value in model.parameters():
            value.grad = torch.ones_like(value)
        random.seed(113)
        np.random.seed(113)
        torch.manual_seed(113)
        expected = (random.random(), np.random.rand(), torch.rand(()).item())
        random.seed(113)
        np.random.seed(113)
        torch.manual_seed(113)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            record_gradient_diagnostic(model, (inputs, targets), [0, 10], losses, path, 0)
            with (path / 'gradient_diagnostics.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(rows), 15)  # Three losses × five parameter groups.
        self.assertEqual({row['stage'] for row in rows}, {'pretrain'})
        self.assertGreater(float(next(row['grad_norm'] for row in rows
                                      if row['component'] == 'MSE' and row['group'] == 'all')), 0)
        self.assertFalse(model.training)
        self.assertFalse(hasattr(model.norm, 'mean'))
        self.assertFalse(hasattr(model.norm, 'stdev'))
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name])
            torch.testing.assert_close(value.grad, torch.ones_like(value))
        self.assertEqual((random.random(), np.random.rand(), torch.rand(()).item()), expected)


if __name__ == '__main__':
    unittest.main()
