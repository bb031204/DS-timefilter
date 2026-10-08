"""Check that financial gradient probes are reproducible and observational."""

import argparse
import csv
from contextlib import redirect_stdout
from io import StringIO
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
from utils.financial_losses import daily_pearson_ic_loss, stockmixer_rank_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FinancialGradientDiagnosticTests(unittest.TestCase):
    def test_config_passes_rankic_and_diagnostic_settings(self):
        cli = argparse.Namespace(config=str(PROJECT_ROOT / 'config.yaml'), dataset=None,
                                 mode=None, checkpoint=None, batch_size=None,
                                 train_epochs=None, learning_rate=None, moe_aux_weight=None)
        command = build_command(cli)
        self.assertEqual(command[command.index('--financial_selection') + 1], 'stockmixer_val_loss')
        self.assertEqual(command[command.index('--gradient_diagnostic_epochs') + 1:
                                 command.index('--gradient_diagnostic_batch_size')], ['0', '1', '5'])
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
            ic = daily_pearson_ic_loss(predictions, batch[1], torch.ones_like(batch[1]))
            return {'MSE': (mse, 1.0), 'Rank': (rank, 0.1),
                    'IC': (ic, 0.01), 'MoE': (moe, 0.005)}

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
            output = StringIO()
            with redirect_stdout(output):
                record_gradient_diagnostic(model, (inputs, targets), [0, 10], losses, path, 0)
            with (path / 'gradient_diagnostics.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(rows), 20)  # Four losses × five parameter groups.
        self.assertIn('Grad norm | MSE:', output.getvalue())
        self.assertIn('Rank:', output.getvalue())
        self.assertIn('IC:', output.getvalue())
        self.assertIn('MoE:', output.getvalue())
        self.assertIn('/ MSE     | Rank:', output.getvalue())
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

    def test_disabled_ic_is_reported_as_zero_contribution(self):
        model = torch.nn.Linear(1, 1)
        inputs = torch.tensor([[1.0], [2.0]])

        def losses(batch):
            predictions = model(batch)
            mse = predictions.square().mean()
            return {'MSE': (mse, 1.0), 'Rank': (mse, 0.0),
                    'IC': (mse, 0.0), 'MoE': (mse, 0.0)}

        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            with redirect_stdout(output):
                record_gradient_diagnostic(model, inputs, [0, 1], losses, Path(directory), 0)
            with (Path(directory) / 'gradient_diagnostics.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
        ic = next(row for row in rows if row['component'] == 'IC' and row['group'] == 'all')
        self.assertEqual(float(ic['grad_norm']), 0.0)
        self.assertIn('IC: 0 (off)', output.getvalue())


if __name__ == '__main__':
    unittest.main()
