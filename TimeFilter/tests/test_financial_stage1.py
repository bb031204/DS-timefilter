"""Focused checks for the finance-only normalization and ranking objective."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml

from models.TimeFilter import Model
from scripts.run_financial import build_command
from utils.financial_losses import stockmixer_rank_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FinancialStage1Tests(unittest.TestCase):
    def test_config_exposes_only_requested_stage1_changes(self):
        cli = argparse.Namespace(config=str(PROJECT_ROOT / 'config.yaml'), dataset=None,
                                 mode=None, checkpoint=None, batch_size=None,
                                 train_epochs=None, learning_rate=None, moe_aux_weight=None)
        command = build_command(cli)
        self.assertEqual(command[command.index('--financial_norm') + 1], '0')
        self.assertEqual(command[command.index('--rank_weight') + 1], '0.1')
        configured_alpha = yaml.safe_load((PROJECT_ROOT / 'config.yaml').read_text(encoding='utf-8'))['model']['alpha']
        self.assertEqual(command[command.index('--alpha') + 1], str(configured_alpha))
        self.assertEqual(command[command.index('--d_model') + 1], '512')
        self.assertEqual(command[command.index('--batch_size') + 1], '32')

    def test_original_normalization_default_and_financial_bypass(self):
        settings = dict(task_name='long_term_forecast', seq_len=4, pred_len=1,
                        c_out=3, enc_in=3, d_model=8, d_ff=16, patch_len=2,
                        alpha=0.1, top_p=0.5, pos=0, n_heads=2, e_layers=1,
                        dropout=0.0)
        original = Model(SimpleNamespace(**settings))
        bypassed = Model(SimpleNamespace(**settings, financial_norm=0))
        self.assertFalse(original.norm.non_norm)
        self.assertTrue(bypassed.norm.non_norm)
        sample = torch.tensor([[[0.01, -0.02, 0.03], [0.04, 0.05, -0.06]]])
        torch.testing.assert_close(bypassed.norm(sample, 'norm'), sample)
        torch.testing.assert_close(bypassed.norm(sample, 'denorm'), sample)

    def test_rank_loss_stays_within_each_day_and_has_expected_sign(self):
        prediction = torch.tensor([[[1., 2.]], [[2., 1.]]])
        target = prediction.clone()
        mask = torch.ones_like(prediction)
        self.assertEqual(stockmixer_rank_loss(prediction, target, mask).item(), 0)
        reversed_prediction = torch.tensor([[[1., 0.]]], requires_grad=True)
        reversed_target = torch.tensor([[[0., 1.]]])
        loss = stockmixer_rank_loss(reversed_prediction, reversed_target,
                                    torch.ones_like(reversed_target))
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertTrue(torch.any(reversed_prediction.grad != 0))

    def test_rank_loss_ignores_invalid_stocks(self):
        prediction = torch.tensor([[[1., 0., 3.]]], requires_grad=True)
        target = torch.tensor([[[0., 1., -2.]]])
        mask = torch.tensor([[[1., 1., 0.]]])
        loss = stockmixer_rank_loss(prediction, target, mask)
        loss.backward()
        self.assertEqual(prediction.grad[0, 0, 2].item(), 0)
        self.assertAlmostEqual(loss.item(), 2 / 9)

    def test_small_financial_forward_and_rank_backward(self):
        settings = dict(task_name='long_term_forecast', seq_len=4, pred_len=1,
                        c_out=3, enc_in=3, d_model=8, d_ff=16, patch_len=2,
                        alpha=0.7, top_p=0.5, pos=0, n_heads=2, e_layers=1,
                        dropout=0.0, financial_norm=0)
        model = Model(SimpleNamespace(**settings))
        inputs = torch.randn(2, 4, 3) * 0.01
        targets = torch.randn(2, 1, 3) * 0.01
        predictions, moe_loss = model(inputs, None, is_training=True)
        self.assertEqual(tuple(predictions.shape), tuple(targets.shape))
        loss = ((predictions - targets) ** 2).mean()
        loss = loss + 0.1 * stockmixer_rank_loss(predictions, targets, torch.ones_like(targets))
        loss = loss + 0.005 * moe_loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.patch_embed.patch_proj.weight.grad)


if __name__ == '__main__':
    unittest.main()
