"""Finance adaptation switches, fallback behavior and gradients."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml

from layers.TimeFilter_layers import GraphFilter
from models.TimeFilter import Model, PatchEmbed
from scripts.run_financial import build_command
from utils.finance_adaptation_config import normalize_finance_adaptation
from utils.financial_losses import daily_pearson_ic_loss


ROOT = Path(__file__).resolve().parents[1]


def settings(**overrides):
    values = dict(task_name='long_term_forecast', seq_len=4, pred_len=1,
                  c_out=3, enc_in=3, d_model=8, d_ff=16, patch_len=2,
                  alpha=0.5, top_p=0.0, pos=1, n_heads=2, e_layers=1,
                  dropout=0.0, financial_norm=0, financial_input_features='eod5')
    values.update(overrides)
    return SimpleNamespace(**values)


class FinanceAdaptationTests(unittest.TestCase):
    def test_config_and_command_keep_switches_and_default_is_disabled(self):
        cli = argparse.Namespace(config=str(ROOT / 'config.yaml'), dataset=None,
                                 mode=None, checkpoint=None, batch_size=None,
                                 train_epochs=None, learning_rate=None, moe_aux_weight=None)
        command = build_command(cli)
        parsed = json.loads(command[command.index('--finance_adaptation') + 1])
        configured = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
        self.assertEqual(parsed, normalize_finance_adaptation(configured['finance_adaptation']))
        self.assertFalse(normalize_finance_adaptation(None)['enabled'])
        with self.assertRaisesRegex(ValueError, 'mapping'):
            normalize_finance_adaptation([])
        with self.assertRaisesRegex(ValueError, 'unknown keys'):
            normalize_finance_adaptation({'unexpected': 1})

    def test_disabled_state_and_output_match_legacy_invocation(self):
        torch.manual_seed(3)
        original = Model(settings())
        torch.manual_seed(3)
        disabled = Model(settings(finance_adaptation={'enabled': False}))
        self.assertEqual(original.state_dict().keys(), disabled.state_dict().keys())
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, disabled.state_dict()[name])
        x = torch.randn(1, 4, 3, 5)
        original.eval()
        disabled.eval()
        torch.testing.assert_close(original(x, None)[0], disabled(x, None)[0])

    def test_patch_positions_repeat_for_each_stock(self):
        embed = PatchEmbed(8, 2, pos=True, position_mode='patch_only', patches_per_variable=2)
        with torch.no_grad():
            embed.patch_proj.weight.zero_()
            embed.patch_proj.bias.zero_()
        output = embed(torch.zeros(1, 12))
        torch.testing.assert_close(output[:, 0], output[:, 2])
        torch.testing.assert_close(output[:, 1], output[:, 3])
        self.assertFalse(torch.equal(output[:, 0], output[:, 1]))

    def test_strict_graph_removes_edges_and_preserves_self_loop(self):
        layer = GraphFilter(8, n_vars=3, n_heads=2, top_p=0.0,
                            dropout=0.0, strict_masked_softmax=True)
        seen = []
        layer.graph_conv.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0].detach()))
        x = torch.randn(2, 6, 8, requires_grad=True)
        output, _ = layer(x, alpha=0.5)
        expected = torch.eye(6).expand(2, 2, 6, 6)
        torch.testing.assert_close(seen[0], expected)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_adapter_and_ic_loss_backward(self):
        finance = normalize_finance_adaptation({
            'enabled': True,
            'input_adapter': {'enabled': True, 'input_layernorm': True,
                              'indicator_mixer': {'enabled': True},
                              'temporal_mixer': {'enabled': True}},
            'positional_encoding': {'mode': 'patch_only'},
            'graph': {'strict_masked_softmax': True},
            'loss': {'ic_weight': 0.05},
        })
        model = Model(settings(finance_adaptation=finance))
        x = torch.randn(2, 4, 3, 5)
        target = torch.randn(2, 1, 3)
        pred, _ = model(x, None, is_training=True)
        loss = pred.square().mean() + 0.05 * daily_pearson_ic_loss(pred, target, torch.ones_like(target))
        loss.backward()
        self.assertEqual(tuple(pred.shape), (2, 1, 3))
        self.assertIsNotNone(model.finance_input_adapter.indicator_mixer[1].weight.grad)
        self.assertIsNotNone(model.finance_input_adapter.temporal_mixer[1].weight.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_ic_loss_ignores_masked_stock(self):
        pred = torch.tensor([[[0., 1., 100.]]], requires_grad=True)
        true = torch.tensor([[[0., 1., -100.]]])
        mask = torch.tensor([[[1., 1., 0.]]])
        loss = daily_pearson_ic_loss(pred, true, mask)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        loss.backward()
        self.assertEqual(pred.grad[0, 0, 2].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
