import tempfile
from pathlib import Path
import unittest
import torch
from utils.financial_selection import FinancialSelection
from scripts.run_patch_study import choose


class Report:
    def write_json(self, name, data):
        self.data = data


class SelectionTests(unittest.TestCase):
    def test_best_and_last_diverge_on_worse_validation(self):
        with tempfile.TemporaryDirectory() as root:
            report = Report()
            selector = FinancialSelection(root, report)
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1)
            selector.update(model, 1, {'mse': .1})
            with torch.no_grad():
                model.weight.fill_(2)
            selector.update(model, 2, {'mse': .2})
            self.assertEqual(torch.load(selector.finish(), weights_only=True)['weight'].item(), 1)
            self.assertEqual(torch.load(Path(root) / 'last.pth', weights_only=True)['weight'].item(), 2)
            self.assertEqual(report.data['best']['epoch'], 1)

    def test_rank_selection_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as root:
            selector = FinancialSelection(root, Report(), 'RankIC')
            model = torch.nn.Linear(1, 1)
            selector.update(model, 1, {'RankIC': float('nan')})
            with self.assertRaises(ValueError):
                selector.finish()
            selector.update(model, 2, {'RankIC': -.1})
            selector.update(model, 3, {'RankIC': -.2})
            self.assertEqual(selector.best['epoch'], 2)

    def test_patch_choice_ignores_test_and_preserves_tie_order(self):
        rows = [{'validation_value': .2, 'test_value': 100, 'patch_len': 16},
                {'validation_value': .1, 'test_value': -100, 'patch_len': 8},
                {'validation_value': .1, 'test_value': 100, 'patch_len': 4}]
        self.assertEqual(choose(rows, 'mse')['patch_len'], 8)
        self.assertEqual(choose(rows, 'RankIC')['patch_len'], 16)
