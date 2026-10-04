import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
import torch
from utils.financial_selection import FinancialSelection, best_epoch_for_checkpoint
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

    def test_stockmixer_validation_loss_selects_lower_value(self):
        with tempfile.TemporaryDirectory() as root:
            selector = FinancialSelection(root, Report(), 'stockmixer_val_loss')
            model = torch.nn.Linear(1, 1)
            selector.update(model, 1, {'stockmixer_val_loss': 0.2})
            selector.update(model, 2, {'stockmixer_val_loss': 0.1})
            selector.update(model, 3, {'stockmixer_val_loss': 0.3})
            self.assertEqual(selector.best['epoch'], 2)

    def test_new_best_is_green_and_only_printed_on_improvement(self):
        with tempfile.TemporaryDirectory() as root:
            selector = FinancialSelection(root, Report(), 'stockmixer_val_loss')
            model = torch.nn.Linear(1, 1)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                selector.update(model, 1, {'stockmixer_val_loss': 0.2})
                selector.update(model, 2, {'stockmixer_val_loss': 0.3})
                selector.update(model, 3, {'stockmixer_val_loss': 0.1})
            message = output.getvalue()
            self.assertEqual(message.count('New Best'), 2)
            self.assertIn('\033[32mNew Best | epoch 3', message)
            self.assertIn('best.pth\033[0m', message)

    def test_final_test_epoch_comes_from_checkpoint_selection_record(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            checkpoint = run / 'checkpoints' / 'setting' / 'best.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertIsNone(best_epoch_for_checkpoint(checkpoint))
            record = run / 'results' / 'setting' / 'financial' / 'selection.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({'best': {'epoch': 7},
                                          'final_evaluation_checkpoint': 'best.pth'}), encoding='utf-8')
            self.assertEqual(best_epoch_for_checkpoint(checkpoint), 7)

    def test_patch_choice_ignores_test_and_preserves_tie_order(self):
        rows = [{'validation_value': .2, 'test_value': 100, 'patch_len': 16},
                {'validation_value': .1, 'test_value': -100, 'patch_len': 8},
                {'validation_value': .1, 'test_value': 100, 'patch_len': 4}]
        self.assertEqual(choose(rows, 'mse')['patch_len'], 8)
        self.assertEqual(choose(rows, 'RankIC')['patch_len'], 16)
