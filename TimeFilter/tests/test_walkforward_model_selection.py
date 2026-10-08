"""Focused checks for the opt-in four-rule historical selection audit."""

import argparse
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from scripts.run_walkforward_model_selection import load_plan, make_command, collect_summary
from utils.walkforward_model_selection import (ValidationRuleCheckpoints,
                                               top5_return_metrics, _epoch_diagnostics,
                                               evaluate_selected_future)
from utils.financial_report import financial_metrics
from utils.financial_checkpoint import resolve_financial_checkpoint
from utils.financial_selection import best_epoch_for_checkpoint
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


ROOT = Path(__file__).resolve().parents[1]


class Report:
    def __init__(self, path):
        self.path = Path(path)

    def write_json(self, name, data):
        (self.path / name).write_text(json.dumps(data), encoding='utf-8')


class WalkforwardModelSelectionTests(unittest.TestCase):
    def test_frozen_command_and_pretest_fold_boundaries(self):
        plan, base, _ = load_plan(ROOT / 'walkforward_model_selection.yaml')
        self.assertEqual(len(plan['folds']), 3)
        self.assertLessEqual(max(f['future_end'] for f in plan['folds']), 1259)
        with tempfile.TemporaryDirectory() as tmp:
            command = make_command(base, plan['folds'][0], 2021, Path(tmp), False, .5)
        self.assertIn('--financial_model_selection', command)
        self.assertNotIn('--financial_walkforward', command)
        self.assertEqual(json.loads(command[command.index('--financial_split') + 1]),
                         {'train_end': 628, 'valid_end': 881, 'future_end': 1007})
        self.assertEqual(command[command.index('--financial_stability_lambda') + 1], '0.5')
        self.assertEqual(command[command.index('--financial_test_each_epoch') + 1], '0')
        self.assertEqual(json.loads(command[command.index('--finance_adaptation') + 1])['loss']['ic_weight'], .01)
        self.assertEqual(command[command.index('--rank_weight') + 1], '10')

    def test_plan_rejects_formal_test_overlap(self):
        original = yaml.safe_load((ROOT / 'walkforward_model_selection.yaml').read_text(encoding='utf-8'))
        original['folds'][-1]['future_end'] = 1260
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.yaml'
            path.write_text(yaml.safe_dump(original), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'before day 1259'):
                load_plan(path)

    def test_four_rules_use_validation_only_and_independent_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            report = Report(folder)
            model = torch.nn.Linear(1, 1, bias=False)
            selector = ValidationRuleCheckpoints(folder, report, .5)
            first = selector.add_stability_score({'IC': .3, 'RankIC': .2,
                                                  'RankIC_early': .1, 'RankIC_late': .3})
            second = selector.add_stability_score({'IC': .1, 'RankIC': .4,
                                                   'RankIC_early': .4, 'RankIC_late': .4})
            self.assertAlmostEqual(first['stable_rankic_score'], .15)
            self.assertAlmostEqual(second['stable_rankic_score'], .4)
            with torch.no_grad():
                model.weight.fill_(1)
            torch.save(model.state_dict(), folder / 'best.pth')
            selector.update(model, 1, first)
            with torch.no_grad():
                model.weight.fill_(2)
            selector.update(model, 2, second)
            with self.assertRaisesRegex(ValueError, 'validation metrics only'):
                selector.update(model, 3, {**second, 'future_RankIC': .9})
            selector.finish({'epoch': 1, 'value': .01, 'validation_metrics': first})
            summary = json.loads((folder / 'selection_summary.json').read_text())
            self.assertEqual([summary['rules'][key]['selected_epoch'] for key in 'ABCD'],
                             [1, 1, 2, 2])
            self.assertEqual(torch.load(folder / 'best_rule_A.pth', weights_only=True)['weight'].item(), 1)
            self.assertEqual(torch.load(folder / 'best_rule_B.pth', weights_only=True)['weight'].item(), 1)
            self.assertEqual(torch.load(folder / 'best_rule_C.pth', weights_only=True)['weight'].item(), 2)
            self.assertEqual(torch.load(folder / 'best_rule_D.pth', weights_only=True)['weight'].item(), 2)

    def test_top5_return_is_distinct_from_sharpe(self):
        pred = np.tile(np.arange(6, dtype=float)[:, None], (1, 3))
        true = np.asarray([[0, 0, 0], [0, 0, 0], [.1, .2, .3],
                           [.1, .2, .3], [.1, .2, .3], [.1, .2, .3]])
        result, top5, universe = top5_return_metrics(pred, true)
        self.assertAlmostEqual(result['mean_top5_return'], top5.mean())
        self.assertAlmostEqual(result['mean_top5_excess_return'], (top5 - universe).mean())
        self.assertGreater(result['mean_top5_excess_return'], 0)

    def test_future_epoch_diagnostics_are_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with (folder / 'epoch_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['epoch', 'val_IC', 'val_RankIC',
                                                              'val_stockmixer_val_loss'])
                writer.writeheader()
                writer.writerow({'epoch': 1, 'val_IC': .1, 'val_RankIC': .2,
                                 'val_stockmixer_val_loss': .3})
            diagnostics = _epoch_diagnostics(folder)
            self.assertIsNone(diagnostics['IC_to_IC'])
            self.assertFalse((folder / 'future_epoch_metrics.csv').exists())
            self.assertTrue((folder / 'selection_epoch_metrics.csv').exists())

    def test_selected_future_uses_frozen_weights_and_same_targets(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, batch_x, masks, is_training=False):
                return batch_x * self.scale, None

        class FakeExp:
            def __init__(self, root, report):
                self.model = Model()
                self.device = torch.device('cpu')
                self.masks = None
                self.args = SimpleNamespace(checkpoints=str(root / 'checkpoints'), pred_len=1)
                self._financial_report = Report(report)

            def _get_data(self, flag):
                self_test.assertEqual(flag, 'test')
                dataset = SimpleNamespace(target_start=20, target_end=22,
                                          START_DAY=915, seq_len=16, pred_len=1)
                return dataset, [(batch_x, batch_y, None, torch.zeros_like(batch_y))]

            def _financial_mask(self, batch_y_mark, true):
                return torch.ones_like(true)

        self_test = self
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / 'results' / 'setting' / 'financial'
            checkpoint_dir = root / 'checkpoints' / 'setting'
            report.mkdir(parents=True)
            checkpoint_dir.mkdir(parents=True)
            batch_x = torch.tensor([[[1., 2., 3., 4., 5., 6.]],
                                    [[6., 5., 4., 3., 2., 1.]]])
            batch_y = torch.tensor([[[.01, -.02, .03, -.01, .02, .04]],
                                    [[-.01, .02, -.03, .01, .04, .02]]])
            exp = FakeExp(root, report)
            pred = batch_x.detach().numpy()
            true = batch_y.detach().numpy()
            metrics = financial_metrics(pred, true, np.ones_like(true))
            (report / 'test_metrics.json').write_text(json.dumps(metrics))
            np.savez_compressed(report / 'test_predictions.npz',
                                prediction=pred[:, 0, :].T, ground_truth=true[:, 0, :].T,
                                mask=np.ones((6, 2)), target_index=np.arange(20, 22))
            (report / 'selection_summary.json').write_text(json.dumps({
                'rules': {rule: {'checkpoint': f'best_rule_{rule}.pth'} for rule in 'ABCD'}}))
            with (report / 'epoch_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['epoch', 'val_IC', 'test_IC'])
                writer.writeheader()
                writer.writerow({'epoch': 1, 'val_IC': .1, 'test_IC': ''})
            for rule, scale in [('B', 2.), ('C', 3.), ('D', 4.)]:
                with torch.no_grad():
                    exp.model.scale.fill_(scale)
                torch.save(exp.model.state_dict(), checkpoint_dir / f'best_rule_{rule}.pth')
            evaluate_selected_future(exp, 'setting')
            for rule in 'ABCD':
                result = json.loads((report / f'future_metrics_rule_{rule}.json').read_text())
                self.assertIn('mean_top5_excess_return', result)
                with np.load(report / f'future_predictions_rule_{rule}.npz') as saved:
                    np.testing.assert_array_equal(saved['ground_truth'], true[:, 0, :].T)
                    np.testing.assert_array_equal(saved['target_index'], np.arange(20, 22))

    def test_one_training_trajectory_saves_four_validation_checkpoints(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x, masks, is_training=False):
                return x[:, -1:, :] * self.scale, x.new_zeros(())

        rng = np.random.default_rng(51)
        x = torch.tensor(rng.normal(0, .01, (4, 16, 12)), dtype=torch.float32)
        y = torch.tensor(rng.normal(0, .01, (4, 1, 12)), dtype=torch.float32)
        marks = torch.zeros((4, 1, 1))
        batch = (x, y, marks, marks)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = []

            def get_data(flag):
                requested.append(flag)
                return range(4), [batch]

            exp = object.__new__(Exp_Long_Term_Forecast)
            exp.args = SimpleNamespace(
                data='SP500', features='M', pred_len=1,
                checkpoints=str(root / 'checkpoints'),
                financial_output_dir=str(root / 'output'), learning_rate=1e-4,
                financial_optimizer='adamw', financial_weight_decay=1e-4,
                financial_grad_clip_norm=1.0, train_epochs=1, patience=3,
                moe_aux_weight=0.0, rank_weight=0.0,
                finance_adaptation={'enabled': False},
                financial_selection='stockmixer_val_loss',
                stockmixer_selection_rank_weight=.1,
                financial_validation_only=False, financial_test_each_epoch=0,
                financial_walkforward=False, financial_model_selection=True,
                financial_stability_lambda=.5, gradient_diagnostic_epochs=[],
                use_amp=False, lradj='cosine', c_out=12)
            exp.device = torch.device('cpu')
            exp.model = TinyModel()
            exp.masks = None
            exp._get_data = get_data
            exp.train('tiny')
            self.assertEqual(requested, ['train', 'val'])
            checkpoints = root / 'checkpoints' / 'tiny'
            for rule in 'ABCD':
                self.assertTrue((checkpoints / f'best_rule_{rule}.pth').is_file())
            report = root / 'output' / 'results' / 'tiny' / 'financial'
            with (report / 'epoch_metrics.csv').open(newline='', encoding='utf-8') as stream:
                row = next(csv.DictReader(stream))
            self.assertIn('val_stable_rankic_score', row)
            self.assertNotIn('test_IC', row)
            chosen = json.loads((report / 'selection_summary.json').read_text())
            self.assertEqual({entry['selected_epoch'] for entry in chosen['rules'].values()}, {1})

    def test_summary_has_one_row_per_fold_seed_rule(self):
        plan, _, _ = load_plan(ROOT / 'walkforward_model_selection.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold in plan['folds']:
                folder = root / fold['name'] / 'seed_2021' / 'results' / 'setting' / 'financial'
                folder.mkdir(parents=True)
                rules = {rule: {'selected_epoch': 7, 'metric': rule,
                                'validation_score': .1} for rule in 'ABCD'}
                (folder / 'selection_summary.json').write_text(json.dumps({'rules': rules}))
                for rule in 'ABCD':
                    metrics = {key: .1 for key in ('mse', 'mae', 'IC', 'RIC', 'RankIC',
                                                 'RankIC_std', 'RankICIR', 'prec_10',
                                                 'sharpe5', 'directional_accuracy',
                                                 'mean_top5_return', 'mean_universe_return',
                                                 'mean_top5_excess_return')}
                    (folder / f'future_metrics_rule_{rule}.json').write_text(json.dumps(metrics))
            summary = collect_summary(root, plan)
            self.assertEqual(len(summary['rows']), 12)
            self.assertEqual(len({(row['fold'], row['seed'], row['rule'])
                                  for row in summary['rows']}), 12)
            self.assertAlmostEqual(summary['by_rule']['A']['mean_future_RankIC'], .1)

    def test_selected_checkpoint_can_be_independently_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'checkpoints' / 'setting' / 'best_rule_D.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            split = {'train_end': 628, 'valid_end': 881, 'future_end': 1007}
            (root / 'config.yaml').write_text(yaml.safe_dump({
                'is_training': 1, 'data': 'SP500', 'financial_model_selection': True,
                'financial_stability_lambda': .5, 'financial_split': split,
            }), encoding='utf-8')
            report = root / 'results' / 'setting' / 'financial'
            report.mkdir(parents=True)
            (report / 'selection_summary.json').write_text(json.dumps({
                'rules': {'D': {'checkpoint': checkpoint.name, 'selected_epoch': 42}},
            }), encoding='utf-8')
            args = SimpleNamespace(financial_checkpoint=str(checkpoint),
                                   checkpoints=str(root / 'checkpoints'), data='SP500',
                                   financial_model_selection=True,
                                   financial_stability_lambda=.5, financial_split=split)
            self.assertEqual(best_epoch_for_checkpoint(checkpoint), 42)
            self.assertEqual(resolve_financial_checkpoint(args, 'setting', root), checkpoint)
            args.financial_split = {**split, 'valid_end': 882}
            with self.assertRaisesRegex(ValueError, 'financial_split'):
                resolve_financial_checkpoint(args, 'setting', root)


if __name__ == '__main__':
    unittest.main()
