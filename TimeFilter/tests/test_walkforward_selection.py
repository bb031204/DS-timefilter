"""Walk-forward isolation and target-day boundary checks."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from data_provider.stockmixer_sp500 import Dataset_SP500
from scripts.run_walkforward_selection import load_plan, make_command
from utils.financial_checkpoint import resolve_financial_checkpoint
from utils.financial_selection import best_epoch_for_checkpoint
from utils.walkforward_selection import WalkforwardRankCheckpoint, epoch_correlations


ROOT = Path(__file__).resolve().parents[1]


class Report:
    def __init__(self, path):
        self.path = Path(path)

    def write_json(self, name, data):
        (self.path / name).write_text(json.dumps(data), encoding='utf-8')


class WalkforwardSelectionTests(unittest.TestCase):
    def test_folds_use_only_past_inputs_and_pretest_targets(self):
        plan, _, _ = load_plan(ROOT / 'walkforward_selection.yaml')
        raw = np.load(ROOT / 'stockmixer_dataset' / 'SP500' / 'SP500.npy', mmap_mode='r')
        close = raw[:, 914:, -1].T
        returns = (close[1:] / close[:-1] - 1).astype(np.float32)
        for fold in plan['folds']:
            args = SimpleNamespace(seq_len=16, label_len=0, pred_len=1, patch_len=8,
                                   augmentation_ratio=0, financial_input_features='eod5',
                                   financial_norm=0, enc_in=474, dec_in=474, c_out=474,
                                   financial_split={key: fold[key] for key in
                                                    ('train_end', 'valid_end', 'future_end')})
            bounds = {'train': (16, fold['train_end']),
                      'val': (fold['train_end'], fold['valid_end']),
                      'test': (fold['valid_end'], fold['future_end'])}
            for flag, (start, end) in bounds.items():
                dataset = Dataset_SP500(args, str(ROOT / 'stockmixer_dataset' / 'SP500'),
                                        flag=flag, size=[16, 0, 1], features='M')
                self.assertEqual((dataset.target_start, dataset.target_end), (start, end))
                self.assertEqual(len(dataset), end - start)
                for index in (0, len(dataset) - 1):
                    day = start + index
                    features, label, _, _ = dataset[index]
                    np.testing.assert_array_equal(label[0], returns[day])
                    np.testing.assert_array_equal(
                        features, raw[:, 915 + day - 16:915 + day, :].transpose(1, 0, 2).astype(np.float32))
            self.assertLessEqual(fold['future_end'], 1259)

    def test_command_uses_frozen_config_and_fold_boundaries(self):
        plan, base, _ = load_plan(ROOT / 'walkforward_selection.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            command = make_command(base, plan['folds'][0], 2021, Path(tmp), True)
        self.assertEqual(command[command.index('--rank_weight') + 1], '5')
        self.assertEqual(command[command.index('--financial_selection') + 1],
                         'stockmixer_val_loss')
        self.assertIn('--financial_walkforward', command)
        self.assertEqual(json.loads(command[command.index('--financial_split') + 1]),
                         {'train_end': 628, 'valid_end': 881, 'future_end': 1007})
        self.assertEqual(command[command.index('--financial_test_each_epoch',
                                               command.index('--financial_walkforward')) + 1], '1')

    def test_rank_candidate_uses_validation_only_and_preserves_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(1, 1, bias=False)
            report = Report(root)
            selector = WalkforwardRankCheckpoint(root, report)
            with torch.no_grad():
                model.weight.fill_(1)
            torch.save(model.state_dict(), root / 'best.pth')
            selector.update(model, 1, {'RankIC': .2, 'test_RankIC': -.5})
            with torch.no_grad():
                model.weight.fill_(2)
            selector.update(model, 2, {'RankIC': .1, 'test_RankIC': .9})
            selector.finish({'epoch': 1, 'value': .01})
            saved = torch.load(root / 'best_rankic.pth', weights_only=True)
            self.assertEqual(saved['weight'].item(), 1)
            saved_a = torch.load(root / 'best_stockmixer_loss.pth', weights_only=True)
            self.assertEqual(saved_a['weight'].item(), 1)
            summary = json.loads((root / 'selection_summary.json').read_text())
            self.assertEqual(summary['B']['epoch'], 1)

    def test_correlations_are_descriptive_and_ignore_missing_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'epoch_metrics.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['val_RankIC', 'test_RankIC',
                                                             'val_IC', 'test_IC', 'val_mse',
                                                             'val_stockmixer_val_loss'])
                writer.writeheader()
                writer.writerow({'val_RankIC': .1, 'test_RankIC': .2,
                                 'val_IC': .1, 'test_IC': .2, 'val_mse': .3,
                                 'val_stockmixer_val_loss': .4})
                writer.writerow({'val_RankIC': .2, 'test_RankIC': .3,
                                 'val_IC': .2, 'test_IC': .3, 'val_mse': .2,
                                 'val_stockmixer_val_loss': .3})
            result = epoch_correlations(tmp)
            self.assertAlmostEqual(result['RankIC'], 1.0)
            self.assertEqual(result['epochs'], 2)

    def test_rank_candidate_can_be_independently_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'checkpoints' / 'setting' / 'best_rankic.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            split = {'train_end': 628, 'valid_end': 881, 'future_end': 1007}
            (root / 'config.yaml').write_text(yaml.safe_dump({
                'is_training': 1, 'data': 'SP500', 'financial_walkforward': True,
                'financial_split': split,
            }), encoding='utf-8')
            report = root / 'results' / 'setting' / 'financial'
            report.mkdir(parents=True)
            (report / 'selection_summary.json').write_text(json.dumps({
                'B_checkpoint': 'best_rankic.pth', 'B': {'epoch': 7},
            }), encoding='utf-8')
            args = SimpleNamespace(financial_checkpoint=str(checkpoint),
                                   checkpoints=str(root / 'checkpoints'), data='SP500',
                                   financial_walkforward=True, financial_split=split)
            self.assertEqual(best_epoch_for_checkpoint(checkpoint), 7)
            self.assertEqual(resolve_financial_checkpoint(args, 'setting', root), checkpoint)
            args.financial_split = {**split, 'valid_end': 882}
            with self.assertRaisesRegex(ValueError, 'financial_split'):
                resolve_financial_checkpoint(args, 'setting', root)


if __name__ == '__main__':
    unittest.main()
