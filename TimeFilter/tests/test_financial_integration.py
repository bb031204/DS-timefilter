"""Financial routing, exchange masks, configuration and run isolation."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from data_provider.financial_registry import is_financial_dataset, validate_files
from data_provider.stockmixer_exchange import Dataset_Exchange
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from scripts.run_financial import build_command, PROJECT_ROOT
from utils.financial_losses import stockmixer_validation_loss
from utils.financial_report import financial_metrics
from utils.financial_runtime import allocate_run
from utils.stockmixer_metrics import compute_metrics


class FinancialIntegrationTests(unittest.TestCase):
    def test_only_financial_markets(self):
        for name in ('SP500', 'S&P500', 'NASDAQ', 'NYSE'):
            self.assertTrue(is_financial_dataset(name))
        for name in ('ETTh1', 'custom', 'Solar', 'PEMS', 'Climate'):
            self.assertFalse(is_financial_dataset(name))

    def test_config_market_channels_and_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            exchange_config = yaml.safe_load((PROJECT_ROOT / 'config.yaml').read_text(encoding='utf-8'))
            exchange_config['forecast']['input_features'] = 'returns'
            exchange_config['training']['financial_selection'] = 'RankIC'
            exchange_path = Path(root) / 'exchange.yaml'
            exchange_path.write_text(yaml.safe_dump(exchange_config), encoding='utf-8')
            for market, channels in [('SP500', 474), ('NASDAQ', 1026), ('NYSE', 1737)]:
                config_path = PROJECT_ROOT / 'config.yaml' if market == 'SP500' else exchange_path
                cli = argparse.Namespace(config=str(config_path), dataset=market,
                    mode=None, checkpoint=None, batch_size=4, train_epochs=1, learning_rate=None)
                command = build_command(cli)
                self.assertEqual(command[command.index('--c_out') + 1], str(channels))
                self.assertEqual(command[command.index('--batch_size') + 1], '4')
                self.assertEqual(command[command.index('--d_model') + 1], '512')
                self.assertEqual(command[command.index('--financial_optimizer') + 1], 'adamw')
                self.assertEqual(command[command.index('--financial_weight_decay') + 1], '0.0001')
                self.assertEqual(command[command.index('--financial_grad_clip_norm') + 1], '1.0')
                self.assertEqual(command[command.index('--financial_test_each_epoch') + 1], '1')
                self.assertTrue(Path(command[command.index('--root_path') + 1]).is_absolute())

    def test_five_feature_config_cannot_be_used_for_another_market(self):
        cli = argparse.Namespace(config=str(PROJECT_ROOT / 'config.yaml'), dataset='NASDAQ',
                                 mode=None, checkpoint=None, batch_size=None, train_epochs=None,
                                 learning_rate=None)
        with self.assertRaisesRegex(ValueError, 'only for SP500'):
            build_command(cli)

    def test_test_each_epoch_config_switch_accepts_only_yaml_boolean(self):
        with tempfile.TemporaryDirectory() as root:
            config = yaml.safe_load((PROJECT_ROOT / 'config.yaml').read_text(encoding='utf-8'))
            path = Path(root) / 'switch.yaml'
            cli = argparse.Namespace(config=str(path), dataset='SP500', mode=None,
                                     checkpoint=None, batch_size=None, train_epochs=None,
                                     learning_rate=None, moe_aux_weight=None)
            config['training']['test_each_epoch'] = False
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            command = build_command(cli)
            self.assertEqual(command[command.index('--financial_test_each_epoch') + 1], '0')
            config['training']['test_each_epoch'] = 'false'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must be YAML true or false'):
                build_command(cli)

    def test_same_minute_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            now = datetime(2026, 9, 10, 15, 40)
            first = allocate_run('SP500', root, now)
            second = allocate_run('SP500', root, now)
            self.assertEqual(first.relative_to(root).as_posix(), 'SP500_2026_09_10_15_40')
            self.assertEqual(second.name, 'SP500_2026_09_10_15_40_02')
            self.assertEqual(second.parent, first.parent)

    def test_empty_data_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'price_data.pkl').touch()
            with self.assertRaisesRegex(ValueError, 'Missing or empty'):
                validate_files(root, 'NYSE')

    def test_exchange_split_history_mask_and_metric_parity(self):
        rng = np.random.default_rng(9)
        prices = np.exp(np.cumsum(rng.normal(0, .01, (20, 1015)), axis=1)).astype(np.float32)
        returns = np.zeros_like(prices)
        returns[:, 1:] = prices[:, 1:] / prices[:, :-1] - 1
        mask = np.ones_like(prices)
        mask[0, 1000] = 0
        args = SimpleNamespace(data='NYSE', patch_len=16, enc_in=20, dec_in=20, c_out=20)
        with tempfile.TemporaryDirectory() as root:
            for name, value in [('price', prices), ('gt', returns), ('mask', mask)]:
                with (Path(root) / f'{name}_data.pkl').open('wb') as stream:
                    pickle.dump(value, stream)
            train = Dataset_Exchange(args, root, size=(16, 0, 1))
            val = Dataset_Exchange(args, root, flag='val', size=(16, 0, 1))
            test = Dataset_Exchange(args, root, flag='test', size=(16, 0, 1))
            self.assertEqual((len(train), len(val), len(test)), (739, 252, 7))
            np.testing.assert_array_equal(train[0][0], returns[:, 1:17].T)
            np.testing.assert_array_equal(train[-1 + len(train)][1][0], returns[:, 755])
            self.assertEqual(test[0][3][0, 0], 0)
            np.testing.assert_array_equal(test[0][3][0, 1:], 1)
            targets = np.stack([test[i][1] for i in range(len(test))])
            masks = np.stack([test[i][3] for i in range(len(test))])
            predictions = rng.normal(0, .01, targets.shape).astype(np.float32)
            actual = financial_metrics(predictions, targets, masks)
            expected = compute_metrics(predictions[:, 0].T, targets[:, 0].T, masks[:, 0].T)
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], equal_nan=True)

    def test_invalid_stock_excluded_from_loss_without_changing_sp500(self):
        experiment = object.__new__(Exp_Long_Term_Forecast)
        experiment.args = SimpleNamespace(data='NASDAQ', pred_len=1)
        pred = torch.tensor([[[2., 100.]]], requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.tensor([[[1., 0.]]])
        loss = experiment._prediction_loss(pred, target, mask, torch.nn.MSELoss())
        self.assertEqual(loss.item(), 4)
        loss.backward()
        self.assertEqual(pred.grad[0, 0, 1].item(), 0)
        experiment.args.data = 'SP500'
        self.assertEqual(experiment._prediction_loss(pred, target, mask, torch.nn.MSELoss()).item(), 5002)

    def test_stockmixer_selection_uses_all_validation_days_equally(self):
        class LastInput(torch.nn.Module):
            def forward(self, x, masks, is_training=False):
                return x[:, -1:, :], 0.0

        rng = np.random.default_rng(12)
        prediction = torch.tensor(rng.normal(0, 0.01, (3, 1, 12)), dtype=torch.float32)
        target = torch.tensor(rng.normal(0, 0.01, (3, 1, 12)), dtype=torch.float32)
        marks = torch.zeros((3, 1, 1))
        batches = [(prediction[:2], target[:2], marks[:2], marks[:2]),
                   (prediction[2:], target[2:], marks[2:], marks[2:])]
        experiment = object.__new__(Exp_Long_Term_Forecast)
        experiment.args = SimpleNamespace(data='SP500', pred_len=1, features='M',
                                          stockmixer_selection_rank_weight=0.1)
        experiment.model = LastInput()
        experiment.device = torch.device('cpu')
        experiment.masks = None
        experiment.vali(None, batches, torch.nn.MSELoss(), stockmixer_selection=True)
        expected = np.mean([
            stockmixer_validation_loss(prediction[i:i + 1], target[i:i + 1],
                                       torch.ones_like(target[i:i + 1]), 0.1).item()
            for i in range(3)
        ])
        self.assertAlmostEqual(experiment._last_financial_metrics['stockmixer_val_loss'], expected)

    def test_financial_optimizer_excludes_norm_and_bias_from_weight_decay(self):
        experiment = object.__new__(Exp_Long_Term_Forecast)
        experiment.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3))
        experiment.args = SimpleNamespace(data='SP500', learning_rate=1e-4,
                                          financial_optimizer='adamw', financial_weight_decay=1e-4)
        optimizer = experiment._select_optimizer()
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual([group['weight_decay'] for group in optimizer.param_groups], [1e-4, 0.0])
        self.assertEqual(len(optimizer.param_groups[0]['params']), 1)
        self.assertEqual(len(optimizer.param_groups[1]['params']), 3)
        experiment.args.data = 'ETTh1'
        self.assertIsInstance(experiment._select_optimizer(), torch.optim.Adam)

    def test_financial_test_epoch_switch_preserves_validation_selection(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x, masks, is_training=False):
                return x[:, -1:, :] * self.scale, x.new_zeros(())

        rng = np.random.default_rng(19)
        x = torch.tensor(rng.normal(0, 0.01, (4, 16, 12)), dtype=torch.float32)
        y = torch.tensor(rng.normal(0, 0.01, (4, 1, 12)), dtype=torch.float32)
        marks = torch.zeros((4, 1, 1))
        batch = (x, y, marks, marks)
        test_batch = (x, y + 0.5, marks, marks)

        for validation_only, test_each_epoch in ((False, True), (False, False), (True, True)):
            with self.subTest(validation_only=validation_only, test_each_epoch=test_each_epoch), \
                    tempfile.TemporaryDirectory() as root:
                requested = []

                def get_data(flag):
                    requested.append(flag)
                    return range(4), [test_batch if flag == 'test' else batch]

                experiment = object.__new__(Exp_Long_Term_Forecast)
                experiment.args = SimpleNamespace(
                    data='SP500', features='M', pred_len=1, checkpoints=str(Path(root) / 'checkpoints'),
                    financial_output_dir=str(Path(root) / 'output'), learning_rate=1e-4,
                    financial_optimizer='adamw', financial_weight_decay=1e-4,
                    financial_grad_clip_norm=1.0, train_epochs=1, patience=3,
                    moe_aux_weight=0.0, rank_weight=0.0, finance_adaptation={'enabled': False},
                    financial_selection='mse', financial_validation_only=validation_only,
                    financial_test_each_epoch=int(test_each_epoch),
                    gradient_diagnostic_epochs=[], use_amp=False, lradj='cosine', c_out=12)
                experiment.device = torch.device('cpu')
                experiment.model = TinyModel()
                experiment.masks = None
                experiment._get_data = get_data
                experiment.train('tiny')
                evaluate_each_epoch = test_each_epoch and not validation_only
                self.assertEqual(requested, ['train', 'val'] + (['test'] if evaluate_each_epoch else []))
                self.assertTrue((Path(root) / 'checkpoints' / 'tiny' / 'best.pth').is_file())
                report_dir = Path(root) / 'output' / 'results' / 'tiny' / 'financial'
                with (report_dir / 'epoch_metrics.csv').open(newline='', encoding='utf-8') as stream:
                    epoch = next(csv.DictReader(stream))
                selected = json.loads((report_dir / 'selection.json').read_text(encoding='utf-8'))
                self.assertIn('grad_clip_fraction', epoch)
                self.assertEqual('test_IC' in epoch, evaluate_each_epoch)
                if evaluate_each_epoch:
                    self.assertGreater(float(epoch['test_mse']), float(epoch['val_mse']))
                self.assertAlmostEqual(selected['best']['value'], float(epoch['val_mse']))


if __name__ == '__main__':
    unittest.main()
