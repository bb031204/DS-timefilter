"""Focused checks for the local StockMixer S&P500 adapter (no training)."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from data_provider.data_factory import data_provider
from data_provider.stockmixer_sp500 import Dataset_SP500
from scripts.run_sp500 import build_command


DATA_ROOT = Path(__file__).resolve().parents[1] / 'stockmixer_dataset' / 'SP500'


def make_args(**overrides):
    args = dict(
        data='SP500', root_path=str(DATA_ROOT), data_path='SP500.npy',
        seq_len=16, label_len=0, pred_len=1, features='M', target='OT',
        enc_in=474, dec_in=474, c_out=474, patch_len=16,
        augmentation_ratio=0, embed='timeF', freq='d',
        seasonal_patterns='Monthly', batch_size=2, num_workers=0,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def make_dataset(flag='train', **overrides):
    args = make_args(**overrides)
    return Dataset_SP500(
        args, args.root_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
    )


class SP500InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = np.load(DATA_ROOT / 'SP500.npy', mmap_mode='r')
        close = raw[:, 914:, -1].T
        cls.expected_returns = (close[1:] / close[:-1] - 1).astype(np.float32)

    def test_split_boundaries_and_labels(self):
        for flag, start, end in [('train', 16, 1006), ('val', 1006, 1259),
                                 ('test', 1259, 1611)]:
            with self.subTest(flag=flag):
                dataset = make_dataset(flag)
                self.assertEqual(len(dataset), end - start)
                for index in (0, len(dataset) - 1):
                    x, y, x_mark, y_mark = dataset[index]
                    day = start + index
                    np.testing.assert_array_equal(x, self.expected_returns[day - 16:day])
                    np.testing.assert_array_equal(y, self.expected_returns[day:day + 1])
                    self.assertEqual(x.shape, (16, 474))
                    self.assertEqual(y.shape, (1, 474))
                    self.assertEqual(x_mark.shape, (16, 1))
                    self.assertEqual(y_mark.shape, (1, 1))
                    self.assertEqual(x.dtype, np.float32)
                with self.assertRaises(IndexError):
                    dataset[len(dataset)]

    def test_label_context_and_longer_history(self):
        dataset = make_dataset('val', seq_len=32, label_len=8)
        x, y, _, y_mark = dataset[0]
        np.testing.assert_array_equal(x, self.expected_returns[974:1006])
        np.testing.assert_array_equal(y, self.expected_returns[998:1007])
        self.assertEqual(y_mark.shape, (9, 1))
        self.assertEqual(len(dataset), 253)

    def test_no_external_scaling(self):
        dataset = make_dataset()
        self.assertFalse(dataset.scale)
        np.testing.assert_array_equal(dataset.inverse_transform(dataset.data_x), dataset.data_x)
        # Retain a real first-day return, instead of a synthetic all-zero row.
        self.assertTrue(np.any(dataset.data_x[0] != 0))

    def test_existing_factory_and_batch_interface(self):
        dataset, loader = data_provider(make_args(), 'test')
        x, y, x_mark, y_mark = next(iter(loader))
        self.assertEqual(tuple(x.shape), (2, 16, 474))
        self.assertEqual(tuple(y.shape), (2, 1, 474))
        self.assertEqual(tuple(x_mark.shape), (2, 16, 1))
        self.assertEqual(tuple(y_mark.shape), (2, 1, 1))
        np.testing.assert_array_equal(y.numpy()[0], dataset[0][1])

    def test_reject_incompatible_arguments(self):
        for overrides in [dict(pred_len=2), dict(features='MS'), dict(c_out=5),
                          dict(seq_len=17), dict(label_len=48), dict(patch_len=0),
                          dict(augmentation_ratio=1)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_dataset(**overrides)

    def test_launcher_preserves_model_defaults_and_user_overrides(self):
        command = build_command(['--train_epochs', '1'])
        for name in ('d_model', 'd_ff', 'e_layers', 'n_heads', 'patch_len',
                     'alpha', 'top_p', 'dropout', 'learning_rate', 'batch_size'):
            self.assertNotIn('--' + name, command)
        self.assertEqual(command[-2:], ['--train_epochs', '1'])


if __name__ == '__main__':
    unittest.main()
