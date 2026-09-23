"""NASDAQ/NYSE return windows; preserve the four-item forecasting interface."""
import pickle
from pathlib import Path

import numpy as np
from data_provider.stockmixer_sp500 import Dataset_SP500
from data_provider.financial_registry import validate_files


class Dataset_Exchange(Dataset_SP500):
    START_DAY = 0
    TRAIN_END = 756
    VALID_END = 1008

    def __init__(self, args, root_path, flag='train', size=None, features='M', **kwargs):
        validate_files(root_path, args.data)
        self.seq_len, self.label_len, self.pred_len = size or (16, 0, 1)
        if flag not in ('train', 'val', 'test'):
            raise ValueError('flag must be train, val or test')
        if not 0 < self.seq_len < self.TRAIN_END - 1 or not 0 <= self.label_len <= self.seq_len:
            raise ValueError('Invalid financial history/label length')
        if self.pred_len != 1 or features != 'M':
            raise ValueError('Financial datasets require pred_len=1 and features=M')
        if args.patch_len < 1 or self.seq_len % args.patch_len:
            raise ValueError('seq_len must be divisible by patch_len')
        if getattr(args, 'augmentation_ratio', 0):
            raise ValueError('Financial data augmentation is not supported')
        arrays = []
        for name in ('price_data.pkl', 'gt_data.pkl', 'mask_data.pkl'):
            # These are the trusted local StockMixer dataset artifacts.
            with (Path(root_path) / name).open('rb') as stream:
                arrays.append(np.asarray(pickle.load(stream)))
        price, returns, observed = arrays
        if price.ndim != 2 or not price.shape == returns.shape == observed.shape:
            raise ValueError('Expected matching [stocks, days] price/return/mask arrays')
        if not np.isin(observed, (0, 1)).all():
            raise ValueError('StockMixer mask must contain only 0 and 1')
        self.n_stocks, self.TOTAL_DAYS = price.shape
        if self.TOTAL_DAYS <= self.VALID_END:
            raise ValueError('Dataset does not contain the complete train/validation split plus test days')
        for name in ('enc_in', 'dec_in', 'c_out'):
            if getattr(args, name) != self.n_stocks:
                raise ValueError(f'{args.data} requires {name}={self.n_stocks}')
        valid = observed.astype(bool)
        valid[:, 1:] &= observed[:, :-1].astype(bool)
        valid[:, 0] = False  # No preceding close exists for day zero.
        if not np.isfinite(returns[valid]).all():
            raise ValueError('Valid returns must be finite')
        pairs = valid[:, 1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            derived = price[:, 1:] / price[:, :-1] - 1
        if not np.allclose(derived[pairs], returns[:, 1:][pairs], rtol=1e-4, atol=2e-7):
            raise ValueError('StockMixer returns do not match one-day close returns')
        values = np.where(valid, returns, 0).T.astype(np.float32)
        self.target_start, self.target_end = {
            'train': (self.seq_len + 1, self.TRAIN_END),
            'val': (self.TRAIN_END, self.VALID_END),
            'test': (self.VALID_END, self.TOTAL_DAYS),
        }[flag]
        self.data_x = values[self.target_start - self.seq_len:self.target_end]
        self.data_y = self.data_x
        self.scale = False
        self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)
        # A stock is eligible only if its entire history and target are observed.
        missing = np.concatenate([np.zeros((1, self.n_stocks), dtype=np.int32),
                                  np.cumsum(~valid.T, axis=0)], axis=0)
        days = np.arange(self.target_start, self.target_end)
        self.sample_masks = (missing[days + 1] == missing[days - self.seq_len]).astype(np.float32)

    def __getitem__(self, index):
        x, y, x_mark, _ = super().__getitem__(index)
        # TimeFilter does not consume y_mark; keep the existing four-item API.
        y_mask = np.repeat(self.sample_masks[index:index + 1], self.label_len + 1, axis=0)
        return x, y, x_mark, y_mask
