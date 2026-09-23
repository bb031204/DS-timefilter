"""StockMixer S&P500 daily returns, using the existing forecasting interface."""

import os

import numpy as np
from torch.utils.data import Dataset


class Dataset_SP500(Dataset):
    # StockMixer crops SP500.npy at day 915. Its paper's SP500 split is
    # 1006 training days, 253 validation days and 352 test days.
    START_DAY = 915
    TRAIN_END = 1006
    VALID_END = 1259
    TOTAL_DAYS = 1611

    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='SP500.npy', target='OT', scale=False,
                 timeenc=0, freq='d', seasonal_patterns=None):
        if flag not in ('train', 'val', 'test'):
            raise ValueError("SP500 flag must be train, val or test")
        self.seq_len, self.label_len, self.pred_len = (
            [16, 0, 1] if size is None else size
        )
        if self.seq_len < 1 or self.seq_len >= self.TRAIN_END:
            raise ValueError("SP500 seq_len must be between 1 and 1005")
        if not 0 <= self.label_len <= self.seq_len:
            raise ValueError("SP500 requires 0 <= label_len <= seq_len")
        if self.pred_len != 1:
            raise ValueError("SP500 supports next-trading-day prediction: pred_len=1")
        if features != 'M':
            raise ValueError("SP500 requires features=M to forecast all stocks jointly")
        if args.patch_len < 1 or self.seq_len % args.patch_len:
            raise ValueError("TimeFilter requires seq_len divisible by patch_len")
        if getattr(args, 'augmentation_ratio', 0):
            raise ValueError("SP500 does not implement data augmentation")

        path = os.path.join(root_path, data_path)
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        if raw.ndim != 3 or raw.shape[2] != 5:
            raise ValueError("Expected SP500.npy shaped [stocks, days, 5]")
        if raw.shape[1] != self.START_DAY + self.TOTAL_DAYS:
            raise ValueError("Expected StockMixer SP500.npy with 2526 trading days")
        self.n_stocks = raw.shape[0]
        for name in ('enc_in', 'dec_in', 'c_out'):
            if getattr(args, name) != self.n_stocks:
                raise ValueError(f"SP500 requires {name}={self.n_stocks}")

        # The final feature is the author's normalized close price. Its fixed
        # per-stock scale cancels in the return ratio. Keep the previous close
        # to compute the first day's return without inventing a zero return.
        close = raw[:, self.START_DAY - 1:, -1].T
        if not np.isfinite(close).all() or np.any(close <= 0):
            raise ValueError("SP500 close prices must be finite and positive")
        returns = np.asarray(close[1:] / close[:-1] - 1.0, dtype=np.float32)

        # Split by TARGET day. Validation/test inputs may use earlier observed
        # history, but no training label can cross into validation or test.
        target_start, target_end = {
            'train': (self.seq_len, self.TRAIN_END),
            'val': (self.TRAIN_END, self.VALID_END),
            'test': (self.VALID_END, self.TOTAL_DAYS),
        }[flag]
        self.target_start = target_start
        self.target_end = target_end
        self.data_x = returns[target_start - self.seq_len:target_end]
        self.data_y = self.data_x
        # No external scaling: predictions/labels are actual daily returns.
        # TimeFilter's internal normalization remains completely unchanged.
        self.scale = False
        # run.py's forecasting experiment does not consume calendar features.
        self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        s_end = index + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        return (
            self.data_x[index:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[index:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return self.target_end - self.target_start

    def inverse_transform(self, data):
        return np.asarray(data)
