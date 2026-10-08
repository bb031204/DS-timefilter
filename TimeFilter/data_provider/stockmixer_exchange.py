"""StockMixer NASDAQ/NYSE windows, split by next-day target index."""
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
        self.input_features = getattr(args, 'financial_input_features', 'returns')
        if self.input_features not in ('returns', 'eod5'):
            raise ValueError('Exchange financial_input_features must be returns or eod5')
        if self.input_features == 'eod5' and getattr(args, 'financial_norm', 0):
            raise ValueError('Exchange eod5 input requires financial_norm=0')
        validate_files(root_path, args.data, self.input_features)
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
        def read_array(name):
            with (Path(root_path) / name).open('rb') as stream:
                return np.asarray(pickle.load(stream))

        price, returns, observed = (read_array(name) for name in
                                    ('price_data.pkl', 'gt_data.pkl', 'mask_data.pkl'))
        if price.ndim != 2 or not price.shape == returns.shape == observed.shape:
            raise ValueError('Expected matching [stocks, days] price/return/mask arrays')
        if not np.isfinite(price).all() or not np.isfinite(returns).all():
            raise ValueError('Exchange prices and returns must be finite')
        if not np.isin(observed, (0, 1)).all():
            raise ValueError('StockMixer mask must contain only 0 and 1')
        self.n_stocks, self.TOTAL_DAYS = price.shape
        self.FUTURE_END = self.TOTAL_DAYS
        if self.TOTAL_DAYS <= self.VALID_END:
            raise ValueError('Dataset does not contain the complete train/validation split plus test days')
        for name in ('enc_in', 'dec_in', 'c_out'):
            if getattr(args, name) != self.n_stocks:
                raise ValueError(f'{args.data} requires {name}={self.n_stocks}')
        if self.input_features == 'eod5':
            eod = read_array('eod_data.pkl')
            if eod.shape != (self.n_stocks, self.TOTAL_DAYS, 5) or not np.isfinite(eod).all():
                raise ValueError('Expected finite StockMixer EOD data [stocks, days, 5]')
            # Keep the released five indicators, including its missing-value
            # fill. The StockMixer window mask excludes incomplete samples.
            inputs = eod.transpose(1, 0, 2)
        else:
            inputs = returns.T
        self.target_start, self.target_end = {
            # Official offset 0 uses EOD days 0..15 to predict target day 16.
            # The released trainer shuffles offsets 0..755 and takes 740 of
            # them, which can put target days 756..771 in training. Keep the
            # official 756/1008 boundaries by target day to avoid leakage.
            'train': (self.seq_len, self.TRAIN_END),
            'val': (self.TRAIN_END, self.VALID_END),
            'test': (self.VALID_END, self.TOTAL_DAYS),
        }[flag]
        begin = self.target_start - self.seq_len
        self.data_x = np.ascontiguousarray(inputs[begin:self.target_end], dtype=np.float32)
        self.data_y = np.ascontiguousarray(returns.T[begin:self.target_end], dtype=np.float32)
        self.scale = False
        self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)
        # Match StockMixer get_batch: min(mask[offset:offset+seq_len+1]).
        # The final element is the target day; all 16 input days are included.
        missing = np.pad(np.cumsum(observed == 0, axis=1, dtype=np.int32),
                         ((0, 0), (1, 0)))
        days = np.arange(self.target_start, self.target_end)
        self.sample_masks = (missing[:, days + 1] == missing[:, days - self.seq_len]).T.astype(np.float32)

    def __getitem__(self, index):
        x, y, x_mark, _ = super().__getitem__(index)
        # TimeFilter does not consume y_mark; keep the existing four-item API.
        y_mask = np.repeat(self.sample_masks[index:index + 1], self.label_len + self.pred_len, axis=0)
        return x, y, x_mark, y_mask
