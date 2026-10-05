"""Validation-only candidate checkpoints for historical SP500 selection audits."""

import csv
import math
from pathlib import Path
import shutil

import numpy as np
import torch


class WalkforwardRankCheckpoint:
    """Keep the RankIC candidate without changing the official best.pth rule."""

    def __init__(self, checkpoint_dir, report):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.report = report
        self.best = None

    def update(self, model, epoch, validation_metrics):
        value = float(validation_metrics['RankIC'])
        if math.isfinite(value) and (self.best is None or value > self.best['value']):
            self.best = {'epoch': epoch, 'metric': 'RankIC', 'value': value,
                         'validation_metrics': validation_metrics.copy()}
            torch.save(model.state_dict(), self.checkpoint_dir / 'best_rankic.pth')

    def finish(self, official_best):
        if self.best is None:
            raise ValueError('No finite validation RankIC; cannot finish walk-forward audit')
        shutil.copy2(self.checkpoint_dir / 'best.pth',
                     self.checkpoint_dir / 'best_stockmixer_loss.pth')
        self.report.write_json('selection_summary.json', {
            'A': official_best,
            'B': self.best,
            'A_checkpoint': 'best_stockmixer_loss.pth',
            'B_checkpoint': 'best_rankic.pth',
            'selection_data': 'validation only',
        })


def epoch_correlations(report_path):
    """Descriptive diagnostics only; correlated epochs are not independent trials."""
    pairs = {
        'RankIC': ('val_RankIC', 'test_RankIC'),
        'IC': ('val_IC', 'test_IC'),
        'MSE_to_future_RankIC': ('val_mse', 'test_RankIC'),
        'stockmixer_val_loss_to_future_RankIC':
            ('val_stockmixer_val_loss', 'test_RankIC'),
    }
    with (Path(report_path) / 'epoch_metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    result = {'epochs': len(rows), 'interpretation': 'descriptive; never used for checkpoint selection'}
    for name, (left, right) in pairs.items():
        values = np.asarray([[float(row.get(left) or 'nan'), float(row.get(right) or 'nan')]
                             for row in rows], dtype=float)
        valid = np.isfinite(values).all(axis=1)
        values = values[valid]
        result[name] = (float(np.corrcoef(values.T)[0, 1])
                        if len(values) > 1 and np.std(values, axis=0).min() > 0 else None)
    return result
