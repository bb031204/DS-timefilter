"""Opt-in, validation-only checkpoint candidates for historical SP500 audits.

The ordinary TimeFilter training and its best.pth remain unchanged. Historical
future data is read only after the validation-selected checkpoints are frozen.
"""

import csv
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from utils.financial_report import financial_metrics, metric_line


RULES = {
    'A': ('stockmixer_val_loss', False),
    'B': ('IC', True),
    'C': ('RankIC', True),
    'D': ('stable_rankic_score', True),
}


class ValidationRuleCheckpoints:
    """Select B/C/D on validation metrics; A is the existing best.pth rule."""

    def __init__(self, checkpoint_dir, report, stability_lambda):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.report = report
        self.stability_lambda = float(stability_lambda)
        if not math.isfinite(self.stability_lambda) or self.stability_lambda < 0:
            raise ValueError('stability_lambda must be finite and non-negative')
        self.best = {}

    def add_stability_score(self, validation_metrics):
        metrics = validation_metrics.copy()
        early = float(metrics['RankIC_early'])
        late = float(metrics['RankIC_late'])
        metrics['stable_rankic_score'] = (early + late) / 2 - self.stability_lambda * abs(early - late) / 2
        return metrics

    def update(self, model, epoch, validation_metrics):
        if any(key.startswith(('test_', 'future_')) for key in validation_metrics):
            raise ValueError('Checkpoint selection accepts validation metrics only')
        for rule in ('B', 'C', 'D'):
            metric, _ = RULES[rule]
            value = float(validation_metrics[metric])
            old = self.best.get(rule)
            if math.isfinite(value) and (old is None or value > old['validation_score']):
                self.best[rule] = {
                    'rule': rule, 'metric': metric, 'selected_epoch': epoch,
                    'validation_score': value,
                    'validation_metrics_at_selected_epoch': validation_metrics.copy(),
                    'checkpoint': f'best_rule_{rule}.pth',
                }
                torch.save(model.state_dict(), self.checkpoint_dir / f'best_rule_{rule}.pth')

    def finish(self, official_best):
        if set(self.best) != {'B', 'C', 'D'}:
            raise ValueError('All three alternative validation rules require a finite score')
        shutil.copy2(self.checkpoint_dir / 'best.pth', self.checkpoint_dir / 'best_rule_A.pth')
        a = {
            'rule': 'A', 'metric': 'stockmixer_val_loss',
            'selected_epoch': official_best['epoch'],
            'validation_score': official_best['value'],
            'validation_metrics_at_selected_epoch': official_best['validation_metrics'],
            'checkpoint': 'best_rule_A.pth',
        }
        self.report.write_json('selection_summary.json', {
            'selection_data': 'validation only',
            'stability_lambda': self.stability_lambda,
            'rules': {'A': a, **self.best},
        })


def top5_return_metrics(prediction, ground_truth):
    """SP500 all-valid, gross equal-weight returns; no costs or risk-free rate."""
    if prediction.shape != ground_truth.shape or prediction.ndim != 2 or prediction.shape[0] < 5:
        raise ValueError('Top5 return needs matching [stocks, days] arrays')
    indices = np.argsort(prediction, axis=0)[-5:, :]
    daily_top5 = np.take_along_axis(ground_truth, indices, axis=0).mean(axis=0)
    daily_universe = ground_truth.mean(axis=0)
    return {
        'mean_top5_return': float(daily_top5.mean()),
        'mean_universe_return': float(daily_universe.mean()),
        'mean_top5_excess_return': float((daily_top5 - daily_universe).mean()),
    }, daily_top5, daily_universe


def _future_predictions(exp):
    dataset, loader = exp._get_data(flag='test')
    preds, trues, masks = [], [], []
    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, batch_y_mark in loader:
            outputs, _ = exp.model(batch_x.float().to(exp.device), exp.masks, is_training=False)
            pred = outputs[:, -exp.args.pred_len:, :].detach().cpu()
            true = batch_y[:, -exp.args.pred_len:, :].float()
            preds.append(pred.numpy())
            trues.append(true.numpy())
            masks.append(exp._financial_mask(batch_y_mark, true).numpy())
    prediction = np.concatenate(preds, axis=0)
    ground_truth = np.concatenate(trues, axis=0)
    mask = np.concatenate(masks, axis=0)
    return dataset, prediction, ground_truth, mask


def _epoch_diagnostics(folder):
    source = folder / 'epoch_metrics.csv'
    with source.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    selected = ['epoch', 'val_mse', 'val_stockmixer_val_loss', 'val_IC', 'val_RIC',
                'val_RankIC', 'val_RankIC_std', 'val_RankICIR', 'val_prec_10',
                'val_sharpe5', 'val_directional_accuracy', 'val_IC_early',
                'val_IC_late', 'val_RankIC_early', 'val_RankIC_late',
                'val_stable_rankic_score']
    with (folder / 'selection_epoch_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=selected)
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in selected} for row in rows)
    pairs = {
        'IC_to_IC': ('val_IC', 'test_IC'),
        'RankIC_to_RankIC': ('val_RankIC', 'test_RankIC'),
        'StockMixerLoss_to_IC': ('val_stockmixer_val_loss', 'test_IC'),
        'StockMixerLoss_to_RankIC': ('val_stockmixer_val_loss', 'test_RankIC'),
        'Precision10_to_Precision10': ('val_prec_10', 'test_prec_10'),
        'Sharpe5_to_Sharpe5': ('val_sharpe5', 'test_sharpe5'),
        'IC_to_RankIC': ('val_IC', 'test_RankIC'),
        'RankIC_to_IC': ('val_RankIC', 'test_IC'),
    }
    correlations = {'epochs': len(rows), 'interpretation': 'descriptive only; epochs are dependent'}
    if rows and rows[0].get('test_IC'):
        future_keys = ['epoch'] + [key for key in rows[0] if key.startswith('test_')]
        with (folder / 'future_epoch_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=future_keys)
            writer.writeheader()
            writer.writerows({key: row.get(key, '') for key in future_keys} for row in rows)
    for label, (left, right) in pairs.items():
        values = np.asarray([[float(row.get(left) or 'nan'), float(row.get(right) or 'nan')]
                             for row in rows], dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        correlations[label] = (float(np.corrcoef(values.T)[0, 1])
                               if len(values) > 1 and np.std(values, axis=0).min() > 0 else None)
    return correlations


def evaluate_selected_future(exp, setting):
    """Evaluate frozen A/B/C/D checkpoints on the historical future block."""
    import json

    folder = exp._financial_report.path
    selection = json.loads((folder / 'selection_summary.json').read_text(encoding='utf-8'))
    with np.load(folder / 'test_predictions.npz') as reference:
        expected_prediction = reference['prediction'].copy()
        expected_target = reference['ground_truth'].copy()
        expected_index = reference['target_index'].copy()
    for rule in RULES:
        if rule == 'A':
            prediction = expected_prediction
            ground_truth = expected_target
            metrics = json.loads((folder / 'test_metrics.json').read_text(encoding='utf-8'))
            shutil.copy2(folder / 'test_predictions.npz', folder / 'future_predictions_rule_A.npz')
        else:
            checkpoint = Path(exp.args.checkpoints) / setting / selection['rules'][rule]['checkpoint']
            exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device, weights_only=True))
            dataset, pred, true, mask = _future_predictions(exp)
            if not np.array_equal(true[:, 0, :].T, expected_target):
                raise ValueError('Historical future targets changed between selection rules')
            prediction = np.ascontiguousarray(pred[:, 0, :].T)
            ground_truth = np.ascontiguousarray(true[:, 0, :].T)
            metrics = financial_metrics(pred, true, mask)
            np.savez_compressed(folder / f'future_predictions_rule_{rule}.npz',
                                prediction=prediction, ground_truth=ground_truth,
                                mask=np.ascontiguousarray(mask[:, 0, :].T),
                                target_index=np.arange(dataset.target_start, dataset.target_end),
                                source_day_index=np.arange(dataset.target_start, dataset.target_end) + dataset.START_DAY,
                                lookback_length=np.asarray(dataset.seq_len), horizon=np.asarray(dataset.pred_len))
            if not np.array_equal(np.arange(dataset.target_start, dataset.target_end), expected_index):
                raise ValueError('Historical future dates changed between selection rules')
        returns, daily_top5, daily_universe = top5_return_metrics(prediction, ground_truth)
        metrics.update(returns)
        exp._financial_report.write_json(f'future_metrics_rule_{rule}.json', metrics)
        np.savez_compressed(folder / f'future_daily_rule_{rule}.npz',
                            target_index=expected_index,
                            daily_top5_realized_return=daily_top5,
                            daily_universe_return=daily_universe)
        print(metric_line(f'Future {rule}', metrics), flush=True)
    exp._financial_report.write_json('epoch_correlations.json', _epoch_diagnostics(folder))
