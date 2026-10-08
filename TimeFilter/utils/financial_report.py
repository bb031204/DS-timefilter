"""Persist the three financial datasets' reference evaluation reports."""

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from utils.stockmixer_metrics import compute_metrics


def financial_metrics(preds, trues, masks=None):
    """Convert TimeFilter [days, 1, stocks] output to the reference layout."""
    if preds.ndim != 3 or preds.shape != trues.shape or preds.shape[1] != 1:
        raise ValueError('Financial evaluation requires matching [days, 1, stocks] arrays')
    if not np.isfinite(preds).all() or not np.isfinite(trues).all():
        raise ValueError('Financial evaluation received non-finite returns')
    predictions = np.ascontiguousarray(preds[:, 0, :].T)
    targets = np.ascontiguousarray(trues[:, 0, :].T)
    if masks is not None and (masks.shape != preds.shape or not np.isin(masks, (0, 1)).all()):
        raise ValueError('Financial masks must be binary and match prediction shape')
    mask = np.ones_like(targets) if masks is None else np.ascontiguousarray(masks[:, 0, :].T)
    return compute_metrics(predictions, targets, mask)


def metric_line(name, metrics):
    return (f"{name} Financial | IC: {metrics['IC']:.6f} "
            f"RIC(Pearson ICIR): {metrics['RIC']:.6f} RankIC: {metrics['RankIC']:.6f} "
            f"Precision@10: {metrics['prec_10']:.6f} SR(top5): {metrics['sharpe5']:.6f}")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class FinancialReport:
    def __init__(self, setting, args, mode):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path = Path('results') / setting / 'financial' / timestamp
        if getattr(args, 'financial_output_dir', None):
            self.path = Path(args.financial_output_dir) / 'results' / setting / 'financial'
        self.path.mkdir(parents=True, exist_ok=False)
        config = {
            'mode': mode,
            'args': vars(args),
            'metric_reference': 'D:/finance/baseline/StockMixer-master/src/evaluator.py (SP500 all-valid mask); TimeFilter/utils/stockmixer_metrics.py',
            'RIC_definition': 'mean(daily Pearson IC) / std(daily Pearson IC), ddof=0',
            'RankIC_definition': 'mean(valid daily Spearman correlation), average ranks for ties',
            'precision_n': 10,
            'precision_positive_includes_zero': True,
            'sharpe_top_k': 5,
            'sharpe_annualization': 15.87,
            'sharpe_costs_and_risk_free_rate': 0,
            'array_layout': '[stocks, prediction_days]',
            'data': {'market': args.data,
                     'input': 'StockMixer five EOD features' if getattr(args, 'financial_input_features', 'returns') == 'eod5' else 'daily returns',
                     'mask': 'all ones for SP500; valid history and target for NASDAQ/NYSE'},
        }
        self.write_json('config.json', config)
        print(f'Financial reports: {self.path.resolve()}')

    def write_json(self, filename, data):
        (self.path / filename).write_text(
            json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )

    def epoch(self, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics, components=None):
        row = {'epoch': epoch, 'train_loss': float(train_loss),
               'val_loss': float(val_loss), 'test_loss': float(test_loss)}
        row.update({'val_' + key: value for key, value in val_metrics.items()})
        row.update({'test_' + key: value for key, value in test_metrics.items()})
        if components:
            row.update(components)
        filename = self.path / 'epoch_metrics.csv'
        new_file = not filename.exists()
        with filename.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def final(self, metrics, preds, trues, dataset, masks=None):
        self.write_json('test_metrics.json', metrics)
        self.write_json('data_split.json', {
            'source_start_day': dataset.START_DAY,
            'train_end': getattr(dataset, 'TRAIN_END', None),
            'valid_end': getattr(dataset, 'VALID_END', None),
            'future_end': getattr(dataset, 'FUTURE_END', None),
            'test_target_start': dataset.target_start,
            'test_target_end_exclusive': dataset.target_end,
            'test_days': len(dataset),
        })
        predictions = np.ascontiguousarray(preds[:, 0, :].T)
        targets = np.ascontiguousarray(trues[:, 0, :].T)
        if targets.shape[1] != len(dataset):
            raise ValueError('Final financial report requires the complete ordered test set')
        saved = {
            'prediction': predictions,
            'ground_truth': targets,
            'mask': np.ones_like(targets) if masks is None else np.ascontiguousarray(masks[:, 0, :].T),
            'target_index': np.arange(dataset.target_start, dataset.target_end),
            'source_day_index': np.arange(dataset.target_start, dataset.target_end) + dataset.START_DAY,
        }
        if hasattr(dataset, 'seq_len') and hasattr(dataset, 'pred_len'):
            saved['lookback_length'] = np.asarray(dataset.seq_len)
            saved['horizon'] = np.asarray(dataset.pred_len)
        np.savez_compressed(self.path / 'test_predictions.npz', **saved)
        print(f'Financial results saved: {self.path.resolve()}')
