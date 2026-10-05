"""Validation-only checkpoint selection for financial experiments."""
import json
import math
from pathlib import Path
import torch


def best_epoch_for_checkpoint(checkpoint):
    """Read the recorded best epoch for a run's best.pth, if available."""
    checkpoint = Path(checkpoint).resolve()
    candidate = {'best.pth': None, 'best_stockmixer_loss.pth': 'A',
                 'best_rankic.pth': 'B'}.get(checkpoint.name, 'unknown')
    if candidate == 'unknown':
        return None
    selection_path = (checkpoint.parents[2] / 'results' / checkpoint.parent.name / 'financial'
                      / ('selection.json' if candidate is None else 'selection_summary.json'))
    if not selection_path.is_file():
        return None
    try:
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if candidate is None:
        if selection.get('final_evaluation_checkpoint') != checkpoint.name:
            return None
        epoch = (selection.get('best') or {}).get('epoch')
    else:
        if selection.get(f'{candidate}_checkpoint') != checkpoint.name:
            return None
        epoch = (selection.get(candidate) or {}).get('epoch')
    return epoch if type(epoch) is int and epoch > 0 else None


class FinancialSelection:
    def __init__(self, path, report, criterion='mse'):
        self.path = Path(path)
        self.report = report
        self.criterion = criterion
        self.best = None

    def update(self, model, epoch, metrics):
        value = float(metrics[self.criterion])
        torch.save(model.state_dict(), self.path / 'last.pth')
        minimize = self.criterion in ('mse', 'stockmixer_val_loss')
        better = math.isfinite(value) and (self.best is None or (
            value < self.best['value'] if minimize else value > self.best['value']))
        if better:
            self.best = {'epoch': epoch, 'metric': self.criterion, 'value': value,
                         'validation_metrics': metrics}
            torch.save(model.state_dict(), self.path / 'best.pth')
            print(f'\033[32mNew Best | epoch {epoch} | validation {self.criterion}={value:.8f} '
                  f'| saved best.pth\033[0m', flush=True)
        self.report.write_json('selection.json', {'best': self.best, 'last_epoch': epoch,
                               'final_evaluation_checkpoint': 'best.pth'})

    def finish(self):
        if self.best is None:
            raise ValueError('No finite validation selection metric; no best checkpoint exists')
        return str(self.path / 'best.pth')
