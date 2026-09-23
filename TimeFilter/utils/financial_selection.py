"""Validation-only checkpoint selection for financial experiments."""
import math
from pathlib import Path
import torch


class FinancialSelection:
    def __init__(self, path, report, criterion='mse'):
        self.path = Path(path)
        self.report = report
        self.criterion = criterion
        self.best = None

    def update(self, model, epoch, metrics):
        value = float(metrics[self.criterion])
        torch.save(model.state_dict(), self.path / 'last.pth')
        better = math.isfinite(value) and (self.best is None or (
            value < self.best['value'] if self.criterion == 'mse' else value > self.best['value']))
        if better:
            self.best = {'epoch': epoch, 'metric': self.criterion, 'value': value,
                         'validation_metrics': metrics}
            torch.save(model.state_dict(), self.path / 'best.pth')
            print(f'Validation best: epoch={epoch}, {self.criterion}={value:.8f}; saved best.pth')
        self.report.write_json('selection.json', {'best': self.best, 'last_epoch': epoch,
                               'final_evaluation_checkpoint': 'best.pth'})

    def finish(self):
        if self.best is None:
            raise ValueError('No finite validation selection metric; no best checkpoint exists')
        return str(self.path / 'best.pth')
