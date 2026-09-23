"""Numerical parity with Signed_StockMixer and saved-report checks."""

import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from utils.financial_report import FinancialReport, financial_metrics
from utils.stockmixer_metrics import compute_metrics


REFERENCE_ROOT = Path(r'D:\finance\model\Signed_StockMixer\src')


class FinancialMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.pred = rng.normal(0, 0.02, (20, 11)).astype(np.float32)
        self.true = (0.2 * self.pred + rng.normal(0, 0.03, (20, 11))).astype(np.float32)
        self.pred[:3, :2] = 0.01  # ties exercise average-rank semantics
        self.true[0, 0] = 0.0  # zero counts as nonnegative for Precision@10
        self.mask = np.ones_like(self.true)

    @unittest.skipUnless((REFERENCE_ROOT / 'train_signed_stockmixer.py').exists(),
                         'Local Signed_StockMixer reference is unavailable')
    def test_all_metrics_match_local_signed_reference(self):
        # Extract ONLY the four pure reference functions; do not import its
        # training program or execute any training/configuration side effects.
        source = (REFERENCE_ROOT / 'train_signed_stockmixer.py').read_text(encoding='utf-8-sig')
        names = {'rankdata_average', 'safe_corr', 'original_stockmixer_metrics', 'compute_metrics'}
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(nodes), 4)
        namespace = {'np': np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<reference metrics>', 'exec'), namespace)
        for masked in (False, True):
            mask = self.mask.copy()
            if masked:
                mask[:4, :3] = 0
            expected = namespace['compute_metrics'](self.pred, self.true, mask)
            actual = compute_metrics(self.pred, self.true, mask)
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12, equal_nan=True)

    @unittest.skipUnless((REFERENCE_ROOT / 'evaluator.py').exists(),
                         'Local StockMixer evaluator is unavailable')
    def test_original_evaluator_parity_for_complete_sp500_mask(self):
        spec = importlib.util.spec_from_file_location('reference_evaluator', REFERENCE_ROOT / 'evaluator.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        expected = reference.evaluate(self.pred, self.true, self.mask)
        actual = compute_metrics(self.pred, self.true, self.mask)
        for key in ('mse', 'IC', 'RIC', 'prec_10', 'sharpe5'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-7, atol=1e-10)

    def test_daily_layout_and_manual_portfolio_metrics(self):
        preds = self.pred.T[:, None, :]
        trues = self.true.T[:, None, :]
        actual = financial_metrics(preds, trues)
        expected_ic = np.asarray([np.corrcoef(self.pred[:, i], self.true[:, i])[0, 1]
                                  for i in range(self.pred.shape[1])])
        top10 = np.argsort(self.pred, axis=0)[-10:]
        top5 = top10[-5:]
        columns = np.arange(self.pred.shape[1])
        daily_return = self.true[top5, columns].sum(axis=0) / 5
        self.assertAlmostEqual(actual['IC'], expected_ic.mean(), places=10)
        self.assertAlmostEqual(actual['RIC'], expected_ic.mean() / expected_ic.std(), places=10)
        self.assertAlmostEqual(actual['prec_10'], (self.true[top10, columns] >= 0).mean(), places=10)
        np.testing.assert_allclose(actual['sharpe5'], daily_return.mean() / daily_return.std() * 15.87, rtol=1e-6)
        reordered = financial_metrics(preds[::-1], trues[::-1])
        for key in actual:
            np.testing.assert_allclose(actual[key], reordered[key], rtol=1e-6, atol=1e-9)

    def test_constant_predictions_do_not_invent_correlations(self):
        pred = np.ones_like(self.pred)
        with np.errstate(invalid='ignore'):
            actual = compute_metrics(pred, self.true, self.mask)
        self.assertTrue(np.isnan(actual['IC']))
        self.assertTrue(np.isnan(actual['RankIC']))
        self.assertTrue(np.isnan(actual['RIC']))

    def test_persistence_and_recompute(self):
        class TestDataset:
            target_start = 1259
            target_end = 1270
            START_DAY = 915

            def __len__(self):
                return self.target_end - self.target_start

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix='timefilter_financial_test_') as temporary:
            try:
                os.chdir(temporary)
                metrics = financial_metrics(self.pred.T[:, None, :], self.true.T[:, None, :])
                report = FinancialReport('unit_test', SimpleNamespace(data='SP500'), 'training')
                report.epoch(1, 0.5, 0.1, 0.2, metrics, metrics)
                report.epoch(2, 0.4, 0.1, 0.2, metrics, metrics)
                report.final(metrics, self.pred.T[:, None, :], self.true.T[:, None, :], TestDataset())
                with (report.path / 'epoch_metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 2)
                self.assertIn('val_RankIC', rows[0])
                self.assertIn('test_sharpe5', rows[0])
                config = json.loads((report.path / 'config.json').read_text())
                self.assertEqual(config['sharpe_annualization'], 15.87)
                with np.load(report.path / 'test_predictions.npz') as arrays:
                    repeated = compute_metrics(arrays['prediction'], arrays['ground_truth'], arrays['mask'])
                    np.testing.assert_array_equal(arrays['source_day_index'], np.arange(2174, 2185))
                self.assertEqual(repeated, metrics)
                report.write_json('undefined.json', {'RIC': np.nan})
                self.assertEqual(json.loads((report.path / 'undefined.json').read_text()), {'RIC': None})
                second = FinancialReport('unit_test', SimpleNamespace(data='SP500'), 'evaluation_only')
                self.assertNotEqual(second.path, report.path)
            finally:
                os.chdir(old_cwd)


if __name__ == '__main__':
    unittest.main()
