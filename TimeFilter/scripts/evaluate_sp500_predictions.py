"""Re-evaluate saved S&P500 predictions from either model on one protocol."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.stockmixer_metrics import compute_metrics  # noqa: E402


START_DAY = 915
TEST_START = 1259
TEST_END = 1611
STOCKS = 474
REFERENCE_KEYS = ("mse", "IC", "RIC", "prec_10", "sharpe5")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path, expected_target):
    with np.load(path, allow_pickle=False) as saved:
        required = {"prediction", "ground_truth", "mask", "target_index", "source_day_index"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        values = {name: saved[name] for name in required}
        for name in ("lookback_length", "horizon"):
            if name in saved.files:
                values[name] = saved[name]
    shape = (STOCKS, len(expected_target))
    for name in ("prediction", "ground_truth", "mask"):
        if values[name].shape != shape or not np.isfinite(values[name]).all():
            raise ValueError(f"{path}: {name} must be finite and shaped {shape}")
    if not np.array_equal(values["target_index"], expected_target):
        raise ValueError(f"{path}: target days differ from paper S&P500 test days")
    if not np.array_equal(values["source_day_index"], expected_target + START_DAY):
        raise ValueError(f"{path}: original SP500.npy day indices do not match")
    if not np.array_equal(values["mask"], np.ones(shape)):
        raise ValueError(f"{path}: expected the all-valid S&P500 stock mask")
    if "lookback_length" in values and "horizon" in values:
        lookback = int(values["lookback_length"])
        horizon = int(values["horizon"])
    else:
        config_path = path.parent / "config.json"  # older TimeFilter output
        if not config_path.exists():
            raise ValueError(f"{path}: missing lookback/horizon metadata and config.json")
        args = json.loads(config_path.read_text(encoding="utf-8"))["args"]
        lookback, horizon = int(args["seq_len"]), int(args["pred_len"])
    if (lookback, horizon) != (16, 1):
        raise ValueError(f"{path}: expected 16-day lookback and 1-day horizon, got {lookback}/{horizon}")
    return values


def reference_evaluator(path):
    spec = importlib.util.spec_from_file_location("stockmixer_reference_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def evaluate_files(named_paths, data_path, reference_path=None):
    target = np.arange(TEST_START, TEST_END)
    raw = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (STOCKS, START_DAY + TEST_END, 5):
        raise ValueError(f"{data_path}: unexpected shape {raw.shape}")
    source_days = target + START_DAY
    previous_close = raw[:, source_days - 1, -1]
    expected_truth = raw[:, source_days, -1] / previous_close - 1
    stockmixer_evaluate = reference_evaluator(reference_path) if reference_path else None
    report = {
        "protocol": "StockMixer paper SP500: 16-day lookback, next-day return, test target 1259..1610",
        "data_sha256": file_sha256(data_path),
        "results": {},
    }
    for name, path in named_paths.items():
        values = load_predictions(path, target)
        if not np.allclose(values["ground_truth"], expected_truth, rtol=0, atol=2e-7):
            raise ValueError(f"{path}: return labels differ from the specified SP500.npy")
        metrics = compute_metrics(values["prediction"], values["ground_truth"], values["mask"])
        if stockmixer_evaluate:
            original = stockmixer_evaluate(values["prediction"], values["ground_truth"], values["mask"])
            for key in REFERENCE_KEYS:
                if not np.isclose(metrics[key], original[key], rtol=1e-5, atol=1e-7, equal_nan=True):
                    raise ValueError(f"{name}: {key} differs from the StockMixer evaluator")
        report["results"][name] = {key: float(value) if np.isfinite(value) else None
                                   for key, value in metrics.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=FILE",
                        help="Repeat for TimeFilter and StockMixer test_predictions.npz files.")
    parser.add_argument("--sp500-data", type=Path,
                        default=PROJECT_ROOT / "stockmixer_dataset" / "SP500" / "SP500.npy")
    parser.add_argument("--stockmixer-evaluator", type=Path,
                        help="Optional path to StockMixer's original src/evaluator.py for parity checking.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    named_paths = {}
    for item in args.prediction:
        if "=" not in item:
            parser.error("--prediction must be NAME=FILE")
        name, filename = item.split("=", 1)
        if not name or name in named_paths:
            parser.error("Prediction names must be nonempty and unique")
        named_paths[name] = Path(filename)
    report = evaluate_files(named_paths, args.sp500_data, args.stockmixer_evaluator)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
