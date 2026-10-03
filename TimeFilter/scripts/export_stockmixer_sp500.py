"""Export a wrapper checkpoint with the original StockMixer model and data."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


START_DAY = 915
VALID_INDEX = 1006
TEST_INDEX = 1259
END_INDEX = 1611
LOOKBACK = 16
STOCKS = 474


def export(checkpoint_path, stockmixer_root, output_path, data_path=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    data_config = config["data"]
    expected = {
        "market_name": "SP500", "stock_num": STOCKS, "fea_num": 5,
        "lookback_length": LOOKBACK, "steps": 1,
        "valid_index": VALID_INDEX, "test_index": TEST_INDEX,
    }
    mismatches = {key: (data_config.get(key), value) for key, value in expected.items()
                  if data_config.get(key) != value}
    if mismatches:
        raise ValueError(f"Checkpoint does not use the paper S&P500 protocol: {mismatches}")
    data_path = data_path or stockmixer_root / data_config["dataset_root"] / "SP500" / "SP500.npy"
    raw = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (STOCKS, START_DAY + END_INDEX, 5):
        raise ValueError(f"Unexpected SP500.npy shape: {raw.shape}")

    spec = importlib.util.spec_from_file_location("stockmixer_model", stockmixer_root / "src" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.StockMixer(stocks=STOCKS, time_steps=LOOKBACK, channels=5,
                              market=int(config["model"]["market_num"]),
                              scale=int(config["model"]["scale_factor"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    target = np.arange(TEST_INDEX, END_INDEX)
    prediction = np.empty((STOCKS, len(target)), dtype=np.float32)
    ground_truth = np.empty_like(prediction)
    with torch.inference_mode():
        for column, target_day in enumerate(target):
            source_day = START_DAY + target_day
            history = np.array(raw[:, source_day - LOOKBACK:source_day, :], dtype=np.float32, copy=True)
            previous_close = np.asarray(raw[:, source_day - 1, -1], dtype=np.float32)
            next_close = np.asarray(raw[:, source_day, -1], dtype=np.float32)
            predicted_close = model(torch.from_numpy(history).to(device)).cpu().numpy().reshape(STOCKS)
            prediction[:, column] = (predicted_close - previous_close) / previous_close
            ground_truth[:, column] = (next_close - previous_close) / previous_close
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, prediction=prediction, ground_truth=ground_truth,
                        mask=np.ones_like(prediction), target_index=target,
                        source_day_index=target + START_DAY,
                        lookback_length=np.asarray(LOOKBACK), horizon=np.asarray(1))
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Modified wrapper's outputs/.../checkpoints/best_model.pt; original train.py saves none.")
    parser.add_argument("--stockmixer-root", type=Path,
                        default=Path(r"D:\finance\baseline\StockMixer-master"),
                        help="Original StockMixer source/data root for model inference.")
    parser.add_argument("--sp500-data", type=Path, help="Optional explicit SP500.npy path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    path = export(args.checkpoint, args.stockmixer_root, args.output, args.sp500_data, args.device)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
