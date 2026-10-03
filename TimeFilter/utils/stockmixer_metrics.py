"""Financial metrics; SP500 reference metrics match original StockMixer.

The four calculation functions below are copied without changes from
D:/finance/model/Signed_StockMixer/src/train_signed_stockmixer.py.
Inputs use [stocks, prediction_days], with returns in decimal units.
For SP500's all-valid mask, IC/RIC/precision/SR match
D:/finance/baseline/StockMixer-master/src/evaluator.py.
RIC is Pearson ICIR in released code; RankIC is mean daily Spearman.
"""

import numpy as np


def rankdata_average(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan
    if method == "spearman":
        x = rankdata_average(x)
        y = rankdata_average(y)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 1e-12:
        return np.nan
    return float(np.sum(x * y) / denom)


def original_stockmixer_metrics(prediction, ground_truth, mask):
    assert ground_truth.shape == prediction.shape, "shape mis-match"
    performance = {}
    mask_sum = np.sum(mask)
    performance["mse"] = float(np.linalg.norm((prediction - ground_truth) * mask) ** 2 / mask_sum) if mask_sum > 0 else np.nan

    masked_prediction = prediction * mask
    masked_ground_truth = ground_truth * mask
    ic = []
    sharpe_li5 = []
    prec_10 = []

    for col in range(prediction.shape[1]):
        ic.append(safe_corr(masked_prediction[:, col], masked_ground_truth[:, col], "pearson"))

        rank_pre = np.argsort(prediction[:, col])
        pre_top5 = []
        pre_top10 = []
        for rank_pos in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * rank_pos]
            if mask[cur_rank][col] < 0.5:
                continue
            if len(pre_top5) < 5:
                pre_top5.append(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.append(cur_rank)
            if len(pre_top5) >= 5 and len(pre_top10) >= 10:
                break

        if len(pre_top5) == 5:
            sharpe_li5.append(float(np.sum(ground_truth[pre_top5, col]) / 5.0))
        if len(pre_top10) == 10:
            prec_10.append(float(np.mean(ground_truth[pre_top10, col] >= 0)))

    ic_arr = np.asarray(ic, dtype=float)
    sharpe_arr = np.asarray(sharpe_li5, dtype=float)
    performance["IC"] = float(np.mean(ic_arr))
    performance["RIC"] = float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 0 else np.nan
    performance["prec_10"] = float(np.mean(prec_10)) if prec_10 else np.nan
    performance["sharpe5"] = float((np.mean(sharpe_arr) / np.std(sharpe_arr)) * 15.87) if sharpe_arr.size and np.std(sharpe_arr) > 0 else np.nan
    return performance


def compute_metrics(prediction, ground_truth, mask):
    valid = mask > 0.5
    if not np.any(valid):
        return {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "directional_accuracy": np.nan,
                "IC": np.nan, "RIC": np.nan, "prec_10": np.nan, "sharpe5": np.nan,
                "RankIC": np.nan, "RankIC_std": np.nan, "RankICIR": np.nan}
    err = (prediction - ground_truth)[valid]
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    da = float(np.mean(np.sign(prediction[valid]) == np.sign(ground_truth[valid])))
    rank_ic_values = []
    for col in range(prediction.shape[1]):
        day_valid = valid[:, col]
        if np.sum(day_valid) < 2:
            continue
        ric = safe_corr(prediction[day_valid, col], ground_truth[day_valid, col], "spearman")
        if np.isfinite(ric):
            rank_ic_values.append(ric)
    ric_arr = np.asarray(rank_ic_values, dtype=float)
    original = original_stockmixer_metrics(prediction, ground_truth, mask)
    metrics = {
        "mae": mae,
        "mse": original["mse"],
        "flat_valid_mse": mse,
        "rmse": float(np.sqrt(mse)),
        "directional_accuracy": da,
        "IC": original["IC"],
        "RIC": original["RIC"],
        "prec_10": original["prec_10"],
        "sharpe5": original["sharpe5"],
        "RankIC": float(np.mean(ric_arr)) if ric_arr.size else np.nan,
        "RankIC_std": float(np.std(ric_arr)) if ric_arr.size else np.nan,
    }
    metrics["RankICIR"] = float(metrics["RankIC"] / metrics["RankIC_std"]) if metrics["RankIC_std"] and np.isfinite(metrics["RankIC_std"]) else np.nan
    return metrics
