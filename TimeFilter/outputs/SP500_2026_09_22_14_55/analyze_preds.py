"""Quantify why the SP500 run has no cross-sectional signal."""
import numpy as np

run = r"D:\finance\model\Timefilter\TimeFilter\outputs\SP500_2026_09_22_14_55\results\long_term_forecast_SP500_returns_16_1_TimeFilter_SP500_ftM_sl16_ll0_pl1_dm512_nh4_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0\financial\test_predictions.npz"
data = np.load(run)
print("keys:", list(data.keys()))
for k in data.keys():
    print(k, data[k].shape, data[k].dtype)

preds = data["predictions"] if "predictions" in data else data[list(data.keys())[0]]
trues = data["targets"] if "targets" in data else data[list(data.keys())[1]]

# array_layout: [stocks, prediction_days]
p = preds.reshape(preds.shape[0], -1)
t = trues.reshape(trues.shape[0], -1)
print("\n--- global stats ---")
print(f"pred  mean={p.mean():.6f} std={p.std():.6f} min={p.min():.6f} max={p.max():.6f}")
print(f"true  mean={t.mean():.6f} std={t.std():.6f} min={t.min():.6f} max={t.max():.6f}")

# Cross-sectional dispersion per day (the only thing IC/RankIC can see)
p_xstd = p.std(axis=0)
t_xstd = t.std(axis=0)
print("\n--- per-day cross-sectional std (across stocks) ---")
print(f"pred cross-std mean={p_xstd.mean():.6f}  (range {p_xstd.min():.6f}..{p_xstd.max():.6f})")
print(f"true cross-std mean={t_xstd.mean():.6f}  (range {t_xstd.min():.6f}..{t_xstd.max():.6f})")
print(f"ratio pred/true cross-std = {p_xstd.mean() / t_xstd.mean():.4f}")

# Time-series variation per stock
print("\n--- per-stock time-series std (across days) ---")
print(f"pred time-std mean={p.std(axis=1).mean():.6f}")
print(f"true time-std mean={t.std(axis=1).mean():.6f}")

# A model that predicts a near-constant per stock: how much of pred variance is
# explained by the per-stock mean alone?
stock_mean = p.mean(axis=1, keepdims=True)
ss_total = ((p - p.mean()) ** 2).sum()
ss_between = ((stock_mean - p.mean()) ** 2).sum() * p.shape[1]
print(f"\nshare of pred variance from per-stock mean alone: {ss_between / ss_total:.4f}")

# Daily Pearson IC recomputed from raw arrays
ics = []
for d in range(p.shape[1]):
    if p[:, d].std() > 0 and t[:, d].std() > 0:
        ics.append(np.corrcoef(p[:, d], t[:, d])[0, 1])
ics = np.array(ics)
print(f"\ndaily Pearson IC: mean={ics.mean():.6f} std={ics.std():.6f} ICIR={ics.mean() / ics.std():.6f}")

# Predict the previous-day return instead: a naive momentum/reversal baseline
# to show what signal magnitude is even available in this label space.
lags = []
for d in range(1, p.shape[1]):
    if t[:, d - 1].std() > 0 and t[:, d].std() > 0:
        lags.append(np.corrcoef(t[:, d - 1], t[:, d])[0, 1])
lags = np.array(lags)
print(f"autocorr of true returns (lag-1, cross-sectional): mean={lags.mean():.6f}")
