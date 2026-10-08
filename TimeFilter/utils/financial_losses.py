"""Finance-only objectives for next-day cross-sectional return prediction."""

import torch.nn.functional as F
import torch


def stockmixer_masked_mse_loss(prediction, target, mask):
    """Official StockMixer MSE: masked errors averaged over all stocks."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError('StockMixer MSE requires matching prediction, target and mask')
    return F.mse_loss(prediction * mask, target * mask)


def stockmixer_rank_loss(prediction, target, mask):
    """StockMixer's pairwise hinge loss, independently for each target day.

    Inputs are [batch days, 1, stocks]. As in StockMixer, the mean includes
    masked pairs as zeros; SP500 has an all-one mask.
    """
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 3 or prediction.shape[1] != 1:
        raise ValueError('Ranking loss expects matching [days, 1, stocks] tensors')
    predicted = prediction[:, 0, :]
    actual = target[:, 0, :]
    valid = mask[:, 0, :]
    predicted_difference = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
    opposite_actual_difference = actual.unsqueeze(-2) - actual.unsqueeze(-1)
    pair_mask = valid.unsqueeze(-1) * valid.unsqueeze(-2)
    daily_loss = F.relu(predicted_difference * opposite_actual_difference * pair_mask)
    return daily_loss.mean(dim=(-2, -1)).mean()


def stockmixer_validation_loss(prediction, target, mask, rank_weight=0.1):
    """Original StockMixer validation objective on predicted daily returns.

    SP500 uses an all-one mask. The mean over a batch of days equals the mean
    of StockMixer's one-day MSE + rank losses when batches are day-weighted.
    """
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError('StockMixer validation loss requires matching arrays')
    mse = stockmixer_masked_mse_loss(prediction, target, mask)
    return mse + rank_weight * stockmixer_rank_loss(prediction, target, mask)


def daily_pearson_ic_loss(prediction, target, mask):
    """Mean 1-Pearson IC across valid target days; degenerate targets are skipped.

    The denominator has a small stabilizer, so constant predictions retain a
    gradient. Masked stocks never contribute to the correlation.
    """
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 3 or prediction.shape[1] != 1:
        raise ValueError('IC loss expects matching [days, 1, stocks] tensors')
    pred = prediction[:, 0, :]
    true = target[:, 0, :]
    valid = mask[:, 0, :] > 0
    count = valid.sum(dim=-1, keepdim=True)
    pred_mean = (pred * valid).sum(dim=-1, keepdim=True) / count.clamp_min(1)
    true_mean = (true * valid).sum(dim=-1, keepdim=True) / count.clamp_min(1)
    pred_centered = (pred - pred_mean) * valid
    true_centered = (true - true_mean) * valid
    true_energy = true_centered.square().sum(dim=-1)
    pred_energy = pred_centered.square().sum(dim=-1)
    eligible = (count[:, 0] >= 2) & (true_energy > 1e-12)
    if not eligible.any():
        return pred.sum() * 0.0
    numerator = (pred_centered * true_centered).sum(dim=-1)
    denominator = torch.sqrt(pred_energy * true_energy + 1e-8)
    ic = numerator / denominator
    return (1.0 - ic[eligible]).mean()
