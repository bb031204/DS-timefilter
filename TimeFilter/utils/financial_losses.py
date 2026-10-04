"""Finance-only objectives for next-day cross-sectional return prediction."""

import torch.nn.functional as F


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
    mse = F.mse_loss(prediction * mask, target * mask)
    return mse + rank_weight * stockmixer_rank_loss(prediction, target, mask)
