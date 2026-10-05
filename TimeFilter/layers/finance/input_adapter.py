"""Per-stock five-feature preprocessing before TimeFilter patch projection."""

import torch
from torch import nn


class FinanceInputAdapter(nn.Module):
    def __init__(self, seq_len, patch_len, config):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.input_norm = nn.LayerNorm((seq_len, 5)) if config['input_layernorm'] else None
        indicator = config['indicator_mixer']
        temporal = config['temporal_mixer']
        self.indicator_mixer = (nn.Sequential(
            nn.LayerNorm(5), nn.Linear(5, indicator['hidden_dim']), nn.GELU(),
            nn.Linear(indicator['hidden_dim'], 5)) if indicator['enabled'] else None)
        self.temporal_mixer = (nn.Sequential(
            nn.LayerNorm(patch_len), nn.Linear(patch_len, temporal['hidden_dim']), nn.GELU(),
            nn.Linear(temporal['hidden_dim'], patch_len)) if temporal['enabled'] else None)

    def forward(self, x):
        # [batch, days, stocks, features] -> [batch, stocks, days, features].
        batch, days, stocks, features = x.shape
        if days != self.seq_len or features != 5:
            raise ValueError('FinanceInputAdapter expects [batch, seq_len, stocks, 5]')
        x = x.permute(0, 2, 1, 3)
        if self.input_norm is not None:
            # StockMixer-style history/feature normalization, using only this input window.
            x = self.input_norm(x)
        if self.indicator_mixer is not None:
            x = x + self.indicator_mixer(x)
        if self.temporal_mixer is not None:
            patches = x.reshape(batch, stocks, days // self.patch_len, self.patch_len, 5)
            patches = patches.permute(0, 1, 2, 4, 3)
            patches = patches + self.temporal_mixer(patches)
            x = patches.permute(0, 1, 2, 4, 3).reshape(batch, stocks, days, 5)
        return x.permute(0, 2, 1, 3)
