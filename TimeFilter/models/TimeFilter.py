import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)
    
    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x) # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1) # L
        self.financial_input_features = getattr(configs, 'financial_input_features', 'returns')
        if self.financial_input_features not in ('returns', 'eod5'):
            raise ValueError('financial_input_features must be returns or eod5')
        if self.financial_input_features == 'eod5' and getattr(configs, 'financial_norm', 1):
            raise ValueError('eod5 input requires financial_norm=0')

        # Filter
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # embed
        input_features = 5 if self.financial_input_features == 'eod5' else 1
        self.patch_embed = PatchEmbed(self.dim, self.patch_len * input_features,
                                     self.stride * input_features, configs.pos)

        # TimeFilter Backbone
        self.backbone = TimeFilter_Backbone(self.dim, self.n_vars, self.d_ff,
                                  configs.n_heads, configs.e_layers, self.top_p, configs.dropout, self.seq_len * self.n_vars // self.patch_len)
        
        # head
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        # Without RevIN
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN,
                              non_norm=not bool(getattr(configs, 'financial_norm', 1)))
    
    def forward(self, x, masks, is_training=False, target=None):
        if self.financial_input_features == 'eod5':
            # Finance-only adapter: one token per stock and time patch, with
            # all five indicators retained inside the patch projection.
            if x.ndim != 4 or x.shape[2:] != (self.n_vars, 5) or x.shape[1] != self.seq_len:
                raise ValueError('eod5 input must be [batch, seq_len, stocks, 5]')
            B, T, C, F = x.shape
            x = x.permute(0, 2, 1, 3).reshape(B, C * T * F)
        else:
            # Original TimeFilter path: [B, T, C].
            B, T, C = x.shape
            x = self.norm(x, 'norm')
            x = x.permute(0, 2, 1).reshape(B, C * T)
        x = self.patch_embed(x) # [B, N, D]  N = [C*T / P]

        x, moe_loss = self.backbone(x, masks, self.alpha, is_training)

        # [B, C, T/P, D]
        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2)) # [B, C, T]
        x = x.permute(0, 2, 1)
        # De-Normalization
        if self.financial_input_features != 'eod5':
            x = self.norm(x, 'denorm')

        return x, moe_loss
