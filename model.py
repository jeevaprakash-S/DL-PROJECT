"""Compact Transformer World Model: next-flow forecast plus attack stage."""
import math
import torch
from torch import nn
import config

class PositionalEncoding(nn.Module):
    def __init__(self, width: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        rate = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        pe = torch.zeros(max_len, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * rate), torch.cos(position * rate)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.D_MODEL, config.N_HEADS, dropout=config.DROPOUT, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(config.D_MODEL, config.D_FF), nn.GELU(), nn.Dropout(config.DROPOUT), nn.Linear(config.D_FF, config.D_MODEL))
        self.n1, self.n2 = nn.LayerNorm(config.D_MODEL), nn.LayerNorm(config.D_MODEL)
    def forward(self, x):
        y, weights = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.n1(x + y)
        return self.n2(x + self.ff(x)), weights

class NetworkWorldModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.input = nn.Sequential(nn.Linear(num_features, config.D_MODEL), nn.GELU())
        self.pos = PositionalEncoding(config.D_MODEL)
        self.layers = nn.ModuleList(AttentionLayer() for _ in range(config.N_LAYERS))
        self.forecast_head = nn.Sequential(nn.LayerNorm(config.D_MODEL), nn.Linear(config.D_MODEL, config.D_MODEL), nn.GELU(), nn.Linear(config.D_MODEL, num_features))
        self.stage_head = nn.Sequential(nn.LayerNorm(config.D_MODEL), nn.Linear(config.D_MODEL, config.D_MODEL // 2), nn.GELU(), nn.Dropout(config.DROPOUT), nn.Linear(config.D_MODEL // 2, config.NUM_STAGES))
    def forward(self, x, return_attention=False):
        h = self.pos(self.input(x)); attention = []
        for layer in self.layers:
            h, a = layer(h); attention.append(a)
        state = h[:, -1]
        result = (self.forecast_head(state), self.stage_head(state))
        return (*result, attention) if return_attention else result

def build_model(num_features: int):
    model = NetworkWorldModel(num_features).to(config.DEVICE)
    print(f"NetworkWorldModel: {sum(p.numel() for p in model.parameters()):,} parameters on {config.DEVICE}")
    return model
