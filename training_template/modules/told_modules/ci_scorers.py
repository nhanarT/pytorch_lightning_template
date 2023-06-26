import torch
import torch.nn as nn
import torch.nn.functional as F

def fc_layers(in_channels, out_channels, dropout_rate):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels, bias=False),
        nn.LayerNorm(out_channels),
        nn.Dropout(dropout_rate),
        nn.ReLU()
    )


class CI_Scorer(nn.Module):
    def __init__(self, in_channels, n_units, n_layers=3, dropout_rate=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_units = n_units

        modules = torch.nn.ModuleList()
        for i in range(n_layers):
            modules.append(fc_layers(in_channels if i == 0 else n_units, n_units, dropout_rate))
        self.blocks = modules

        self.output_layer = nn.Linear(n_units, 1)


    def forward(self, profiles, encoded_feats):
        # profiles: (B, Max_n, emb_dim)
        # encoded_feats: (B, T, emb_dim)
        N, T = profiles.size(1), encoded_feats.size(1)
        profiles = profiles.unsqueeze(1)                    # B, 1, N, D
        profiles = profiles.expand(-1, T, -1, -1)           # B, T, N, D

        encoded_feats = encoded_feats.unsqueeze(2)          # B, T, 1, D
        encoded_feats = encoded_feats.expand_as(profiles)   # B, T, N, D

        x = torch.cat([profiles, encoded_feats], dim=-1)
        for block in self.blocks:
            profiles = block(x)                      # B, T, N, D
        scores = torch.sigmoid(self.output_layer(x)).squeeze(-1)
        return scores
