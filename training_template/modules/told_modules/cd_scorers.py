import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MultiHeadSelfAttention
from .encoder import PositionwiseFeedForward

class CD_Scorer(torch.nn.Module):
    def __init__(self,
                 in_channels=512,
                 h=4,
                 pff_channels=1024,
                 n_layers=4,
                 dropout_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.h = h
        self.pff_channels = pff_channels
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        for i in range(n_layers):
            setattr(self, '{}{:d}'.format("lnorm1_", i),
                    nn.LayerNorm(in_channels))
            setattr(self, '{}{:d}'.format("self_att_", i),
                    MultiHeadSelfAttention(in_channels, h, dropout_rate))
            setattr(self, '{}{:d}'.format("lnorm2_", i),
                    nn.LayerNorm(in_channels))
            setattr(self, '{}{:d}'.format("ff_", i),
                    PositionwiseFeedForward(in_channels, pff_channels, dropout_rate))

        self.lnorm_out = nn.LayerNorm(in_channels)
        self.output_layer = nn.Linear(in_channels, 1)


    def forward(self, profiles, encoded_feats):
        # profiles: (B, Max_n, emb_dim)
        # encoded_feats: (B, T, emb_dim)
        B = profiles.size(0)
        N, T = profiles.size(1), encoded_feats.size(1)
        profiles = profiles.unsqueeze(1)                    # B, 1, N, D
        profiles = profiles.expand(-1, T, -1, -1)           # B, T, N, D

        encoded_feats = encoded_feats.unsqueeze(2)          # B, T, 1, D
        encoded_feats = encoded_feats.expand_as(profiles)   # B, T, N, D

        x = torch.cat([profiles, encoded_feats], dim=-1)    # B, T, N, 2D
        x = x.permute(0,2,1,3).reshape(B*N, T, -1)

        for i in range(self.n_layers):
            x = getattr(self, '{}{:d}'.format("lnorm1_", i))(x)
            s = getattr(self, '{}{:d}'.format("self_att_", i))(x, x.shape[0], None)
            x = x + self.dropout(s).reshape(B*N, T, -1)
            x = getattr(self, '{}{:d}'.format("lnorm2_", i))(x)
            s = getattr(self, '{}{:d}'.format("ff_", i))(x)
            x = x + self.dropout(s)
        x = self.lnorm_out(x)

        x = x.reshape(B, N, T, -1).permute(0,2,1,3)        # B, T, N, out_channels 
        scores = self.output_layer(x).squeeze(-1)         # B, T, N
        scores = torch.sigmoid(scores)
        return scores
