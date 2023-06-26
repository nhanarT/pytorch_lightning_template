# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typeguard import check_argument_types

from . import eend_ola_feature

class WavFrontendMel(nn.Module):
    """Conventional frontend structure for ASR.
    """

    def __init__(
            self,
            fs: int = 16000,
            n_mels: int = 80,
            frame_length: int = 25,
            frame_shift: int = 10,
            context_size: int = 0,
            subsampling: int = 1,
    ):
        assert check_argument_types()
        super().__init__()
        self.fs = fs
        self.frame_length = int(frame_length * fs / 1000)
        self.frame_shift = int(frame_shift * fs / 1000)
        self.context_size = context_size
        self.subsampling = subsampling
        self.n_mels = n_mels

    def output_size(self) -> int:
        return self.n_mels * (2 * self.context_size + 1)

    def forward(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            waveform = waveform.cpu().numpy()
            mat = eend_ola_feature.stft(waveform, self.frame_length, self.frame_shift)
            mat = eend_ola_feature.transform(mat, self.fs, self.n_mels)
            if self.context_size != 0:
                mat = eend_ola_feature.splice(mat, context_size=self.context_size)
            if self.subsampling != 1:
                mat = mat[::self.subsampling]
            mat = torch.from_numpy(mat.copy())
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens
