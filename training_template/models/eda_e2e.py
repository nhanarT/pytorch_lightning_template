from ..modules.hybrid_attention_encoder import TransformerModel
from ..losses.pit import batch_pit_n_speaker_loss, standard_loss
from ..utils.diarization import pad_labels

from typing import List, Tuple

import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F


"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""


class EncoderDecoderAttractor(Module):
    def __init__(
        self,
        n_units: int,
        encoder_dropout: float,
        decoder_dropout: float,
        detach_attractor_loss: bool,
    ) -> None:
        super(EncoderDecoderAttractor, self).__init__()
        self.encoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=encoder_dropout,
            batch_first=True,
            )
        self.decoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=decoder_dropout,
            batch_first=True,
            )
        self.counter = torch.nn.Linear(n_units, 1)
        self.n_units = n_units
        self.detach_attractor_loss = detach_attractor_loss

    def forward(self, xs: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
        _, (hx, cx) = self.encoder(xs)
        attractors, (_, _) = self.decoder(
            zeros.to(xs.device),
            (hx, cx)
        )
        return attractors

    def estimate(
        self,
        xs: torch.Tensor,
        max_n_speakers: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors from embedding sequences
         without prior knowledge of the number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          max_n_speakers (int)
        Returns:
          attractors: List of (N,D)-shaped attractors
          probs: List of attractor existence probabilities
        """
        zeros = torch.zeros((xs.shape[0], max_n_speakers, self.n_units))
        attractors = self.forward(xs, zeros.to(xs.device))
        probs = [torch.sigmoid(
            torch.flatten(self.counter(att)))
            for att in attractors]
        return attractors, probs

    def __call__(
        self,
        xs: torch.Tensor,
        n_speakers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors and loss from embedding sequences
        with given number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          n_speakers: List of number of speakers, or None if the number
                                of speakers is unknown (ex. test phase)
        Returns:
          loss: Attractor existence loss
          attractors: List of (N,D)-shaped attractors
        """
        max_n_speakers = max(n_speakers)
        # assumes all sequences have the same amount of speakers
        if xs.device == torch.device("cpu"):
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers]))
        else:
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units),
                device=torch.device("cuda"))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers])).to(torch.device("cuda"))

        attractors = self.forward(xs, zeros)
        if self.detach_attractor_loss:
            attractors = attractors.detach()
        logit = torch.cat([
            torch.reshape(self.counter(att), (-1, max_n_speakers + 1))
            for att, n_spk in zip(attractors, n_speakers)])
        loss = F.binary_cross_entropy_with_logits(logit, labels)
        n_speakers_predict=[]
        spk_predict=[]
        for i in range(len(labels)):
            p=logit[i].detach().sigmoid()
            silence = np.where(
                p.data.to("cpu") < 0.5)[0]
            n_spk = silence[0] if silence.size else -1
            if n_spk == n_speakers[i]:
                n_speakers_predict.append(1)
            else:
                n_speakers_predict.append(0)
            spk_predict.append(n_spk)
        # The final attractor does not correspond to a speaker so remove it
        attractors = attractors[:, :-1, :]
        return loss, attractors,n_speakers_predict,spk_predict


class TransformerEDADiarization(Module):

    def __init__(
        self,
        in_size: int,
        n_units: int,
        e_units: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        attractor_loss_ratio: float,
        attractor_encoder_dropout: float,
        attractor_decoder_dropout: float,
        detach_attractor_loss: bool,
        time_shuffle:bool,
    ) -> None:
        """ Self-attention-based diarization model.
        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        self.time_shuffle=time_shuffle
        super(TransformerEDADiarization, self).__init__()
        self.enc=TransformerModel(
            in_size,
            n_heads,
            n_units,
            n_layers,dim_feedforward=e_units,dropout=dropout
        )
        self.eda = EncoderDecoderAttractor(
            n_units,
            attractor_encoder_dropout,
            attractor_decoder_dropout,
            detach_attractor_loss,
        )
        self.attractor_loss_ratio = attractor_loss_ratio

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # emb: [(T, E), ...]
        emb = emb.reshape(pad_shape[0], pad_shape[1], -1)
        return emb
    def estimate_sequential(self, xs,estimate_spk_qty_thr=0.5,fixed=None):
        emb=self.get_embeddings(xs)
        ys_active=[]
        attractors, probs = self.eda.estimate(emb, max_n_speakers=10)
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))
        ys = [torch.sigmoid(y) for y in ys]
        for p, y in zip(probs, ys):
         
            silence = np.where(
                p.data.to("cpu") < estimate_spk_qty_thr)[0]
            n_spk = silence[0] if silence.size else None
            if fixed is not None:
                n_spk=fixed
            ys_active.append(y[:, :n_spk])
        return ys_active
    

    def forward(
        self,
        xs: torch.Tensor,
        ts: torch.Tensor,
        n_speakers=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if n_speakers is None:
            n_speakers = [t.shape[1] for t in ts]
        emb = self.get_embeddings(xs)

        if self.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractor_loss, attractors,acc_spk, spk_predict = self.eda(
                torch.stack([e[order] for e, order in zip(emb, orders)]),
                n_speakers)
        else:
            attractor_loss, attractors, acc_spk, spk_predict = self.eda(emb, n_speakers)

        # ys: [(T, C), ...]
        # ys: bs, time, channel
        # att: bs, speaker, channel
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))
        return ys, attractor_loss,acc_spk,spk_predict

    def get_loss(
        self,
        ys: torch.Tensor,
        target: torch.Tensor,
        n_speakers: List[int],
        attractor_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        ts_padded = target
        max_n_speakers = max(n_speakers)
        ts_padded = pad_labels(target, max_n_speakers)
        ys_padded = pad_labels(ys, max_n_speakers)
        loss, labels,_ = batch_pit_n_speaker_loss(
             ys_padded, ts_padded, n_speakers)
        loss = standard_loss(ys_padded, labels.float())

        return loss + attractor_loss * self.attractor_loss_ratio, loss, labels
