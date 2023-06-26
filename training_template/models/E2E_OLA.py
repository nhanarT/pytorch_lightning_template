# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Dict
from typing import Tuple
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as  nn
from typeguard import check_argument_types

from ..modules.told_modules.encoder import EENDOLATransformerEncoder
from ..modules.told_modules.encoder_decoder_attractor import EncoderDecoderAttractor
from ..utils.told_modules.power import create_powerlabel
from ..losses.told_modules.losses import batch_pit_n_speaker_loss


class E2E_OLA(nn.Module):
    """EEND-OLA diarization model"""

    @staticmethod
    def pad_attractor(att, max_n_speakers):
        C, D = att.shape
        if C < max_n_speakers:
            att = torch.cat([att, torch.zeros(max_n_speakers - C, D).to(torch.float32).to(att.device)], dim=0)
        return att

    def __init__(
            self,
            encoder: EENDOLATransformerEncoder,
            encoder_decoder_attractor: EncoderDecoderAttractor,
            subsampling: int = 10,
            n_units: int = 256,
            max_n_speaker: int = 8,
            attractor_loss_weight: float = 1.0,
            mapping_dict = None,
            TOLD_part = False,
            **kwargs,
    ):
        assert check_argument_types()

        super().__init__()
        self.enc = encoder
        self.eda = encoder_decoder_attractor
        self.subsampling = subsampling
        self.attractor_loss_weight = attractor_loss_weight
        self.max_n_speaker = max_n_speaker
        self.TOLD_part = TOLD_part
        
        assert mapping_dict is not None
        self.mapping_dict = mapping_dict

        # PostNet
        self.postnet = nn.LSTM(self.max_n_speaker, n_units, 1, batch_first=True)
        self.output_layer = nn.Linear(n_units, mapping_dict['oov'] + 1)


    def forward_encoder(self, xs, ilens):
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=-1)
        pad_shape = xs.shape
        xs_mask = [torch.ones(ilen).to(xs.device) for ilen in ilens]
        xs_mask = torch.nn.utils.rnn.pad_sequence(xs_mask, batch_first=True, padding_value=0).unsqueeze(-2)
        emb = self.enc(xs, xs_mask)
        # Parallel
        # emb = emb.view(pad_shape[0], pad_shape[1], -1)
        # Non-parallel
        emb = torch.split(emb.view(pad_shape[0], pad_shape[1], -1), 1, dim=0)
        emb = [e[0][:ilen] for e, ilen in zip(emb, ilens)]
        return emb


    def forward_post_net(self, logits, ilens):
        maxlen = torch.max(ilens).to(torch.int).item()
        logits = nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=-1)
        logits = nn.utils.rnn.pack_padded_sequence(logits, ilens.cpu().to(torch.int64), batch_first=True, enforce_sorted=False)
        outputs, (_, _) = self.postnet(logits)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=-1, total_length=maxlen)[0]
        outputs = [output[:ilens[i].to(torch.int).item()] for i, output in enumerate(outputs)]
        outputs = [self.output_layer(output) for output in outputs]
        return outputs


    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        labels: torch.Tensor,
        n_speakers=None,
        shuffle=True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            labels: (Batch, Length, Num_speaker)
            n_speakers: (Batch,)
        """
        labels = labels[:, ::self.subsampling, :]
        if n_speakers is None:
            n_speakers = [label.shape[1] for label in labels]

        speech = [s[:s_len] for s, s_len in zip(speech, speech_lengths)]
        emb = self.forward_encoder(speech, speech_lengths)

        if shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractor_loss, attractors = self.eda(
                [e[order] for e, order in zip(emb, orders)],
                n_speakers
            )
        else:
            attractor_loss, attractors = self.eda(emb, n_speakers)

        # Detach attactor loss
        y_preds = [torch.matmul(e, att.permute(1, 0).detach()) for e, att in zip(emb, attractors)]
        # y_preds = torch.matmul(emb, attractors.permute(0, 2, 1))

        try:
            pit_loss, labels_perm, perms_idx = batch_pit_n_speaker_loss(y_preds, labels, n_speakers)
        except BaseException as err:
            for e, att in zip(emb, attractors):
                print(e.shape, att.shape)
            raise err

        # PSE loss
        # pse_label = [N x (T, 1)]
        perm_pse_labels = [create_powerlabel(label_perm, self.mapping_dict) for label_perm in labels_perm]
        perm_pse_labels = [perm_pse_label[:ilen] for perm_pse_label, ilen in zip(perm_pse_labels, speech_lengths)]
        # logits =  [N x (T, N_pse)]
        logits = self.forward_post_net(y_preds, speech_lengths)
        pse_loss = torch.hstack([nn.functional.cross_entropy(input=logit, target=perm_pse_label) for logit, perm_pse_label in zip(logits, perm_pse_labels)]).mean()

        if self.TOLD_part:
            return logits, n_speakers, perms_idx

        return (
            pit_loss + pse_loss + self.attractor_loss_weight * attractor_loss,
            n_speakers,
            logits,
            labels_perm,
        )


    def estimate_sequential(self,
                            speech: torch.Tensor,
                            speech_lengths: torch.Tensor,
                            n_speakers = None,
                            threshold = 0.5,
                            **kwargs):
        speech = [s[:s_len] for s, s_len in zip(speech, speech_lengths)]
        emb = self.forward_encoder(speech, speech_lengths)
        attractors, probs = self.eda.estimate(emb)
        attractors_active = []

        for idx, (p, att, e) in enumerate(zip(probs, attractors, emb)):
            # if (n_speakers is not None) and (n_speakers >= 0):
            # ge_zero_flag = False
            # try:
                # ge_zero_flag = n_speakers[idx] >= 0
            # except TypeError:
                # pass

            if (n_speakers is not None) and (n_speakers[idx]>=0):
                att = att[:n_speakers[idx].to(torch.long).item(), ]
                attractors_active.append(att)
            elif threshold is not None:
                silence = torch.nonzero(p < threshold)[0]
                n_spk = silence if silence.size else None
                att = att[:n_spk, ]
                attractors_active.append(att)
            else:
                NotImplementedError('n_speakers or threshold has to be given.')

        raw_n_speakers = [att.shape[0] for att in attractors_active]
        attractors = [self.pad_attractor(att, self.max_n_speaker) if att.shape[0] <= self.max_n_speaker else att[:self.max_n_speaker] for att in attractors_active]
        ys = [torch.matmul(e, att.permute(1, 0)) for e, att in zip(emb, attractors)]
        logits = self.forward_post_net(ys, speech_lengths)
        ys = [self.recover_y_from_powerlabel(logit, raw_n_speaker) for logit, raw_n_speaker in
              zip(logits, raw_n_speakers)]
        
        if self.TOLD_part:
            return logits, raw_n_speakers

        return ys, emb, attractors, raw_n_speakers


    def recover_y_from_powerlabel(self, logit, n_speaker):
        pred = torch.argmax(torch.softmax(logit, dim=-1), dim=-1)
        oov_index = torch.where(pred == self.mapping_dict['oov'])[0]
        for i in oov_index:
            if i > 0:
                pred[i] = pred[i - 1]
            else:
                pred[i] = 0
        pred = [self.inv_mapping_func(i) for i in pred]
        decisions = [bin(num)[2:].zfill(self.max_n_speaker)[::-1] for num in pred]
        decisions = torch.from_numpy(
            np.stack([np.array([int(i) for i in dec]) for dec in decisions], axis=0)).to(logit.device).to(torch.float32)
        decisions = decisions[:, :n_speaker]
        return decisions


    def inv_mapping_func(self, label):

        if not isinstance(label, int):
            label = int(label)
        if label in self.mapping_dict['label2dec'].keys():
            num = self.mapping_dict['label2dec'][label]
        else:
            num = -1
        return num
