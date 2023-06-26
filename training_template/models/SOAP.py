from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typeguard import check_argument_types
from ..utils.told_modules.power import create_powerlabel

class SOAP(nn.Module):
    """Speaker overlap-aware post-processing"""

    def __init__(
        self,
        soap_encoder,
        ci_scorer,
        cd_scorer,
        mapping_dict,
        max_n_speaker,
        max_olp_speaker,
        guidance_loss_weight,
    ):
        super().__init__()
        self.soap_encoder = soap_encoder # Done
        self.ci_scorer = ci_scorer # Done
        self.cd_scorer = cd_scorer # Done
        self.mapping_dict = mapping_dict # Done
        self.max_n_speaker = max_n_speaker
        self.max_olp_speaker = max_olp_speaker
        self.guidance_loss_weight = guidance_loss_weight
        self.subsampling = 2 * len(self.soap_encoder.layers_in_block)

        self.output_layer = nn.LSTM(2 * self.max_n_speaker, mapping_dict['oov'] + 1)

    def forward(self, speech, speech_len, profiles, binary_labels, binary_label_lens, perms=None):
        '''
            Args:
                speech: (B, T, ...)
                speech_len: (B,)
                profiles: (B, N, D)
                binary_labels: (B, T, N_speaker)
                binary_label_lens: (B,)
                perms: (B, N_speaker)

            Return:
                loss
        '''
        # if perms is not None:
            # binary_labels = binary_labels[torch.arange(binary_labels.size(0)).unsqueeze(1),::self.subsampling, perms].transpose(1,2)
        # else:
        binary_labels = binary_labels[:, ::self.subsampling, :]
        binary_label_lens = binary_label_lens // self.subsampling
        pse_labels = torch.stack([create_powerlabel(label, self.mapping_dict, self.max_n_speaker, self.max_olp_speaker) for label in binary_labels])
        # tmp_predict_probs = []
        # for b_idx in range(pse_labels.size(0)):
            # tmp_predict_prob = F.one_hot(pse_labels[b_idx,:], self.mapping_dict['oov']+1)
            # tmp_predict_probs.append(tmp_predict_prob)
        # pse_labels = torch.stack(tmp_predict_probs) # (B, T, N_pse_labels)

        # SOAP Encoding
        soap_encoded_speech, soap_encoded_speech_length, _ = self.soap_encoder(speech, speech_len) # (B, T, D)
        
        ci_scores = self.ci_scorer(profiles, soap_encoded_speech) # (B, T, N)
        cd_scores = self.cd_scorer(profiles, soap_encoded_speech) # (B, T, N)

        x = torch.cat([ci_scores, cd_scores], dim=-1) # (B, T, 2N)
        logits, (_, _) = self.output_layer(x)

        # length_diff_tolerance = 2
        # length_diff = abs(pse_labels.shape[1] - logits.shape[1])
        # if length_diff <= length_diff_tolerance:
            # min_len = min(logits.shape[1], pse_labels.shape[1])
            # pse_labels = pse_labels[:, :min_len]
            # binary_labels = binary_labels[:, :min_len]
            # logits = logits[:, :min_len]
            # cd_scores = cd_scores[:, :min_len]
            # ci_scores = ci_scores[:, :min_len]

        # Length matching
        new_pse_labels    = []
        new_binary_labels = []
        new_logits        = []
        new_ci_scores     = []
        new_cd_scores     = []
        for b_idx, (feat_len, lab_len) in enumerate(zip(soap_encoded_speech_length, binary_label_lens)):
            min_len = int(min(feat_len, lab_len))
            binary_label_lens[b_idx] = min_len
            new_pse_labels.append(pse_labels[b_idx, :min_len]) 
            new_binary_labels.append(binary_labels[b_idx, :min_len]) 
            new_logits.append(logits[b_idx, :min_len]) 
            new_ci_scores.append(ci_scores[b_idx, :min_len]) 
            new_cd_scores.append(cd_scores[b_idx, :min_len]) 

        CE_loss = torch.hstack([F.cross_entropy(input=logit, target=pse_label) for logit, pse_label in zip(new_logits, new_pse_labels)]).mean()
        # CE_loss = F.cross_entropy(input=logits.transpose(1,2), target=pse_labels)

        guide_loss = torch.hstack([F.binary_cross_entropy(input=ci_score, target=binary_label)+\
                       F.binary_cross_entropy(input=cd_score, target=binary_label)
                       for ci_score, cd_score, binary_label in zip(new_ci_scores, new_cd_scores, new_binary_labels)]).mean()
        # guidance_loss = \
            # F.binary_cross_entropy(input=ci_scores, target=binary_labels) +\
            # F.binary_cross_entropy(input=cd_scores, target=binary_labels)

        return (
            CE_loss + self.guidance_loss_weight * guide_loss,
            new_logits,
            new_binary_labels,
        )


    def estimate_sequential(self,
                            speech,
                            speech_len,
                            profiles):

        soap_encoded_speech, _, _ = self.soap_encoder(speech, speech_len)
        
        ci_scores = self.ci_scorer(profiles, soap_encoded_speech)
        cd_scores = self.cd_scorer(profiles, soap_encoded_speech)
        x = torch.cat([ci_scores, cd_scores], dim=-1)
        logits, (_, _) = self.output_layer(x)
        
        return logits
