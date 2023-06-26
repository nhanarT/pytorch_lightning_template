import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.told_modules.resnet34_encoder import ResNet34SpL2RegDiar
from ..modules.told_modules.frontend.wav_frontend import WavFrontendMel
from ..modules.told_modules.encoder import EENDOLATransformerEncoder
from ..modules.told_modules.encoder_decoder_attractor import EncoderDecoderAttractor
from ..modules.told_modules.ci_scorers import CI_Scorer
from ..modules.told_modules.cd_scorers import CD_Scorer
from ..modules.told_modules.nets_utils import make_pad_mask
from ..utils.told_modules.power import generate_mapping_dict
from ..utils.told_modules.report import log

from .E2E_OLA import E2E_OLA
from .SOAP import SOAP

class TOLD(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()
        print(f'Initialize TOLD model for stage {config.stage}')
        self.config = config
        self.mapping_dict = generate_mapping_dict(
            max_speaker_num=config.max_n_speaker,
            max_olp_speaker_num=config.max_olp_speaker
        )

        if (config.stage == 1) or (config.stage == 'infer'):
            self.context_frontend = WavFrontendMel(
                fs=config.sr,
                n_mels=config.context_frontend.n_mels,
                frame_length=config.frame_length,
                frame_shift=config.frame_shift,
                context_size=config.context_frontend.context_size,
                subsampling=config.context_frontend.subsampling
            )

            e2e_ola_encoder = EENDOLATransformerEncoder(**config.e2e_ola.encoder_conf)
            e2e_ola_attractor = EncoderDecoderAttractor(**config.e2e_ola.eda_conf)
            self.e2e_ola = E2E_OLA(
                encoder=e2e_ola_encoder,
                encoder_decoder_attractor=e2e_ola_attractor,
                n_units=config.e2e_ola.n_units,
                max_n_speaker=config.max_n_speaker,
                attractor_loss_weight=config.e2e_ola.attractor_loss_weight,
                mapping_dict=self.mapping_dict,
                TOLD_part=(config.stage == 2)
            )

        if (config.stage == 2) or (config.stage == 'infer'):
            self.frontend = WavFrontendMel(
                fs=config.sr,
                n_mels=config.frontend.n_mels,
                frame_length=config.frame_length,
                frame_shift=config.frame_shift,
                context_size=config.frontend.context_size,
                subsampling=config.frontend.subsampling
            )

            self.profile_extractor = ResNet34SpL2RegDiar(input_size=config.frontend.n_mels, **config.profile_extractor_conf)
            soap_encoder = ResNet34SpL2RegDiar(input_size=config.frontend.n_mels, **config.soap.soap_encoder_conf)

            ci_scorer = CI_Scorer(**config.soap.ci_conf)
            cd_scorer = CD_Scorer(**config.soap.cd_conf)

            self.soap = SOAP(
                soap_encoder=soap_encoder,
                ci_scorer=ci_scorer,
                cd_scorer=cd_scorer,
                mapping_dict=self.mapping_dict,
                max_n_speaker=config.max_n_speaker,
                max_olp_speaker=config.max_olp_speaker,
                guidance_loss_weight=config.soap.guidance_loss_weight,
            )

        if config.load is not None:
            self.load_modules(config.load)

        if config.freeze is not None:
            self.freeze_modules(config.freeze)



    def forward(self, speech, speech_lengths, labels, label_lens, n_speakers=None, shuffle=True):
        '''
            Args:
                speech: (B, audio_len) torch.Tensor
                speech_lengths: (B,) torch.Tensor
                labels: (B, frame_len, N_speaker) torch.Tensor
                label_lengths: (B, ) torch.Tensor
                n_speakers:
                shuffle: boolean
                    whether to shuffle speech frame when training Stage 1 eend-ola
        '''
        if self.config.stage == 1:
            e2e_ola_speech, e2e_ola_speech_lengths = self.context_frontend(speech, speech_lengths)
            e2e_ola_speech, e2e_ola_speech_lengths = e2e_ola_speech.to(speech.device), e2e_ola_speech_lengths.to(speech.device)
            loss, n_speakers, logits, e2e_ola_labels = self.e2e_ola(e2e_ola_speech, e2e_ola_speech_lengths, labels, shuffle=shuffle)
            logits = nn.utils.rnn.pad_sequence(logits, batch_first=True)
            binary_preds = [self.recover_y_from_powerlabel(logit, n_speaker) for logit, n_speaker in zip(logits, n_speakers)]
            return loss, binary_preds, e2e_ola_labels
        elif self.config.stage == 2:
            soap_speech, soap_speech_lengths = self.frontend(speech, speech_lengths)
            profiles, profiles_len = self.select_non_overlap(soap_speech, soap_speech_lengths, labels)
            soap_speech, soap_speech_lengths = soap_speech.to(speech.device), soap_speech_lengths.to(speech_lengths.device)
            profiles, profiles_len = profiles.to(soap_speech.device), profiles_len.to(soap_speech_lengths.device)
            
            b_p, n_p, t_p, d_p = profiles.shape
            profiles = profiles.view(b_p * n_p, t_p, d_p)
            
            profiles, _, _ = self.profile_extractor(profiles, profiles_len)
            profiles = profiles.view(b_p, n_p, -1)
            profiles = torch.nan_to_num(profiles, nan=0.)

            soap_loss, logits, soap_labels = self.soap(soap_speech, soap_speech_lengths, profiles, labels, label_lens)
            logits = nn.utils.rnn.pad_sequence(logits, batch_first=True)
            binary_preds = [self.recover_y_from_powerlabel(logit, n_speaker) for logit, n_speaker in zip(logits, n_speakers)]

            return soap_loss, binary_preds, soap_labels
        else:
            raise ValueError('config.stage must be either 1 or 2')
    
    def estimate_sequential(self, speech, speech_lengths, n_speakers=None, threshold=None):
        '''
            Args:
                speech: (B, audio_len) torch.Tensor
                speech_lengths: (B,) torch.Tensor
                n_speakers: (B,) [None, int, torch.Tensor]
                threshold: int
            Return:
                binary_labels: list[torch.Tensor]; len = batch_size; item's shape (T, N_pred_speaker)
        '''
        if n_speakers is not None:
            if isinstance(n_speakers, int):
                n_speakers = torch.full((speech.size(0),), n_speakers)
            elif isinstance(n_speakers, torch.Tensor):
                shape_msg = f"Expect number of n_speakers's dimension = 1 but got {len(n_speakers.shape)}"
                assert len(n_speakers.shape) == 1, shape_msg
                batch_size_msg = f"n_speakers's batch size {n_speakers.size(0)} does not match speech's batch size {speech.size(0)}"
                assert n_speakers.size(0) == speech.size(0), batch_size_msg

        e2e_ola_speech, e2e_ola_speech_lengths = self.context_frontend(speech, speech_lengths)
        predict_probs, pred_n_speakers = self.e2e_ola.estimate_sequential(e2e_ola_speech, e2e_ola_speech_lengths, n_speakers=n_speakers, threshold=threshold)

        # prob2pred
        predict_probs = nn.utils.rnn.pad_sequence(predict_probs, batch_first=True)
        B, _, _ = predict_probs.shape
        y_pred = torch.stack([self.recover_y_from_powerlabel(predict_probs[b_idx,:,:], self.config.max_n_speaker) for b_idx in range(B)])
        y_pred = y_pred.repeat_interleave(self.context_frontend.subsampling, dim=1)
        shift_back = self.config.context_frontend.subsampling // 2
        y_pred = F.pad(y_pred[:, shift_back:, :], pad=(0,0,0,shift_back), mode='replicate')

        # overlap-aware post-processing
        soap_speech, soap_speech_lengths = self.frontend(speech, speech_lengths)
        profiles, profiles_len = self.select_non_overlap(soap_speech, e2e_ola_speech_lengths, y_pred)
        b_p, n_p, t_p, d_p = profiles.shape
        profiles = profiles.view(b_p * n_p, t_p, d_p)
        profiles, _, _ = self.profile_extractor(profiles, profiles_len)
        profiles = profiles.view(b_p, n_p, -1)
        profiles = torch.nan_to_num(profiles, nan=0.)

        logits = self.soap.estimate_sequential(soap_speech, soap_speech_lengths, profiles)
        binary_labels = [self.recover_y_from_powerlabel(logit, n_speaker) for logit, n_speaker in zip(logits, pred_n_speakers)]

        return binary_labels


    def select_non_overlap(self, soap_speech, soap_speech_lengths, y_pred):
        '''
            Args:
                soap_speech: (B, non_subsampled_frame_len, F)
                soap_speech_lengths: (B,)
                y_pred: (B, non_subsampled_frame_len, N_speaker)
            Return:
                non_overlap_speech: (B, N, T, F)
                non_overlap_speech_len: (B*N,)
        '''
        max_length = 8 # For padding in case of not having non-overlap segments
        non_overlap_speech = []
        non_overlap_speech_length = []

        for b_idx in range(y_pred.size(0)):
            batch_segments = []
            sample_length = soap_speech_lengths[b_idx].to(torch.long).item()
            y_pred_sum = y_pred[b_idx, :sample_length].sum(dim=1) == 1
            for n_idx in range(y_pred.size(-1)):
                voice_segment_idx = torch.where(
                    (y_pred[b_idx, :sample_length, n_idx] == 1) & y_pred_sum
                )[0].to('cpu')
                segment_length = voice_segment_idx.size(0)
                if max_length < segment_length:
                    max_length = segment_length
                segment = soap_speech[b_idx, voice_segment_idx, :]
                batch_segments.append(segment)
                non_overlap_speech_length.append(segment_length)

            non_overlap_speech.append(batch_segments)

        padded_non_overlap_speech = []

        for batch in non_overlap_speech:
            batch_segments = []
            for segment in batch:
                if segment.size(0) == 0:
                    segment = torch.zeros(1, segment.size(1))
                shortage = max_length - segment.size(0)
                segment_tmp = F.pad(segment.unsqueeze(0), (0, 0, 0, shortage), value=0)
                batch_segments.append(segment_tmp)

            padded_non_overlap_speech.append(torch.vstack(batch_segments))

        return torch.stack(padded_non_overlap_speech), torch.as_tensor(non_overlap_speech_length)


    def recover_y_from_powerlabel(self, logit, n_speaker):
        pred = torch.argmax(torch.softmax(logit, dim=-1), dim=-1)
        oov_index = torch.where(pred == self.mapping_dict['oov'])[0]
        for i in oov_index:
            if i > 0:
                pred[i] = pred[i - 1]
            else:
                pred[i] = 0
        pred = [self.inv_mapping_func(i) for i in pred]
        decisions = [bin(num)[2:].zfill(self.config.max_n_speaker)[::-1] for num in pred]
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

    def freeze_modules(self, freeze_list):
        for module in freeze_list:
            module_path = module.split('.')
            tmp = self
            for mp in module_path:
                tmp = tmp.__getattr__(mp)
            for p in tmp.parameters():
                p.requires_grad = False

    def load_modules(self, load_list):
        print('Loading weight from:')
        print(f'{load_list}')
        for module in load_list:
            if module != 'self':
                module_path = module.split('.')
            else:
                module_path = []
            tmp = self
            for mp in module_path:
                tmp = tmp.__getattr__(mp)
            model_weight = torch.load(load_list[module])
            tmp.load_state_dict(model_weight)
            print(f'Loaded {module}')

    @staticmethod
    def calc_diarization_error(decisions, label, label_delay=0):
        label = label[:len(label) - label_delay, ...]
        n_ref = torch.sum(label, dim=-1)
        n_sys = torch.sum(decisions, dim=-1)
        res = {}
        res['speech_scored'] = torch.sum(n_ref > 0)
        res['speech_miss'] = torch.sum((n_ref > 0) & (n_sys == 0))
        res['speech_falarm'] = torch.sum((n_ref == 0) & (n_sys > 0))
        res['speaker_scored'] = torch.sum(n_ref)
        res['speaker_miss'] = torch.sum(torch.max(n_ref - n_sys, torch.zeros_like(n_ref)))
        res['speaker_falarm'] = torch.sum(torch.max(n_sys - n_ref, torch.zeros_like(n_ref)))
        n_map = torch.sum(((label == 1) & (decisions == 1)), dim=-1).to(torch.float32)
        res['speaker_error'] = torch.sum(torch.min(n_ref, n_sys) - n_map)
        res['correct'] = torch.sum(label == decisions) / label.shape[1]
        res['diarization_error'] = (res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
        res['DER'] = res['diarization_error'] / res['speaker_scored']
        res['frames'] = len(label)
        return res
