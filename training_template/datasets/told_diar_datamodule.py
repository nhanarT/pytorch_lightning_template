from math import ceil
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class TOLD_Diar_Dataset(Dataset):
    def __init__(self, pandas_df, config, logger=None, is_train=True):
        self.is_train = is_train
        self.config = config
        self.pandas_df = pandas_df
        self.logger = logger
        self.path_wavs = self.pandas_df.groupby("path_wav")['path_wav'].first().values


    def __len__(self):
        return len(self.path_wavs)

    
    def __getitem__(self, index):
        utt = self.path_wavs[index] 
        group_wav = self.pandas_df[self.pandas_df.path_wav==utt]
        starts = group_wav.start.values 
        ends  = group_wav.end.values 
        speakers = group_wav.speaker

        y, sr = librosa.load(utt, sr=self.config.sr)
        start = 0
        end = y.shape[0] 
        if y.shape[0] > int(self.config.max_time * sr):
            if self.is_train:
                if len(set(speakers)) == 0:
                    print("total speaker = 0",utt)
                length = int(self.config.max_time * sr)
                start = np.random.randint(0, end - length + 1)
                end = start + length 
                y = y[start : end]

                start = np.rint(start / self.config.frame_shift).astype(int)
                end = np.rint(end / self.config.frame_shift).astype(int)
            else:
                y = y[start : end]
                start = np.rint(start / self.config.frame_shift).astype(int)
                end = np.rint(end / self.config.frame_shift).astype(int)

        # Create label
        if (y.shape[0] % self.config.frame_shift == 0):
            n_frames = y.shape[0] // self.config.frame_shift
        else:
            n_frames = ceil(y.shape[0] / self.config.frame_shift)
        T = np.zeros((n_frames, self.config.max_n_speaker), dtype=np.int32)
        spk_index = []
        for (start_id, end_id, speaker_id) in zip(starts, ends, speakers):

            start_frame = np.rint(
                start_id * self.config.sr / self.config.frame_shift).astype(int)
            end_frame = np.rint(
                end_id * self.config.sr / self.config.frame_shift).astype(int)
            rel_start = rel_end = None
            if start <= start_frame and start_frame < end:
                rel_start = start_frame - start
            if start < end_frame and end_frame <= end:
                rel_end = end_frame - start
            if rel_start is not None or rel_end is not None:
                if rel_end is None:
                    rel_end = T.shape[0]
                if rel_start is None:
                    rel_start=0
                if rel_end-rel_start<=0:continue
                if speaker_id not in spk_index:
                    spk_index.append(speaker_id)
                speaker_index=spk_index.index(speaker_id)
                T[rel_start:rel_end, speaker_index] = 1
        n_speaker = len(spk_index) 

        return (torch.from_numpy(y), y.shape[0],
                torch.from_numpy(np.copy(T)).to(torch.float), T.shape[0],
                n_speaker)

    @staticmethod
    def collate_fn(batch):
        tensor_batch = defaultdict(list)
        for y, y_len, label, label_len, n_speaker in batch:
            tensor_batch['inputs'].append(y)
            tensor_batch['input_lens'].append(y_len)
            tensor_batch['labels'].append(label)
            tensor_batch['label_lens'].append(label_len)
            tensor_batch['n_speakers'].append(n_speaker)

        tensor_batch['inputs'] = pad_sequence(tensor_batch['inputs'], batch_first=True)
        tensor_batch['input_lens'] = torch.as_tensor(tensor_batch['input_lens'])
        tensor_batch['labels'] = pad_sequence(tensor_batch['labels'], batch_first=True)
        tensor_batch['label_lens'] = torch.as_tensor(tensor_batch['label_lens'])
        tensor_batch['n_speakers'] = torch.as_tensor(tensor_batch['n_speakers'])

        return tensor_batch


class TOLD_Diar_DataModule(LightningDataModule):
    def __init__(self, config, logger=None):
        '''
            Initialize your variables
            Note: Specify your arguments' datatype for a comprehensive code
        '''
        super().__init__()
        self.config = config
        self.logger = logger


    def setup(self, stage: str):
        train_df = pd.read_csv(self.config.train_csv)
        val_df = pd.read_csv(self.config.val_csv)
        test_df = pd.read_csv(self.config.test_csv)
        self.train_ds = TOLD_Diar_Dataset(train_df, self.config, logger=self.logger, is_train=True)
        self.val_ds = TOLD_Diar_Dataset(val_df, self.config, logger=self.logger, is_train=False)
        self.test_ds = TOLD_Diar_Dataset(test_df, self.config, logger=self.logger, is_train=False)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            persistent_workers=True,
            collate_fn=TOLD_Diar_Dataset.collate_fn
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.workers,
            persistent_workers=True,
            collate_fn=TOLD_Diar_Dataset.collate_fn
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.workers,
            persistent_workers=True,
            collate_fn=TOLD_Diar_Dataset.collate_fn
        )
