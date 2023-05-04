from ..utils.diarization import get_labeledSTFT, splice, transform, subsample, diarization_collate_fn
import numpy as np
import pandas as pd
import librosa
import torch 
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class DiarizationDataSet(Dataset):
    '''
        Not implemented cropping a random segment yet
    '''
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

        Y, T = get_labeledSTFT(
            (y, sr, starts, ends, speakers),
            start,
            end,
            frame_shift=self.config.frame_shift,
            frame_size=self.config.frame_size,
        )

        Y = transform(
            Y, sr, self.config.feature_dim, self.config.input_transform)
        
        Y_spliced = splice(Y, self.config.context_size)
        Y_ss, T_ss = subsample(Y_spliced, T, self.config.subsampling)
        is_spk = []
        for i in range(T_ss.shape[1]):
            if T_ss[:,i].sum() > 0:
                is_spk.append(i)
        if len(is_spk) == 0:
            pass
            # Could cause deadlock
            # self.logger.experiment.add_text('0 speaker', utt)
        else:
            T_ss = T_ss[:,is_spk]
        return torch.from_numpy(np.copy(Y_ss)), torch.from_numpy(np.copy(T_ss))


class DiarizationDataModule(LightningDataModule):
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
        self.train_ds = DiarizationDataSet(train_df, self.config, logger=self.logger, is_train=True)
        self.val_ds = DiarizationDataSet(val_df, self.config, logger=self.logger, is_train=False)
        self.test_ds = DiarizationDataSet(test_df, self.config, logger=self.logger, is_train=False)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            persistent_workers=True,
            collate_fn=diarization_collate_fn
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.workers,
            persistent_workers=True,
            collate_fn=diarization_collate_fn
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.workers,
            persistent_workers=True,
            collate_fn=diarization_collate_fn
        )
