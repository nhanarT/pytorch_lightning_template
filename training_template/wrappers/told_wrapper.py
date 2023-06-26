import warnings
from collections import defaultdict
import torch
from pytorch_lightning import LightningModule
from ..models.TOLD import TOLD


class TOLD_Wrapper(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TOLD(config)
        self.oom_counter = 0


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def training_step(self, batch, batch_idx):
        features = batch['inputs']
        feature_lengths = batch['input_lens']
        labels = batch['labels']
        label_lens = batch['label_lens']
        n_speakers = batch['n_speakers']
        loss, _, _  = self.model(features, feature_lengths, labels, label_lens, n_speakers, shuffle=True)
        # loss, _, _ = self.model(features, feature_lengths, labels, label_lens, n_speakers, shuffle=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        try:
            features = batch['inputs']
            feature_lengths = batch['input_lens']
            labels = batch['labels']
            label_lens = batch['label_lens']
            n_speakers = batch['n_speakers']
            loss, binary_preds, labels = self.model(features, feature_lengths, labels, label_lens, n_speakers, shuffle=False)
            self.log('val/loss', loss, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=True)

            val_metrics = defaultdict(list)
            binary_preds = torch.stack(binary_preds)
            labels = torch.stack(labels)
            for pred, label in zip(binary_preds, labels):
                res = self.model.calc_diarization_error(binary_preds, labels)
                for k, v in res.items():
                    val_metrics[k].append(v)
            val_metrics = {f'val/{k}': torch.Tensor(v).mean() for k, v in val_metrics.items()}
            self.log_dict(val_metrics, on_step=False, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=False)

        except torch.cuda.OutOfMemoryError as err:
            self.logger.experiment.add_text('OOM Err:', str(err))


    def test_step(self, batch, batch_idx):
        try:
            features = batch['inputs']
            feature_lengths = batch['input_lens']
            labels = batch['labels']
            label_lens = batch['label_lens']
            n_speakers = batch['n_speakers']
            loss, binary_preds, labels = self.model(features, feature_lengths, labels, label_lens, n_speakers, shuffle=False)
            self.log('test/loss', loss, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=True)
            
            test_metrics = defaultdict(list) 
            binary_preds = torch.stack(binary_preds)
            labels = torch.stack(labels)
            for pred, label in zip(binary_preds, labels):
                res = self.model.calc_diarization_error(binary_preds, labels)
                for k, v in res.items():
                    test_metrics[k].append(v)
            test_metrics = {f'test/{k}': torch.Tensor(v).mean() for k, v in test_metrics.items()}
            self.log_dict(test_metrics, on_step=False, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=False)
            
        except torch.cuda.OutOfMemoryError as err:
            self.logger.experiment.add_text('OOM Err:', str(err))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-9)
        return optimizer  
