import warnings
import torch
from pytorch_lightning import LightningModule
from ..models.eda_e2e import TransformerEDADiarization
from ..utils.diarization import pad_sequence, compute_loss_and_metrics

class Diar_Wrapper(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TransformerEDADiarization(
            in_size = config.dataset.feature_dim*(1+2*config.dataset.context_size),
            **config.model
        )
        self.oom_counter = 0


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def training_step(self, batch, batch_idx):
        features = batch['xs']
        labels = batch['ts']
        features, labels = pad_sequence(features, labels)
        features = torch.stack(features)
        loss = compute_loss_and_metrics(self.model, labels, features, return_metrics=False)
        self.log('train/loss', loss, on_step=True, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        try:
            features = batch['xs']
            labels = batch['ts']
            features, labels = pad_sequence(features, labels)
            features = torch.stack(features)
            loss, acum_val_metrics = compute_loss_and_metrics(self.model, labels, features, return_metrics=True)
            self.log('val/loss', loss, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=True)
            acum_val_metrics = {f'val/{k}': torch.Tensor(v).mean() for k, v in acum_val_metrics.items()}
            self.log_dict(acum_val_metrics, on_step=False, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=False)
        except torch.cuda.OutOfMemoryError as err:
            self.logger.experiment.add_text('OOM Err:', str(err))


    def test_step(self, batch, batch_idx):
        try:
            features = batch['xs']
            labels = batch['ts']
            features, labels = pad_sequence(features, labels)
            features = torch.stack(features)
            loss, acum_test_metrics = compute_loss_and_metrics(self.model, labels, features, return_metrics=True)
            acum_test_metrics = {k: torch.Tensor(v).mean() for k, v in acum_test_metrics.items()}
            self.log_dict(acum_test_metrics, on_step=False, on_epoch=True, batch_size=features.size(0), logger=True, prog_bar=False)
        except torch.cuda.OutOfMemoryError as err:
            self.oom_counter += 1
            self.log('OOM_counter', self.oom_counter, on_step=True, logger=False, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        return optimizer  
