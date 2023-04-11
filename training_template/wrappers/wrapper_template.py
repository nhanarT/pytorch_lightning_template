import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class WrapperTemplate(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        warnings.warn("You're not overriding WrapperTemplate.__init__")


    def setup(self):
        warnings.warn("You're not overriding WrapperTemplate.setup")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("WrapperTemplate.forward is not implemented")


    def training_step(self, batch, batch_idx):
        warnings.warn("You're not overriding WrapperTemplate.training_step")


    def on_train_epoch_end(self):
        '''
            What will model do after complete a training epoch
        '''
        warnings.warn("You're not overriding WrapperTemplate.on_train_epoch_end")


    def validation_step(self, batch, batch_idx):
        warnings.warn("You're not overriding WrapperTemplate.validation_step")


    def on_validation_epoch_end(self):
        '''
            What will model do after complete a validation epoch
        '''
        warnings.warn("You're not overriding WrapperTemplate.on_validation_epoch_end")


    def test_step(self, batch, batch_idx):
        warnings.warn("You're not overriding WrapperTemplate.test_step")


    def predict_step(self, batch, batch_idx):
        warnings.warn("You're not overriding WrapperTemplate.predict_step")

    
    def configure_optimizers(self):
        '''
            More details, search 'configure_optimizers' at https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
        '''
        raise NotImplementedError("WrapperTemplate.configure_optimizers is not implemented")
