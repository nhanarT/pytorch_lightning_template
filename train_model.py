import os, sys, shutil
import argparse
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from training_template.datasets.diarization_dataset import DiarizationDataModule
from training_template.wrappers.diarization_wrapper import DiarizationWrapper

# TODO:
    # Resume training


class PlainProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def main(config):
    # Checkpoint storage
    checkpoint_root = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_root, exist_ok=True)
    if config.resume is None:
        now = datetime.strftime(datetime.now(), "%d-%m-%Y-%H-%M-%S")
        checkpoint_name = f'checkpoint_{now}'
        checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.copy(config.config, os.path.join(checkpoint_dir, 'config.yml'))
    else:
        checkpoint_dir = os.path.dirname(config.resume)
        checkpoint_name = os.path.basename(checkpoint_dir)

    # Logger setting
    logger = TensorBoardLogger(checkpoint_root, checkpoint_name)

    # Start training
    data = DiarizationDataModule(config.dataset)
    model = DiarizationWrapper(config)
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=[0],
        callbacks=[
            PlainProgressBar(),
            EarlyStopping(
                monitor='val/DER',
                min_delta=0.01,
                patience=3,
                mode='min'
            ),
            LearningRateMonitor('step'),
            ModelCheckpoint(
                monitor='val/loss',
                dirpath=checkpoint_dir,
                filename='best_val_loss-{epoch:03d}-{val/loss:.4f}',
                verbose=True,
                save_last=True,
                save_top_k=5,
                save_weights_only=False,
                mode='min',
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                monitor='val/DER',
                dirpath=checkpoint_dir,
                filename='best_val_DER-{epoch:03d}-{val/DER:.4f}',
                verbose=True,
                save_top_k=5,
                save_weights_only=False,
                mode='min',
                auto_insert_metric_name=False,
            ),
        ],
        max_epochs=200,
        num_sanity_val_steps=5,
    )   

    trainer.fit(model, data)

# Check if config path exists
def configuration(path):
    if Path(path).exists():
        return path
    raise ValueError(f'{path} is not exist')

# Check if is checkpoint file
def checkpoint_type(path):
    if (os.path.splitext(path)[1] in ['.pth', '.pt']):
        return path
    raise ValueError(f'{path} is not a checkpoint file')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Here is the description')

    # parser.add_argument(...) reference at https://docs.python.org/3/library/argparse.html
    parser.add_argument('--config', dest='config', type=configuration, required=True)
    # If resume is not specified then no error but if it is specified and extension is not .pth or .pt then got ValueError
    parser.add_argument('--resume', dest='resume', type=checkpoint_type)

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.config = args.config
    config.resume = args.resume

    main(config)
