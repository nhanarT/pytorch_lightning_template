import os, sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import pytorch_lightning as pl

from training_template.datasets.diarization_dataset import DiarizationDataModule
from training_template.wrappers.diarization_wrapper import DiarizationWrapper


def main(config):
    save_dir = os.path.dirname(config.ckpt)
    test_name = os.path.splitext(os.path.basename(config.dataset.test_csv))[0]
    data = DiarizationDataModule(config.dataset)
    model = DiarizationWrapper(config)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
    )

    test_result = trainer.test(model, data, ckpt_path=config.ckpt)
    with open(os.path.join(save_dir, f'{test_name}.result'), 'w') as f:
        for k, v in test_result[0].items():
            f.write(f'{k} = {v}\n')


# Check if config path exists
def configuration(path):
    if Path(path).exists():
        return path
    raise ValueError(f'{path} is not exist')


# Check if is checkpoint file
def checkpoint_type(path):
    if (os.path.splitext(path)[1] in ['.ckpt']):
        return path
    raise ValueError(f'{path} is not a checkpoint file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')

    parser.add_argument('--config', dest='config', type=configuration, required=True)
    parser.add_argument('--ckpt', dest='ckpt', type=checkpoint_type, required=True)

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.config = args.config
    config.ckpt = args.ckpt

    main(config)
