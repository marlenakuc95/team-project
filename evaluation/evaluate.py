import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from NERModel import NERModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logging.basicConfig(
    format='%(levelname)s %(asctime)s %(pathname)s: %(message)s',
    # level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

config = {
    'batch_size': 6,
    'num_workers': 0,
    'optimizer': torch.optim.AdamW,
    'lr': 1e-2,
    'weight_decay': 1e-3,
    'hidden_dropout_prob': 0.5,
}

wandb_logger = WandbLogger(save_dir=str(Path(__file__).absolute().parent.joinpath('wandb')), project="adapter-ernie")

# train model
trainer = pl.Trainer(
    logger=wandb_logger,
    gpus=[0, ],  # torch.cuda.device_count(),
    max_epochs=1,
    num_sanity_val_steps=0
)

if __name__ == "__main__":
    logging.info('Starting training')
    model = NERModel(config=config, adapter=True)
    trainer.fit(model)
