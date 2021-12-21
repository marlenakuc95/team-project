import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from NERModel import NERModel


logging.basicConfig(
    format='%(levelname)s %(asctime)s %(pathname)s: %(message)s',
    # level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

config = {
    'batch_size': 16,
    'num_workers': os.cpu_count(),
    'optimizer': torch.optim.AdamW,
    'lr': 1e-2,
    'weight_decay': 1e-3,
    'dropout_prob': 0.5,
}

model = NERModel(config=config)

# train model
trainer = pl.Trainer(
    logger=TensorBoardLogger(save_dir=str(Path(__file__).absolute().parent), name="", version=".")
)

if __name__ == "__main__":
    logging.info('Starting training')
    trainer.fit(model)
