import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

root_dir = Path(__file__).parent.parent
adapter_ernie_dir = root_dir / 'Adapter_ERNIE'
sys.path.append(str(root_dir.absolute()))
sys.path.append(str(adapter_ernie_dir.absolute()))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from AdapterErnie import AdapterErnie

logging.basicConfig(
    format='%(levelname)s %(asctime)s %(pathname)s: %(message)s',
    # level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

MIN_DELTA = 0.01  # minimum delta in validation loss for early stopping
PATIENCE = 3  # number of consecutive epochs with validation loss < MIN_DELTA after which to stop early

config = {
    'adapter_type': 'pfeiffer',
    'adapter_non_linearity': 'relu',
    'adapter_reduction_factor': 16,
    'batch_size_train': 16,
    'num_workers': 8,
    # 'num_workers': 1,
    'optimizer': torch.optim.AdamW,
    'lr': 1e-2,
    'weight_decay': 1e-3,
    'dropout_prob': 0.5,
}

model = AdapterErnie(config=config)

early_stopping = EarlyStopping(
    monitor='train_loss',
    min_delta=MIN_DELTA,
    patience=PATIENCE)

wandb_logger = WandbLogger(save_dir=str((adapter_ernie_dir / 'wandb').absolute()), project="adapter-ernie")

# train model
trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[early_stopping],
    gpus='0',
    # gpus=0,
)

if __name__ == "__main__":
    logging.info('Starting training')
    trainer.fit(model)
