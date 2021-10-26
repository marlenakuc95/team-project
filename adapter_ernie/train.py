import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import sys

root_dir = Path(__file__).parent.parent
adapter_ernie_dir = root_dir.joinpath('Adapter_ERNIE')
sys.path.append(str(root_dir.absolute()))
sys.path.append(str(adapter_ernie_dir.absolute()))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from AdapterErnie import AdapterErnie
from utils import DATA_DIR

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
    'batch_size_train': 64,
    'num_workers': os.cpu_count(),
    'optimizer': torch.optim.AdamW,
    'lr': 1e-2,
    'weight_decay': 1e-3,
    'dropout_prob': 0.5,
}

model = AdapterErnie(config=config)

early_stopping = EarlyStopping(
    monitor='train_loss',
    min_delta=MIN_DELTA,
    patience=PATIENCE,
    verbose=True)
# train model
trainer = pl.Trainer(
    logger=TensorBoardLogger(save_dir=DATA_DIR, name="", version="."),
    callbacks=[early_stopping],
    gpus='1',#torch.cuda.device_count(),
)

if __name__ == "__main__":
    logging.info('Starting training')
    trainer.fit(model)
