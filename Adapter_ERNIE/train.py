import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from AdapterErnie import AdapterErnie
from utils import DATA_DIR

MIN_DELTA = 0.01  # minimum delta in validation loss for early stopping
PATIENCE = 3  # number of consecutive epochs with validation loss < MIN_DELTA after which to stop early

config = {
    'adapter_type': 'pfeiffer',
    'adapter_non_linearity': 'relu',
    'adapter_reduction_factor': 16,
    'batch_size_train': 2,
    'num_workers': 1,
    'optimizer': torch.optim.AdamW,
    'lr': 1e-2,
    'weight_decay': 1e-3,
    'dropout_prob': 0.5,
}

model = AdapterErnie(config=config)

# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     min_delta=MIN_DELTA,
#     patience=PATIENCE,
#     verbose=True)
# train model
trainer = pl.Trainer(
    logger=TensorBoardLogger(save_dir=DATA_DIR, name="", version="."),
    # callbacks=[early_stopping],
    gpus=torch.cuda.device_count(),
)
if __name__ == "__main__":
    trainer.fit(model)
