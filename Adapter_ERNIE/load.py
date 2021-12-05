from AdapterErnie import AdapterErnie
from constants import PROJECT_DIR

#%%
checkpoint_path = PROJECT_DIR / "Adapter_ERNIE" / "wandb" / "adapter-ernie" /\
                  "22oymqen" / "checkpoints" / "epoch=3-step=176663.ckpt"
model = AdapterErnie.load_from_checkpoint(checkpoint_path)
