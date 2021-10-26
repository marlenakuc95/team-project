import sys
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

root_dir = Path(__file__).parent.parent
adapter_ernie_dir = root_dir.joinpath('Adapter_ERNIE')
sys.path.append(str(root_dir.absolute()))
sys.path.append(str(adapter_ernie_dir.absolute()))

from custom_collate import custom_collate
from ErnieDataset import ErnieDataset
from utils import TRAINING_DATA_DIR, EMBEDDINGS_DIR, ANNOTATIONS_DIR

if __name__ == '__main__':
    BLURB_URI = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    train_set = ErnieDataset(path_to_data=TRAINING_DATA_DIR, path_to_emb=EMBEDDINGS_DIR.joinpath('GAN_embeddings.csv'),
                             path_to_ann=ANNOTATIONS_DIR, tokenizer=AutoTokenizer.from_pretrained(BLURB_URI))

    train_dataloader = DataLoader(dataset=train_set, batch_size=2, collate_fn=custom_collate, num_workers=2)
    dataloader_iterator = iter(train_dataloader)
    first_3_batches = [next(dataloader_iterator) for _ in range(3)]
