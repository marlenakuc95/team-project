import re

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModel, AutoTokenizer
from transformers.adapters.configuration import AdapterConfig

from DeaHead import DeaHead
from utils import DATA_DIR

DEA_NAME = 'dEA'
BLURB_URI = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


class AdapterErnie(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # todo: add actual train set here
        with open(DATA_DIR.joinpath('sample.txt'), 'r') as in_file:
            self.train_set = re.findall(r'AB - (.*)', re.sub(r' +|\n ', ' ', in_file.read()))

        self.batch_size_train = config['batch_size_train']
        self.num_workers = config['num_workers']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        self.tokenizer = AutoTokenizer.from_pretrained(BLURB_URI)
        self.base_model = AutoModel.from_pretrained(BLURB_URI)

        # add adapter to base model
        adapter_config = AdapterConfig.load(
            config=config['adapter_type'],
            non_linearity=config['adapter_non_linearity'],
            reduction_factor=config['adapter_reduction_factor'],
        )
        self.base_model.add_adapter(DEA_NAME, config=adapter_config)
        self.base_model.train_adapter(DEA_NAME)

        self.dea_head = DeaHead(config)

        self.optimizer = config['optimizer']

    def forward(self, x):
        token_embeddings = self.base_model(**x)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size_train,
                          collate_fn=self.collate,
                          shuffle=True, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        token_embeddings = self(batch)
        return 0

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def collate(self, x):
        return self.tokenizer(x, return_tensors="pt", padding=True, max_length=512, truncation=True)
