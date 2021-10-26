import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.adapters.configuration import AdapterConfig
import logging
from ErnieDataset import ErnieDataset
from custom_collate import custom_collate
from utils import TRAINING_DATA_DIR, EMBEDDINGS_DIR, ANNOTATIONS_DIR, CORRECTED_ANNOTATIONS_DIR

DEA_NAME = 'dEA'
BLURB_URI = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
GAN_EMBEDDINGS_DIM = 50


class AdapterErnie(pl.LightningModule):
    def __init__(self, config):
        logging.info('Initializing Adapter Ernie Model')
        super().__init__()

        self.batch_size_train = config['batch_size_train']
        self.num_workers = config['num_workers']

        self.optimizer = config['optimizer']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        self.base_model = AutoModel.from_pretrained(BLURB_URI)

        # add adapter to base model
        adapter_config = AdapterConfig.load(
            config=config['adapter_type'],
            non_linearity=config['adapter_non_linearity'],
            reduction_factor=config['adapter_reduction_factor'],
        )
        self.base_model.add_adapter(DEA_NAME, config=adapter_config)
        self.base_model.train_adapter(DEA_NAME)

        token_embedding_dim = self.base_model.config.hidden_size
        self.downward_projection = nn.Linear(in_features=token_embedding_dim,
                                             out_features=GAN_EMBEDDINGS_DIM)
        self.normalization = nn.BatchNorm1d(num_features=token_embedding_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.relu = nn.ReLU()
        self.loss = nn.BCELoss()

        logging.info('Loading training dataset')
        self.train_set = ErnieDataset(path_to_data=TRAINING_DATA_DIR,
                                      path_to_emb=EMBEDDINGS_DIR.joinpath('GAN_embeddings').with_suffix('.csv'),
                                      path_to_ann=CORRECTED_ANNOTATIONS_DIR,
                                      tokenizer=AutoTokenizer.from_pretrained(BLURB_URI))
        logging.info('Dataset loaded')

    def forward(self, tokens, entity_embeddings):
        # compute token embeddings
        token_embeddings = self.base_model(tokens)

        # project token embeddings into entity embedding space
        hidden = self.downward_projection(token_embeddings.last_hidden_state)

        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)

        # compute logits for alignment of each token to each entity appearing in input
        logits = torch.matmul(hidden, entity_embeddings.permute(0, 2, 1))

        # apply logistic function element-wise to obtain probabilities for multi-label classification
        probs = self.sigmoid(logits)

        return probs

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size_train,
                          collate_fn=custom_collate, num_workers=self.num_workers, pin_memory=True)

    def training_step(self, batch, batch_idx):
        tokens, y, entity_embeddings = batch
        y_hat = self(tokens, entity_embeddings)
        # set all tokens without alignment to 0 to ignore them in computation of loss
        mask = y != -1
        loss = self.loss(mask * y_hat, mask * y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
