import re

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.adapters.configuration import AdapterConfig

from utils import DATA_DIR

DEA_NAME = 'dEA'
BLURB_URI = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
GAN_EMBEDDINGS_DIM = 50


class AdapterErnie(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # todo: add actual train set here
        with open(DATA_DIR.joinpath('sample.txt'), 'r') as in_file:
            self.train_set = re.findall(r'AB - (.*)', re.sub(r' +|\n ', ' ', in_file.read()))
        self.batch_size_train = config['batch_size_train']
        self.num_workers = config['num_workers']

        self.optimizer = config['optimizer']
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


        token_embedding_dim = self.base_model.config.hidden_size
        self.downward_projection = nn.Linear(in_features=token_embedding_dim,
                                             out_features=GAN_EMBEDDINGS_DIM)
        self.normalization = nn.BatchNorm1d(num_features=token_embedding_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.relu = nn.ReLU()
        self.loss = nn.BCELoss()

    def forward(self, tokens, entity_embeddings):
        # compute token embeddings
        token_embeddings = self.base_model(**tokens)

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
                          collate_fn=self.collate,
                          shuffle=True, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        tokens, entity_embeddings, y = batch
        y_hat = self(tokens, entity_embeddings)
        # set all tokens without alignment to 0 to ignore them in computation of loss
        mask = y != -1
        loss = self.loss(mask * y_hat, mask * y)
        return loss

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def collate(self, x):
        """
        mock implementation of collator, randomly initiates entity alignments and embeddings
        """
        tokens = self.tokenizer(x, return_tensors="pt", padding=True, max_length=512, truncation=True)
        input_length = tokens['input_ids'].shape[1]
        # randomly initiate some entity embeddings for this input
        alignment_prob = 0.3
        n_embeddings = int(0.8 * alignment_prob * input_length)
        entity_embeddings = torch.rand(tokens['input_ids'].shape[0], n_embeddings, GAN_EMBEDDINGS_DIM)
        # randomly pick some tokens to have an alignment
        aligned_tokens = torch.bernoulli(
            (torch.zeros(input_length) + alignment_prob).tile((2, 1, 1))
        ).tile((1, n_embeddings, 1)).permute(0, 2, 1)
        # randomly pick some aligned tokens to be aligned to a certain entity
        align_prob_2 = 0.01
        alignments = torch.bernoulli(aligned_tokens * align_prob_2) - (-aligned_tokens + 1)
        return tokens, entity_embeddings, alignments
