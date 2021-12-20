import logging

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.adapters.configuration import AdapterConfig

from ErnieDataset import ErnieDataset
from constants import BLURB_URI
from utils import TRAINING_DATA_DIR, EMBEDDINGS_DIR, CORRECTED_ANNOTATIONS_DIR

DEA_NAME = 'dEA'
GAN_EMBEDDINGS_DIM = 50


class AdapterErnie(pl.LightningModule):
    def __init__(self):
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
                                      path_to_ann=CORRECTED_ANNOTATIONS_DIR,
                                      tokenizer=AutoTokenizer.from_pretrained(BLURB_URI))
        logging.info(f'Loading embeddings file')
        self.embedding_table = pd.read_csv(str(EMBEDDINGS_DIR.joinpath('GAN_embeddings').with_suffix('.csv')),
                                           header=None).set_index(0)
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
                          collate_fn=self.custom_collate, num_workers=self.num_workers, pin_memory=True)

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

    # DataLoader calls the __getitem__ Dataset function, loads single items and stacks the
    def custom_collate(self, batch):
        """Pad data of variable length
        Args:
         batch: (list of tuples) [(input_ids, attention_mask, alignments, embeddings)].
             input_ids - fixed size
             alignments - variable length
             embeddings - variable length
        """
        logging.info('Collating batch')
        # The size of the input_ids will always be the same fixed to =512.
        # Size of alignment matrix and embeddings will be different, as number of entities found vary between inputs.
        # One of the solution is to pad the matrices to the fixed size in the batch.

        if len(batch) == 1:
            input_ids = batch[0][0].unsqueeze(0)
            alignments = batch[0][2].unsqueeze(0)
            cuis = batch[0][3]
            max_len = alignments.size(0)

        else:
            input_ids, alignments, cuis = zip(
                *[(a, b, c) for (a, b, c) in sorted(batch, key=lambda tup: tup[1].size(0), reverse=True)])

            # Padding with 0 or -1? I guess -1?
            max_len = alignments[0].size(0)
            max_inp_len = 512
            alignments = [
                # Todo: check if padding has to be changed to -1 instead of 0
                torch.cat((al, torch.subtract(torch.zeros(max_len - al.size(0), max_inp_len), 1)), 0).T if al.size(0) != max_len else al.T
                for al in alignments]

        alignments = torch.stack(alignments, 0)
        input_ids = torch.tensor(input_ids)

        """ ENTITIES EMBEDDINGS"""
        logging.info('Computing embedding tensor')
        embeddings = torch.zeros((len(batch), max_len, self.embedding_table.shape[1]), dtype=torch.float32)
        for i, cui_series in enumerate(cuis):
            cui_tensor = torch.zeros((len(cui_series), self.embedding_table.shape[1]), dtype=torch.float32)
            cuis_with_embeddings = cui_series.isin(self.embedding_table.index)
            cui_tensor[cuis_with_embeddings.values] = torch.FloatTensor(
                self.embedding_table.loc[cui_series[cuis_with_embeddings]].values)
            embeddings[i, :len(cui_series), :] = cui_tensor

        logging.info('Loading collated batch')
        return input_ids, alignments, embeddings
