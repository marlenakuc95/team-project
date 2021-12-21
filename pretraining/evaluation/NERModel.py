import logging
import torch
from torch import nn
import pytorch_lightning as pl
from GeniaDataset import GeniaDataset
from adapter_ernie.AdapterErnie import AdapterErnie
from torch.utils.data import DataLoader
from constants import BLURB_URI, PROJECT_DIR
from transformers import AutoModel


# Load either pre-trained PubMedBERT without knowledge injection (baseline) or Adapter model
def load_model(model="bert"):

    if model == "bert" or model == "BERT":

        model_path = PROJECT_DIR / "pretraining" / "bert_model"

        return AutoModel.from_pretrained(model_path)

    elif model == "roberta" or model == "ROBERTA":
        model_path = PROJECT_DIR / "pretraining" / "roberta_model"

        return AutoModel.from_pretrained(model_path)


class NERModel(pl.LightningModule):
    logging.info('Initializing NER Model for transfer learning')

    def __init__(self,
                 config,
                 bert = True):

        super().__init__()

        if bert:
            self.model = load_model(model="bert")
        else:
            self.model = load_model(model = "roberta")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_tags)
        self.loss = nn.CrossEntropyLoss()

        # Hyperparameters
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.weight_decay = config['weight_decay']

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits

    def setup(self, stage=None):
        self.train_data = GeniaDataset("train")
        self.val_data = GeniaDataset("validation")
        self.num_tags = self.train_data.no_of_tags

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloaderr(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        loss = self.loss(outputs.view(-1, self.num_labels), labels.view(-1))

        # log training loss
        self.log_dict({'train_loss': loss}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        loss = self.loss(outputs.view(-1, self.num_labels), labels.view(-1))

        # log performance
        metrics = self.compute_performance_metrics(outputs, labels)
        self.log_dict({'val_loss': loss, 'val_f1': metrics['f1'], 'val_accuracy': metrics['accuracy'],
                       'val_precision': metrics['precision'], 'val_recall': metrics['recall']}, prog_bar=True)

        return loss, outputs, labels


    def compute_performance_metrics(self, outputs, labels):
        predictions = torch.argmax(outputs, dim=2)
        results = self.metric.compute(predictions=predictions, references=labels)

        return {"accuracy": results["overall_accuracy"],
                "f1": results["overall_f1"],
                "precision": results["overall_precision"],
                "recall": results["overall_recall"]}
