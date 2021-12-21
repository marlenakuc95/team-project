import logging
import torch
from torch import nn
import pytorch_lightning as pl
from GeniaDataset import GeniaDataset
from adapter_ernie.AdapterErnie import AdapterErnie
from torch.utils.data import DataLoader
from constants import BLURB_URI, PROJECT_DIR
from transformers import AutoModel
from datasets import load_metric
import torchmetrics
import torch.nn.functional as F


# Load either pre-trained PubMedBERT without knowledge injection (baseline) or Adapter model
def load_model(from_checkpoint=False):
    checkpoint_path = PROJECT_DIR / "adapter_ernie/checkpoints" / "wandb" / "adapter-ernie" / \
                      "22oymqen" / "checkpoints" / "epoch=3-step=176663.ckpt"

    if from_checkpoint:
        return AdapterErnie.load_from_checkpoint(checkpoint_path).base_model
    else:
        return AutoModel.from_pretrained(BLURB_URI)


class NERModel(pl.LightningModule):

    def __init__(self, config, adapter=True):
        super().__init__()

        logging.info('Loading data')
        self.train_data = GeniaDataset("train")
        self.val_data = GeniaDataset("validation")
        self.num_tags = self.train_data.no_of_tags

        logging.info('Initializing NER Model for transfer learning')
        # Model
        if adapter:
            self.model = load_model(from_checkpoint=adapter)
        else:
            self.model = load_model()

        for param in self.model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.classifier = nn.Linear(768, self.num_tags)

        self.loss = nn.CrossEntropyLoss()

        logging.info('Loading hyperparameters')
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.weight_decay = config['weight_decay']

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.train_f1 = torchmetrics.F1(num_classes=self.num_tags,
                                        average="micro")
        self.val_f1 = torchmetrics.F1(num_classes=self.num_tags,
                                      average="micro")
        self.val_precision = torchmetrics.Recall(num_classes=self.num_tags, average="micro")
        self.val_recall = torchmetrics.Precision(num_classes=self.num_tags, average="micro")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids, token_type_ids, attention_mask)
        outputs = self.dropout(outputs[0])
        logits = self.classifier(outputs)
        return logits

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_dataloader

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.forward(input_ids, token_type_ids, attention_mask)

        # Compute active loss so as to not compute loss of paddings
        active_loss = batch['attention_mask'].view(-1) == 1

        active_logits = outputs.view(-1, self.num_tags)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(self.loss.ignore_index).type_as(labels)
        )

        # Only compute loss on actual token predictions
        loss = self.loss(active_logits, active_labels)

        # log training loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.forward(input_ids, token_type_ids, attention_mask)  # (batch_s, input_len, no_class)

        # Compute active loss so as to not compute loss of paddings
        active_loss = batch['attention_mask'].view(-1) == 1

        flatten_logits = outputs.view(-1, self.num_tags)  # (batch_s * input_len, no_classes)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(self.loss.ignore_index).type_as(labels)
        )  # (batch_s * input_len)

        # Only compute loss on actual token predictions
        loss = self.loss(flatten_logits, active_labels)
        self.log("val_loss", loss)

        # log performance
        true_labels, true_predictions = self.get_true_labels(flatten_logits, active_labels)

        self.val_acc.update(true_predictions, true_labels)
        self.val_f1.update(true_predictions, true_labels)
        self.val_precision.update(true_predictions, true_labels)
        self.val_recall.update(true_predictions, true_labels)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        # compute metrics
        val_loss = torch.tensor(validation_step_outputs).mean()
        val_accuracy = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        va_recall = self.val_recall.compute()
        val_precision = self.val_precision.compute()

        # log metrics
        self.log("val_accuracy", val_accuracy)
        self.log("val_loss", val_loss)
        self.log("val_f1", val_f1)
        self.log("val_recall", va_recall)
        self.log("val_precision", val_precision)

        # reset all metrics
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def get_true_labels(outputs, labels):
        """
        Filters out padded elements --> compute metrics only on non-padded tokens
        """
        predictions = torch.argmax(outputs, dim=1)

        mask = labels != -100
        indices = torch.nonzero(mask)

        true_labels = labels[indices]
        true_predictions = predictions[indices]

        return true_labels, true_predictions
