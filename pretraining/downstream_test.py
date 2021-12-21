from transformers import pipeline, AutoTokenizer, DistilBertForTokenClassification, AdamW, get_scheduler, AutoModelForTokenClassification
import os
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


wd = Path(__file__).parent

bert_path = wd.joinpath("bert_model")
model = AutoModelForTokenClassification.from_pretrained(bert_path)
tokenizer = AutoTokenizer.from_pretrained("fran-martinez/scibert_scivocab_cased_ner_jnlpba")

# Get the NER pipeline with out model and tokenizer
pipe = pipeline('ner', model=model, tokenizer=tokenizer)

#train_ds, test_ds = load_dataset('jnlpba', split=['train', 'validation'])
raw_datasets = load_dataset('jnlpba')


inputs = tokenizer(sentences, padding="max_length", truncation=True)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


print(train_ds[0:10])