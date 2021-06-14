import pathlib
import torch
from torch.utils.data import Dataset

class GeniaDataset(Dataset):
    def __init__(self, tokenizer, data_path: pathlib, block_size=512):
        """

        :param tokenizer: Tokenizer for data
        :param data_path: path to eval or train dataset
        :param block_size: input size, 512 is recommended for ROBERTA
        """

        with open(data_path, encoding="utf-8") as f:
            text = f.read()

        self.examples = []

        tokens = tokenizer.tokenize(text)
        tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
        total_length = (len(tokenized_text) // block_size) * block_size
        for i in range(0, total_length, block_size):  # Truncate in block of block_size
            example = tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size])
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])



