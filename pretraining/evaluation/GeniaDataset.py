from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from constants import BLURB_URI

class GeniaDataset(Dataset):
    def __init__(self, split: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(BLURB_URI)
        self.data = load_dataset('jnlpba', split=split)
        self.no_of_tags = self.get_no_of_tags()

    def __len__(self):
        return len(self.data)

    def get_no_of_tags(self):
        # Number of unique tags is Genia dataset
        tags = []
        for item in self.data:
            tags.append(item['ner_tags'])
            tags = [item for sublist in tags for item in sublist]
        return len(set(tags))

    def __getitem__(self, i):
        words = self.data[i]['tokens']
        tags = self.data[i]['ner_tags']
        tokenized_input = self.tokenizer(words, is_split_into_words=True)
        tokenized_input_idx = tokenized_input.word_ids()

        label_ids = []
        start_word_idx = None

        for idx in tokenized_input_idx:
            # Ignore special tokens:
            if idx is None:
                label_ids.append(-100)
            # In case a word is split into two or more tokens extend tags to all of them
            elif idx == start_word_idx:
                label_ids.append(tags[start_word_idx])
            else:
                label_ids.append(tags[idx])
                start_word_idx = idx

        tokenized_input['labels'] = label_ids

        return tokenized_input
