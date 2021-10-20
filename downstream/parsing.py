import xml.etree.ElementTree as ET
import pathlib as p
import re
from torch.utils.data import Dataset, DataLoader
import transformers
import torch

ROOT_PATH = p.Path(__file__).absolute().parent


# Update MAX_LEN or TOKENIZER if needed
class Cfg:
    DATA_DIR = ROOT_PATH.joinpath('GENIAcorpus3.02p', 'GENIAcorpus3.02.pos.xml')
    MAX_LEN = 512
    TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


# 2000 abstracts
class GeniaNER_Dataset(Dataset):
    def __init__(self, cfg):
        self.tokens = []
        self.pos_tags = []
        self.cfg = cfg

        tree = ET.parse(str(self.cfg.DATA_DIR))
        root = tree.getroot()

        pattern = re.compile("^[A-Z]")

        # Iterate over articles
        for art in root.iter('article'):
            tags = []
            tokenized_text = []
            abstract = art.find('abstract')
            for sen in abstract.findall('sentence'):
                for word_tag in sen.iter('w'):
                    pos = word_tag.attrib['c']
                    token = word_tag.text

                    # Sometimes POS-tag is a '.', '(' etc. - punctuation - to be deleted'
                    if pattern.match(pos):
                        tags.append(pos)
                        tokenized_text.append(token)

            self.tokens.append(tokenized_text)
            self.pos_tags.append(tags)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        ids = []
        labels = []

        sentence = self.tokens[item]
        pos_tag = self.pos_tags[item]

        for i, word in enumerate(sentence):
            # Encode each single token in a sentence
            inputs = self.cfg.TOKENIZER.encode(
                word,
                add_special_tokens=False
            )
            # Some words will split into multiple tokens, hence POS labels need to be adjusted
            input_len = len(inputs)
            ids += inputs
            labels += [pos_tag[i]] * input_len

        # Adjust for max length
        ids = ids[:self.cfg.MAX_LEN - 2]
        labels = labels[:self.cfg.MAX_LEN - 2]

        # Add special tokens
        ids = [101] + ids + [102]
        labels = [0] + labels + [0]

        # Masks
        mask = torch.ones(len(ids))

        # Pad too short sentences & labels
        padd_len = self.cfg.MAX_LEN - len(ids)
        if padd_len > 0:
            ids = ids + [0] * padd_len
            labels = labels + [0] * padd_len
            mask = torch.cat([mask, torch.zeros(padd_len)], dim=0)

        return {"ids": torch.tensor(ids, dtype=torch.long),
                "labels": labels,
                "mask": mask}


#test = GeniaNER_Dataset(Cfg())
#a = test[0]
