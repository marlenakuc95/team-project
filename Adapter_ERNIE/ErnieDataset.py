import torch
from torch.utils.data import Dataset
import pandas as pd
import pathlib
from transformers import BertTokenizerFast  # to be then deleted


# TODO: Parametrize more (add input length, etc.)

class ErnieDataset(Dataset):

    def __init__(self,
                 path_to_data: pathlib.Path,
                 path_to_emb: pathlib.Path,
                 path_to_ann: pathlib.Path,
                 tokenizer
                 ):

        self.tokenizer = tokenizer
        self.data_folders = list(path_to_data.glob('**/*'))
        self.doc_ids = []
        self.input_texts = []
        self.annotations = []
        self.embedding_table = pd.read_csv(str(path_to_emb), header=None)

        for file_dir in path_to_data.iterdir():
            path_list = file_dir.glob('*.txt')

            for path in path_list:
                # Read data
                doc_id = path.stem.split('_parsed')[0]
                self.doc_ids.append(doc_id)

                with open(str(path), encoding="utf-8") as f:
                    input_text = f.read()
                    self.input_texts.append(input_text)

                # Read annotations and CORRECT OFFSETS!
                df = pd.read_csv(
                    str(path_to_ann.joinpath("annotations_" + file_dir.stem + "_parsed").with_suffix(".csv")))

                for index, rows in df.iterrows():
                    if rows['text'] == 'ABS':
                        starts = int(rows['start'])
                        starts = starts + 6
                df["moved_start"] = df.apply(lambda x: x["start"] - starts, axis=1)

                self.annotations.append(df[df['document'] == int(doc_id)])

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx: int):

        """ TRANSFORMER INPUT"""
        example = self.input_texts[idx]
        encoding = self.tokenizer(text=example, return_offsets_mapping=True, max_length=512, padding='max_length')
        offsets = encoding["offset_mapping"]
        input_ids = encoding["input_ids"]  # 512 x 1
        attention_mask = encoding['attention_mask']

        """ ALIGNMENT TENSOR"""
        # TODO: Special tokens have offset = 0, implement to ignore
        # Dimensions
        # I - input tokens
        # N_e - number of entities
        # Output: N_e x I (Values: 1 - aligned, 0 - non-aligned, (-1) - irrelevant)

        entities = self.annotations[idx]
        entities_pt = torch.tensor(entities["moved_start"].values).unsqueeze(dim=1)
        entities_pt = entities_pt.repeat(1, len(input_ids))

        # Get tensor of tokens from input with their offsets
        entities_size = entities_pt.size()[0]
        offsets_pt = torch.tensor([x for (x, y) in offsets]).repeat(entities_size, 1)

        # Compare offsets for ALL entities, match = 1, no-match = 0 (get local alignment tensor)
        loc_alignment_pt = torch.eq(entities_pt, offsets_pt).long()

        # Get max of columns to get global alignment tensor
        glob_align_pt = torch.max(loc_alignment_pt, 0)[0].repeat(entities_size, 1)

        # Turn all zeros to -1
        loc_alignment_pt = torch.where(loc_alignment_pt == 0, -1, loc_alignment_pt)

        # Compare local alignment with global alignment. If local == -1 and glob == 1, set local alignment to 0.
        alignments = torch.where((loc_alignment_pt == -1) & (glob_align_pt == 1), 0, loc_alignment_pt)

        """ ENTITIES EMBEDDINGS"""
        embedd_subset = self.embedding_table.iloc[:, 0].isin(entities["CUI"].tolist())
        embedd_subset = self.embedding_table[embedd_subset]
        embeddings = []

        for cui in entities["CUI"].tolist():
            # If cui is found, convert df row to tensor
            try:
                em = embedd_subset.loc[cui]
                embeddings.append(em.values)

            # If cui is not found, append tensor with zeroes
            except KeyError:
                embeddings.append([0] * 50)

        # Convert to tensor
        embeddings = torch.tensor(embeddings)

        return input_ids, attention_mask, alignments, embeddings


# Calling the class
ROOT = pathlib.Path(__file__).absolute().parent.parent.joinpath('datasets', 'pubmed')
PUBMED_TR = ROOT.joinpath('parsed_tr')
ANN_PATH = ROOT.joinpath('annotations')
EMB_PATH = ROOT.parent.joinpath('embedds', 'GAN_embeddings.csv')
TOK = BertTokenizerFast.from_pretrained("bert-base-cased")

ErnieDataset(PUBMED_TR, EMB_PATH, ANN_PATH, TOK)
