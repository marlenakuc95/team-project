import logging
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


# TODO: Parametrize more (add input length, etc.)

class ErnieDataset(IterableDataset):

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
        self.embedding_table = pd.read_csv(str(path_to_emb), header=None).set_index(0)
        self.data_path = path_to_data
        self.annotations_path = path_to_ann
        self.file_dir_names = os.listdir(self.data_path)

    def __iter__(self):
        for file_dir_idx in range(torch.utils.data.get_worker_info().id, len(self.file_dir_names),
                                  torch.utils.data.get_worker_info().num_workers):
            file_dir = self.data_path.joinpath(self.file_dir_names[file_dir_idx])

            annotations_file_path = str(
                self.annotations_path.joinpath("annotations_" + file_dir.stem + "_parsed").with_suffix(".csv"))
            logging.info(f'Loading annotations file {annotations_file_path}')
            annotation_df = pd.read_csv(annotations_file_path)

            path_list = file_dir.glob('*.txt')
            for path in path_list:
                # Read data
                logging.info(f'Loading text file {path}')
                doc_id = path.stem.split('_parsed')[0]
                self.doc_ids.append(doc_id)

                with open(str(path), encoding="utf-8") as f:
                    input_text = f.read()

                """ TRANSFORMER INPUT"""
                logging.info('Tokenizing text')
                encoding = self.tokenizer(text=input_text, return_offsets_mapping=True, max_length=512,
                                          padding='max_length')
                offsets = encoding["offset_mapping"]
                input_ids = encoding["input_ids"]  # 512 x 1

                """ ALIGNMENT TENSOR"""
                # TODO: Special tokens have offset = 0, implement to ignore
                # Dimensions
                # I - input tokens
                # N_e - number of entities
                # Output: N_e x I (Values: 1 - aligned, 0 - non-aligned, (-1) - irrelevant)

                logging.info('Computing alignment tensor')
                entities = annotation_df[annotation_df['document'] == int(doc_id)]
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
                logging.info('Computing embedding tensor')

                embeddings = torch.zeros((entities_size, self.embedding_table.shape[1]), dtype=torch.float32)
                cuis_with_embeddings = entities['CUI'].isin(self.embedding_table.index)
                embeddings[cuis_with_embeddings.values, :] = torch.tensor(
                    self.embedding_table.loc[entities.loc[cuis_with_embeddings, 'CUI']].values.astype(np.float32))

                logging.info('Yielding tensors')
                yield input_ids, alignments, embeddings
