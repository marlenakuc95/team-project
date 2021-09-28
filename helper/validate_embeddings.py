import os

import pandas as pd

from utils import ANNOTATIONS_DIR, DATASETS_DIR

EMBEDDINGS_PATH = DATASETS_DIR.joinpath('embedds').joinpath('GAN_embeddings.csv')

cuis = set()

for filename in os.listdir(ANNOTATIONS_DIR):
    cuis.update(set(pd.read_csv(ANNOTATIONS_DIR.joinpath(filename))['CUI'].values))

embeddings = pd.read_csv(EMBEDDINGS_PATH).iloc[:, 0]

cuis_with_embeddings = pd.Series(list(cuis)).isin(embeddings).sum()
cuis_without_embeddings = len(cuis) - cuis_with_embeddings

print(f'{cuis_without_embeddings:,} out of {len(cuis):,} CUIs contained in annotation files in {ANNOTATIONS_DIR} are '
      f'not contained in {EMBEDDINGS_PATH}')
