import pandas as pd

from constants import EMBEDDINGS_PATH

file_name = "pubmed21n0018"
doc_id = '547985'


annotation_df = pd.read_csv('/work-ceph/glavas-tp2021/team_project/pubmed_parsing/pubmed/annotations_corrected/'
                            f'annotations_{file_name}_parsed.csv')
cuis = annotation_df[annotation_df['document'] == int(doc_id)]['CUI'].values
embedding_cuis = pd.read_csv(EMBEDDINGS_PATH).iloc[:, 0]

cuis_with_embeddings = pd.Series(list(cuis)).isin(embedding_cuis).sum()
cuis_without_embeddings = len(cuis) - cuis_with_embeddings

print(f'{cuis_without_embeddings:,} out of {len(cuis):,} CUIs contained in document {doc_id} are '
      f'not contained in {EMBEDDINGS_PATH}')
