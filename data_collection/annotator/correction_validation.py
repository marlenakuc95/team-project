import pandas as pd
from transformers import AutoTokenizer

from constants import BLURB_URI

doc_id = '547985'
file_name = "pubmed21n0018"
with open(str(f'/work-ceph/glavas-tp2021/team_project/pubmed_parsing/pubmed/parsed_tr/{file_name}/{doc_id}_parsed.txt'),
          encoding="utf-8") as f:
    input_text = f.read()
offsets = {x for (x, y) in
           AutoTokenizer.from_pretrained(BLURB_URI)(text=input_text, return_offsets_mapping=True, max_length=512,
                                                    padding='max_length')["offset_mapping"]}


annotation_df = pd.read_csv('/work-ceph/glavas-tp2021/team_project/pubmed_parsing/pubmed/annotations_corrected/'
                            f'annotations_{file_name}_parsed.csv')
entities = annotation_df[annotation_df['document'] == int(doc_id)]["moved_start"].values
offsets.intersection(entities)
