import re

import pandas as pd

import numpy as np

from utils import DATA_DIR, ANNOTATIONS_DIR

INPUT_FILE_NAME = "pubmed21n0001.txt"

# Load UIs in Input file
input_text_path = DATA_DIR.joinpath(INPUT_FILE_NAME)
with open(input_text_path, 'r') as in_file:
    input_text = in_file.read()
UIs_in_text = np.array(re.findall(r'UI {2}- (\d+)\n', input_text), dtype=int)

# Load UIs in dataframe
annotation_df = pd.read_csv(ANNOTATIONS_DIR.joinpath(f'annotations_{input_text_path.stem}.csv'))
UIs_in_df = annotation_df['document'].unique()

# Check that all UIs are contained in dataframe
assert np.isin(UIs_in_df, UIs_in_text).all()
