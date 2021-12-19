import os

import numpy as np
import pandas as pd
from utils import ANNOTATIONS_DIR, CORRECTED_ANNOTATIONS_DIR

for file_name in os.listdir(ANNOTATIONS_DIR):
    df = pd.read_csv(ANNOTATIONS_DIR.joinpath(file_name))
    df = df.sort_values(['document', 'start'], ascending=True)
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    ind = df.index[df['text']== "ABS"].tolist()
    series = pd.Series(df.to_numpy()[ind, 7])
    series_1 = pd.Series(df.to_numpy()[ind, 0])
    d_id = series_1.to_numpy()
    s_np = series.to_numpy()
    index_arr = np.array(ind)
    s_np += 6
    df1 = df.loc[df['document'].isin(d_id)]
    s_array = df1[["document", "start"]].to_numpy()
    id = d_id[0]
    i = 0
    for x in s_array:
        if(id == x[0]):
            x[1] = x[1] - s_np[i]
        else:
            i += 1
            id = d_id[i]
            x[1] = x[1] - s_np[i]

    moved_start = s_array[:, 1]
    df1['moved_start'] = moved_start.tolist()
    df1 = df1.query('moved_start>0')
    df1.to_csv(CORRECTED_ANNOTATIONS_DIR.joinpath(file_name), index=False)