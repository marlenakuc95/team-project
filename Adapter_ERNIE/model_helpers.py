import pandas as pd
import torch
import numpy as np


# Input text and the tokenizer, get the input ids, attentionmask and start_locations of each token (offsets)
def get_ids_masks_offsets(text, tokenizer):
    encoding = tokenizer(text, padding=True, return_offsets_mapping=True)
    inputids = encoding['input_ids']
    attentionmask = encoding['attention_mask']
    start_locs = [start for (start, end) in encoding['offset_mapping']]
    return inputids, attentionmask, start_locs


def get_token_rep(inputids, attentionmask, model):
    output = model(inputids, attentionmask)
    return output.last_hidden_state


# Input path to receive gan embeddings as a dataframe
def load_gan_embeddings(path):
    return pd.read_csv(path, header=None, nrows=500, index_col=0)


######

# Input annotations path and receive a dataframe of annotations
def load_annotations(path):
    annotations = pd.read_csv(path)
    annotations.set_index("document", inplace=True)
    return annotations


# Input annotations dataframe (coming from load_annotations) and document id, receive annotations of the
# specified document as a list
# Brings annotations of the specified document id
def bring_annotations(annotations_df: pd.DataFrame, doc_id):
    df = annotations_df.loc[doc_id, ["CUI", "start", "length"]]
    annot_list = df.values.tolist()
    return annot_list


# Bring annotations of this specific cui.
# We input output of bring_annotations
# we will use this function's output as an input to
# make_matrix1_row_per_cui function as the second argument

def annotations_of_this_cui(cui, annotations_list):
    annotations_of_this_cui = [[cui_of_list, start, length] for [cui_of_list, start, length] in annotations_list if
                               cui_of_list == cui]
    return annotations_of_this_cui


# Brings all the alignments as a set for the specific document_id.
# We will know how many different alignments we have
def all_alignments_set(annotations_df: pd.DataFrame, doc_id):
    df = annotations_df.loc[doc_id, ["CUI"]]
    alignments_set = set([cui[0] for cui in df.values.tolist()])
    return alignments_set


#######


# This function will make each alignment row per cui (cui * all_tokens)
# It will return rows like ["C1979842",[-1,-1,-1,1,.......,-1]]
# We will use each row to make our dataframe
# Second argument comes from annotations_of_this_cui functional
# Third argument comes from get_ids_masks_offsets function's 3rd output
def make_matrix1_row_per_cui(cui, annotations_of_this_cui, start_locs_of_tokens):
    cui_alignments = []
    for i in range(len(start_locs_of_tokens)):
        for annotation in annotations_of_this_cui:
            if start_locs_of_tokens[i] == annotation[1]:
                cui_alignments.append(1)
            else:
                cui_alignments.append(-1)
    return [cui, cui_alignments]


# This function will take the make_matrix1_row_per_cui functions outputs (as many as there are alignments)
# It will return a dataframe of all the alignment matches
# This is similar to matrix1 we discussed about
# Outputs of the make_matrix1_row_per_cui functions should be input to this function to get the dataframe
def make_df_from_matrix1_rows(list_of_matrix1_rows):
    data = {}
    for row in list_of_matrix1_rows:
        data[row[0]] = row[1]
    # Creates pandas DataFrame.
    df = pd.DataFrame(data, index=[i for i in range(1, len(row[1]) + 1)])

    return df


#####

# This function will take embeddings and cui as input
# It will look up that specific cui's embedding and bring it as a series
# We should apply this function to all cuis that we get from all_alignments_set function
def make_matrix2_row_from_cui_embeds(embeddings, cui):
    row = embeddings.loc[cui, :]
    return [cui, list(row)]


def make_df_from_matrix2_rows(list_of_matrix1_rows):
    data = {}
    for row in list_of_matrix1_rows:
        data[row[0]] = row[1]
    print(data)
    # Creates pandas DataFrame.
    df = pd.DataFrame(data, index=[i for i in range(1, len(row[1]) + 1)])
    return df.T
