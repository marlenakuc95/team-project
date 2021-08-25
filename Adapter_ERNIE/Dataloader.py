from torch.utils.data import Dataset, DataLoader
from model_helpers import *
import pandas as pd


# Dataset is prepared
class ErnieDataset(Dataset):
    def __init__(self, path_to_texts, path_to_gan_embeddings, path_to_annotations, tokenizer, model):
        super(self)
        # TODO: write code that reads raw text data from the parsed pubmed files
        # This should be corrected, Idk in what format the texts are coming, I'm just putting a dataframe
        self.dataset = pd.read_csv(path_to_texts, error_bad_lines=False)

        # This should also come from the text dataset
        text = ""
        doc_id = ""

        # We get the inputids, attentionmask and the start locations of each token
        # First two will be used as an input to BERT model
        # start_locs_of_tokens will be used when we are making the matrices
        inputids, attentionmask, start_locs_of_tokens = get_ids_masks_offsets(text, tokenizer)

        # TODO: This does not belong in the dataset but in the forward() method
        # We get the transformer_reps from the BERT model
        transformer_reps = get_token_rep(inputids, attentionmask, model)

        # TODO: This should be a class field because it will have to be accessed multiple times
        # We get the GAN embeddings
        embeddings = load_gan_embeddings(path_to_gan_embeddings)

        # TODO: same as embeddings should be class field
        # Getting the annotations as a df
        annotations_df = load_annotations(path_to_annotations)

        # Annotations of the document that we are interested
        annotations_of_this_doc = bring_annotations(annotations_df, doc_id)

        # All cuis as a set of this document
        cuis_of_this_doc = all_alignments_set(annotations_df, doc_id)

        # ---------- MAKING THE MATRICES ----------
        matrix1_rows = []
        matrix2_rows = []
        # TODO: not sure about this but the for loop looks inefficient to me, probably we can make this a lot faster
        for cui in cuis_of_this_doc:
            cui_annotations = annotations_of_this_cui(cui, annotations_of_this_doc)
            matrix1_rows.append(make_matrix1_row_per_cui(cui, cui_annotations, start_locs_of_tokens))

            matrix2_rows.append(make_matrix2_row_from_cui_embeds(embeddings, cui))

        self.matrix1 = make_df_from_matrix1_rows(matrix1_rows)
        self.matrix2 = make_df_from_matrix2_rows(matrix2_rows)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, itemID):
        # todo: return the next <max BERT input length> tokens from dataset
        row = self.dataset.iloc[itemID]

        text = row['text']

        # Encoding the text
        encoding = tokenizer(text)
        inputids = torch.LongTensor(encoding['input_ids'])
        attentionmask = torch.LongTensor(encoding['attention_mask'])

        return input_to_base, input_for_dea_head
