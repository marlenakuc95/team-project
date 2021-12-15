import logging

import torch


# DataLoader calls the __getitem__ Dataset function, loads single items and stacks the
def custom_collate(batch):
    """Pad data of variable length
    Args:
     batch: (list of tuples) [(input_ids, attention_mask, alignments, embeddings)].
         input_ids - fixed size
         alignments - variable length
         embeddings - variable length
    """
    logging.info('Collating batch')
    # The size of the input_ids will always be the same fixed to =512.
    # Size of alignment matrix and embeddings will be different, as number of entities found vary between inputs.
    # One of the solution is to pad the matrices to the fixed size in the batch.

    if len(batch) == 1:
        input_ids = batch[0][0].unsqueeze(0)
        alignments = batch[0][2].unsqueeze(0)
        embeddings = batch[0][3].unsqueeze(0)

    else:
        input_ids, alignments, embeddings = zip(
            *[(a, b, c) for (a, b, c) in sorted(batch, key=lambda tup: tup[1].size(0), reverse=True)])

        # Padding with 0 or -1? I guess -1?
        max_len = alignments[0].size(0)
        max_inp_len = 512
        alignments = [
            torch.cat((al, torch.zeros(max_len - al.size(0), max_inp_len)), 0).T if al.size(0) != max_len else al.T
            for al in alignments]

        # Padding with 0?
        embeddings = [
            torch.cat((em, torch.zeros(max_len - em.size(0), em.size(1))), 0) if em.size(0) != max_len else em
            for em in embeddings]

        alignments = torch.stack(alignments, 0)
        embeddings = torch.stack(embeddings, 0)
        input_ids = torch.tensor(input_ids)

    logging.info('Loading collated batch')
    return input_ids, alignments, embeddings