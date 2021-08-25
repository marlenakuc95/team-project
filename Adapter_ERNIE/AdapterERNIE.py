import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel


# Example for custom head on top of base model taken from
# https://discuss.huggingface.co/t/how-do-i-change-the-classification-head-of-a-model/4720/5
class AERNIE(nn.Module):
    def __init__(self):
        super(AERNIE, self).__init__()

        # This will be our BLURB model with injected adapters
        self.base_model = AutoModel.from_pretrained('bert-base-uncased')

        # define our dEA head
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 2)  # output features from bert is 768 and 2 is ur number of labels

    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        # Define computations with our dEA head
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)

        return outputs

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size_train,
                          collate_fn=self.generate_features,
                          shuffle=True, num_workers=self.num_workers)


model = AERNIE()
next(model.train_dataloader())
model.to('cuda')
