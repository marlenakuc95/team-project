from transformers import AutoTokenizer, AutoModel
from transformers.adapters.configuration import AdapterConfig

BLURB_URI = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# load BLURB tokenizer
tokenizer = AutoTokenizer.from_pretrained(BLURB_URI)

# load BLURB pre-trained model
model = AutoModel.from_pretrained(BLURB_URI)


# Example code for adding adapter
def get_adapter_args():
    return AdapterConfig.load(
        config="pfeiffer",
        non_linearity="relu",
        reduction_factor=16,
    )


adapter_config = get_adapter_args()
model.add_adapter('dEA', config=adapter_config)
model.train_adapter('dEA')
