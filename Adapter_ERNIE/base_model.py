from transformers import AutoTokenizer, AutoModel

BLURB_URI = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

tokenizer = AutoTokenizer.from_pretrained(BLURB_URI)

model = AutoModel.from_pretrained(BLURB_URI)
