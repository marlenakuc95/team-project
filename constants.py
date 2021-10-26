from pathlib import Path

BLURB_URI = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
PROJECT_DIR = Path(__file__).parent
DATASETS_DIR = PROJECT_DIR.joinpath('datasets')
EMBEDDINGS_PATH = DATASETS_DIR.joinpath('embedds').joinpath('GAN_embeddings.csv')
