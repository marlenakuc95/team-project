import logging

from data_collection.annotator.annotatorService import AnnotatorService
from utils import ANNOTATOR_INPUT_DIR

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

annotator_service = AnnotatorService(e_mail='l.loos95@gmail.com', api_key='34d7c577-561b-4e81-9d7b-ef12f37537e6')
csv_text = annotator_service.annotate_batch(ANNOTATOR_INPUT_DIR.joinpath('small_sample.txt'))
