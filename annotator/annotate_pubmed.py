import argparse
import logging
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from annotatorService import AnnotatorService
from utils import ANNOTATOR_INPUT_DIR, ANNOTATIONS_DIR
from constants import DATASETS_DIR

parser = argparse.ArgumentParser()
# required parameters
parser.add_argument(
    '--email',
    default=None,
    type=str,
    required=True,
    help='email address for login to UMLS Terminology Services'
)
parser.add_argument(
    '--api_key',
    default=None,
    type=str,
    required=True,
    help='api key for authentication to MetaMap API'
)

args = parser.parse_args()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# other settings

OUTPUT_COLUMNS = ["document", "type", "score", "name", "CUI", "semantic type list", "trigger", "location", "positions",
                  "dummy"]
TRIGGER_COLUMNS = ['short name', 'location', 'utterance', 'text', 'PoS', 'is_negated']
IRRELEVANT_POS_TAGS = {"real_number", "word_numeral", "adv", "prep", "integer", "det", "conj", "percentage"}
IRRELEVANT_TRIGGER_COLUMNS = ['location', 'utterance', 'is_negated']

PUBMED_POINTERS_PATH = DATASETS_DIR.joinpath('pubmed_pointers.csv')


def get_pubmed_pointers():
    return pd.read_csv(PUBMED_POINTERS_PATH, index_col=0)


annotator_service = AnnotatorService(e_mail=args.email, api_key=args.api_key)
pubmed_pointers = get_pubmed_pointers()

while pubmed_pointers['annotator_status'].notnull().sum() < 50:
    input_file_idx = pubmed_pointers['annotator_status'].notnull().idxmin()
    pubmed_pointers.loc[input_file_idx, 'annotator_status'] = 'processing'
    input_file_name = pubmed_pointers.loc[input_file_idx, "file_name"]
    try:
        pubmed_pointers.to_csv(PUBMED_POINTERS_PATH)
        logging.info(f'Blocked file {input_file_name} for annotating')

        input_text_path = ANNOTATOR_INPUT_DIR.joinpath(f'{input_file_name}_parsed.txt')
        csv_text = annotator_service.annotate_batch(input_text_path)

        # preprocess data
        logging.info('Annotations retrieved, processing annotations response.')
        annotation_df = pd.read_csv(StringIO(csv_text), sep='|', header=None, names=OUTPUT_COLUMNS).drop(
            columns=['dummy', 'score'])

        # Remove abbreviations
        annotation_df = annotation_df[(annotation_df['type'] == 'MMI')].drop(columns='type')

        clean_annotations = annotation_df.copy()

        # explode concepts appearing multiple times in same document
        annotation_df['positions'] = annotation_df['positions'].str.split(';')
        annotation_df['trigger'] = annotation_df['trigger'].str[2:-1].str.split(',"')
        # remove annotations with corrupted position/trigger column
        annotation_df = annotation_df[
            (annotation_df['trigger'].apply(len) == annotation_df['positions'].apply(len))]

        annotation_df = annotation_df.explode(column=['trigger', 'positions'])

        # split trigger into multiple columns
        annotation_df[TRIGGER_COLUMNS] = annotation_df['trigger'].str.extract(r'(.*)"-(.*)-(.*)-"(.*)"-(.*)-([01])')
        annotation_df = annotation_df.drop(columns=IRRELEVANT_TRIGGER_COLUMNS + ['trigger'])

        # remove concepts with irrelevant PoS tags
        annotation_df = annotation_df[~annotation_df['PoS'].isin(IRRELEVANT_POS_TAGS)]

        # explode concepts appearing multiple times in same utterance
        annotation_df.loc[annotation_df['positions'].str.contains(r'\['), 'positions'] = \
            annotation_df.loc[annotation_df['positions'].str.contains(r'\['), 'positions'].str[1:-1].str.split(
                r'\],\[')
        annotation_df = annotation_df.explode('positions')

        # extract start position and length of each mention
        annotation_df['start'] = annotation_df['positions'].str.extract(r'(\d+)/.*').astype(int)
        annotation_df['length'] = annotation_df['positions'].str.extract(r'.*/(\d+)').astype(int)
        annotation_df = annotation_df.drop(columns='positions')

        target_path = ANNOTATIONS_DIR.joinpath(f'annotations_{input_text_path.stem}.csv')
        logging.info(f'Processing finished, saving annotations to {target_path}')
        annotation_df.to_csv(target_path, index=False)

        pubmed_pointers = get_pubmed_pointers()
        pubmed_pointers.loc[input_file_idx, 'annotator_status'] = 'done'
        logging.info(f'Setting file {input_file_name} to done')
        pubmed_pointers.to_csv(PUBMED_POINTERS_PATH)
        pubmed_pointers = get_pubmed_pointers()
    except BaseException as e:
        logging.error(f'An error occurred: {repr(e)}: {e}, unblocking {input_file_name}')
        pubmed_pointers.loc[input_file_idx, 'annotator_status'] = np.nan
        pubmed_pointers.to_csv(PUBMED_POINTERS_PATH)
        if type(e) == KeyboardInterrupt:
            logging.info('Execution interrupted, stopping annotator')
            break
