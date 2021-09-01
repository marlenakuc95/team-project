import argparse
import re
import sys
import urllib.parse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from utils import ANNOTATOR_INPUT_DIR, ANNOTATIONS_DIR, DATASETS_DIR

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

# other settings
DATA = {
    "SKR_API": True,
    # See Batch commands here: https://metamap.nlm.nih.gov/Docs/MM_2016_Usage.pdf
    "Batch_Command": "metamap -N -E -Z 2021AA -V Base",
    "Batch_Env": "",
    "RUN_PROG": "GENERIC_V",
    "Email_Address": args.email,
    "BatchNotes": "SKR Web API test",
    "SilentEmail": True,
}

TICKET_GRANTING_TICKET_URL = 'https://utslogin.nlm.nih.gov/cas/v1/api-key'
TICKET_URL = 'https://utslogin.nlm.nih.gov/cas/v1/tickets'
GENERIC_BATCH_SERVICE_URL = 'https://ii.nlm.nih.gov/cgi-bin/II/UTS_Required/API_batchValidationII.pl'

TITLE_START = 21
TITLE_END = 114

OUTPUT_COLUMNS = ["document", "type", "score", "name", "CUI", "semantic type list", "trigger", "location", "positions",
                  "dummy"]
TRIGGER_COLUMNS = ['short name', 'location', 'utterance', 'text', 'PoS', 'is_negated']
IRRELEVANT_POS_TAGS = {"real_number", "word_numeral", "adv", "prep", "integer", "det", "conj", "percentage"}
IRRELEVANT_TRIGGER_COLUMNS = ['location', 'utterance', 'is_negated']

# we need a session for requests to allow batch service to recognize us when calling api for the second time
session = requests.Session()

pubmed_pointers_path = DATASETS_DIR.joinpath('pubmed_pointers.csv')
pubmed_pointers = pd.read_csv(pubmed_pointers_path, index_col=0)
while pubmed_pointers['annotator_status'].notnull().sum() < 50:
    input_file_idx = pubmed_pointers['annotator_status'].notnull().idxmin()
    pubmed_pointers.loc[input_file_idx, 'annotator_status'] = 'processing'
    pubmed_pointers.to_csv(pubmed_pointers_path)
    input_file_name = pubmed_pointers.loc[input_file_idx, "file_name"]
    print(f'Blocked file {input_file_name} for annotating')

    # get ticket granting ticket
    print(f'Fetching ticket granting ticket from {TICKET_GRANTING_TICKET_URL}')
    ticket_granting_ticket = urllib.parse.urlparse(
        BeautifulSoup(
            session.post(
                url=TICKET_GRANTING_TICKET_URL,
                data={'apikey': args.api_key}
            ).content,
            features='html5lib'
        ).find('form').get('action')
    ).path.split('/')[-1]

    # get service ticket
    service_ticket_url = f'{TICKET_URL}/{ticket_granting_ticket}'
    print(f'Fetching service ticket from {service_ticket_url}')
    service_ticket = BeautifulSoup(
        session.post(
            url=service_ticket_url,
            data={'service': GENERIC_BATCH_SERVICE_URL}
        ).content,
        features='html5lib'
    ).find('body').text

    # make actual api call
    input_text_path = ANNOTATOR_INPUT_DIR.joinpath(f'{input_file_name}_parsed.txt')
    service_url = f'{GENERIC_BATCH_SERVICE_URL}?ticket={service_ticket}'
    print(f'Submitting file {input_text_path} to {service_url}')
    session.post(
        url=service_url,
        files={"UpLoad_File": open(input_text_path, 'r'), },
        data=DATA,
        headers={'Connection': "close"},
        allow_redirects=False,
    )

    # call api twice because it only works this way (same in original java implementation)
    print(f'Calling {service_url} again to retrieve annotations')
    response = session.post(
        url=service_url,
        files={"UpLoad_File": open(input_text_path, 'r'), },
        data=DATA,
    )
    csv_text = re.sub(r'NOT DONE LOOP\n', '', response.text)

    # preprocess data

    print('Annotations retrieved, processing preprocessing annotations response.')
    annotation_df = pd.read_csv(StringIO(csv_text), sep='|', header=None, names=OUTPUT_COLUMNS).drop(
        columns=['dummy', 'score'])

    # Remove abbreviations
    annotation_df = annotation_df[(annotation_df['type'] == 'MMI')].drop(columns='type')

    clean_annotations = annotation_df.copy()

    # explode concepts appearing multiple times in same document
    annotation_df['positions'] = annotation_df['positions'].str.split(';')
    annotation_df['trigger'] = annotation_df['trigger'].str[2:-1].str.split(',"')
    # remove annotations with corrupted position/trigger column
    annotation_df = annotation_df[(annotation_df['trigger'].apply(len) == annotation_df['positions'].apply(len))]

    annotation_df = annotation_df.explode(column=['trigger', 'positions'])

    # split trigger into multiple columns
    annotation_df[TRIGGER_COLUMNS] = annotation_df['trigger'].str.extract(r'(.*)"-(.*)-(.*)-"(.*)"-(.*)-([01])')
    annotation_df = annotation_df.drop(columns=IRRELEVANT_TRIGGER_COLUMNS + ['trigger'])

    # remove concepts with irrelevant PoS tags
    annotation_df = annotation_df[~annotation_df['PoS'].isin(IRRELEVANT_POS_TAGS)]

    # explode concepts appearing multiple times in same utterance
    annotation_df.loc[annotation_df['positions'].str.contains(r'\['), 'positions'] = \
        annotation_df.loc[annotation_df['positions'].str.contains(r'\['), 'positions'].str[1:-1].str.split(r'\],\[')
    annotation_df = annotation_df.explode('positions')

    # extract start position and length of each mention
    annotation_df['start'] = annotation_df['positions'].str.extract(r'(\d+)/.*').astype(int)
    annotation_df['length'] = annotation_df['positions'].str.extract(r'.*/(\d+)').astype(int)
    annotation_df = annotation_df.drop(columns='positions')

    target_path = ANNOTATIONS_DIR.pubmed_pointers_path(f'annotations_{input_text_path.stem}.csv')
    print(f'Preprocessing finished, saving annotations to {target_path}')
    annotation_df.to_csv(target_path, index=False)

    pubmed_pointers = pd.read_csv(pubmed_pointers_path, index_col=0)
    pubmed_pointers.loc[input_file_idx, 'annotator_statur'] = 'done'
    print(f'Setting file {input_file_name} to done')
    pubmed_pointers.to_csv(pubmed_pointers_path)
    pubmed_pointers = pd.read_csv(pubmed_pointers_path, index_col=0)
