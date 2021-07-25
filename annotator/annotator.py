import re
import urllib.parse
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils import DATA_DIR

# before running, enter your credentials here:
EMAIL_ADDRESS = ''
API_KEY = ''

# other settings
INPUT_TEXT_PATH = DATA_DIR.joinpath("small_sample.txt")
DATA = {
    "SKR_API": True,
    # See Batch commands here: https://metamap.nlm.nih.gov/Docs/MM_2016_Usage.pdf
    "Batch_Command": "metamap -N -E -Z 2021AA -V Base",
    "Batch_Env": "",
    "RUN_PROG": "GENERIC_V",
    "Email_Address": EMAIL_ADDRESS,
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
RELEVANT_COLUMNS = ["document", "CUI", "start", "length"]

# we need a session for requests to allow batch service to recognize us when calling api for the second time
session = requests.Session()

# get ticket granting ticket
ticket_granting_ticket = urllib.parse.urlparse(
    BeautifulSoup(
        session.post(
            url=TICKET_GRANTING_TICKET_URL,
            data={'apikey': API_KEY}
        ).content,
        features='html5lib'
    ).find('form').get('action')
).path.split('/')[-1]

# get service ticket
service_ticket = BeautifulSoup(
    session.post(
        url=f'{TICKET_URL}/{ticket_granting_ticket}',
        data={'service': GENERIC_BATCH_SERVICE_URL}
    ).content,
    features='html5lib'
).find('body').text

# make actual api call
service_url = f'{GENERIC_BATCH_SERVICE_URL}?ticket={service_ticket}'
session.post(
    url=service_url,
    files={"UpLoad_File": open(INPUT_TEXT_PATH, 'r'), },
    data=DATA,
    headers={'Connection': "close"},
    allow_redirects=False, )

# call api twice because it only works this way (same in original java implementation)
response = session.post(
    url=service_url,
    files={"UpLoad_File": open(INPUT_TEXT_PATH, 'r'), },
    data=DATA)
csv_text = re.sub(r'NOT DONE LOOP\n', '', response.text)

# preprocess data

annotation_df = pd.read_csv(StringIO(csv_text), sep='|', header=None, names=OUTPUT_COLUMNS).drop(
    columns=['dummy', 'score'])

# Remove abbreviations
annotation_df = annotation_df[(annotation_df['type'] == 'MMI')].drop(columns='type')

# explode concepts appearing multiple times in same document
annotation_df['positions'] = annotation_df['positions'].str.split(';')
annotation_df['trigger'] = annotation_df['trigger'].str[2:-1].str.split(',"')
annotation_df = annotation_df.explode(column=['trigger', 'positions'])

# split trigger into multiple columns
annotation_df[TRIGGER_COLUMNS] = annotation_df['trigger'].str.extract(r'(.*)"-(.*)-(.*)-"(.*)"-(.*)-([01])')

# explode concepts appearing multiple times in same utterance
annotation_df.loc[annotation_df['positions'].str.contains(r'\['), 'positions'] = \
    annotation_df.loc[annotation_df['positions'].str.contains(r'\['), 'positions'].str[1:-1].str.split(r'\],\[')
annotation_df = annotation_df.explode('positions')

# extract start position of each mention
annotation_df['start'] = annotation_df['positions'].str.extract(r'(\d+)/.*').astype(int)
annotation_df['length'] = annotation_df['positions'].str.extract(r'.*/(\d+)').astype(int)
annotation_df = annotation_df.drop(columns='trigger')

# keep only relevant columns
relevant_annotations = annotation_df[RELEVANT_COLUMNS]
