import logging
import re
import urllib.parse

import requests
from bs4 import BeautifulSoup

TICKET_GRANTING_TICKET_URL = 'https://utslogin.nlm.nih.gov/cas/v1/api-key'
TICKET_URL = 'https://utslogin.nlm.nih.gov/cas/v1/tickets'
GENERIC_BATCH_SERVICE_URL = 'https://ii.nlm.nih.gov/cgi-bin/II/UTS_Required/API_batchValidationII.pl'


class AnnotatorService:
    def __init__(self, e_mail, api_key):
        # we need a session for requests to allow batch service to recognize us when calling api for the second time
        self.session = requests.Session()
        self.e_mail = e_mail
        self.api_key = api_key

    def annotate_batch(self, batch_path):
        # get ticket granting ticket
        logging.info(f'Fetching ticket granting ticket from {TICKET_GRANTING_TICKET_URL}')
        ticket_granting_ticket = urllib.parse.urlparse(
            BeautifulSoup(
                self.session.post(
                    url=TICKET_GRANTING_TICKET_URL,
                    data={'apikey': self.api_key}
                ).content,
                features='html5lib'
            ).find('form').get('action')
        ).path.split('/')[-1]
        # get service ticket
        service_ticket_url = f'{TICKET_URL}/{ticket_granting_ticket}'
        logging.info(f'Fetching service ticket from {service_ticket_url}')
        service_ticket = BeautifulSoup(
            self.session.post(
                url=service_ticket_url,
                data={'service': GENERIC_BATCH_SERVICE_URL}
            ).content,
            features='html5lib'
        ).find('body').text
        # make actual api call
        service_url = f'{GENERIC_BATCH_SERVICE_URL}?ticket={service_ticket}'
        logging.info(f'Submitting file {batch_path} to {service_url}')
        body = {
            "SKR_API": True,
            # See Batch commands here: https://metamap.nlm.nih.gov/Docs/MM_2016_Usage.pdf
            "Batch_Command": "metamap -N -E -Z 2021AA -V Base",
            "Batch_Env": "",
            "RUN_PROG": "GENERIC_V",
            "Email_Address": self.e_mail,
            "BatchNotes": "SKR Web API test",
            "SilentEmail": True,
        }
        self.session.post(
            url=service_url,
            files={"UpLoad_File": open(batch_path, 'r'), },
            data=body,
            headers={'Connection': "close"},
            allow_redirects=False,
        )
        # call api twice because it only works this way (same in original java implementation)
        logging.info(f'Calling {service_url} again to retrieve annotations')
        response = self.session.post(
            url=service_url,
            files={"UpLoad_File": open(batch_path, 'r'), },
            data=body,
        )
        return re.sub(r'NOT DONE LOOP\n', '', response.text)
