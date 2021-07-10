import json
import sys
import urllib.parse
from pprint import pprint
import pandas as pd
import rdflib

import requests

g = rdflib.Graph()
REST_URL = "http://data.bioontology.org"
API_KEY = '598d2acb-d88f-41d8-909e-fe7d79d74dfa'


# PREFIX
# owl: < http: // www.w3.org / 2002 / 07 / owl  #>
#
# SELECT
# DISTINCT ?p ?o
# WHERE
# {
# < http: // www.ebi.ac.uk / efo / EFO_0000756 > ?p ?o.
#     FILTER(isURI(?o))
# }

def load_json(url):
    return json.loads(requests.get(url, headers={'Authorization': f'apikey token={API_KEY}'}).content)


resources = load_json(REST_URL)
a = load_json('https://data.bioontology.org/ontologies/OCHV/classes/http%3A%2F%2Fsbmi.uth.tmc.edu%2Fontology%2Fochv%23C0025202/mappings')
text_to_annotate = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in ' \
                   'the bowel and the eye. '

annotations = load_json(resources['links']['annotator'] + f'?text={urllib.parse.quote(text_to_annotate)}')
parents = load_json(annotations[0]['annotatedClass']['links']['parents'])
#
a = pd.DataFrame([{
    'text': annotation['annotations'][0]['text'],
    'matchType': annotation['annotations'][0]['matchType'],
    'ontology': annotation['annotatedClass']['links']['ontology'],
    'id': annotation['annotatedClass']['@id'],
} for annotation in annotations])
a['id'].nunique()
{annotation['annotatedClass']['@type'] for annotation in annotations}
{len(annotation['annotations']) for annotation in annotations}
{len(annotation['hierarchy']) for annotation in annotations}
{annotation['annotations'][0]['text'] for annotation in annotations}

a = {annotation['annotatedClass']['@id'] for annotation in annotations}

pprint(annotations[1])

ontologies = load_json(resources['links']['ontologies'])
pprint(ontologies[1])
