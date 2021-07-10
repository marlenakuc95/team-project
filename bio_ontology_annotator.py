import json
import sys
import urllib.parse
from pprint import pprint
import pandas as pd
import rdflib
from SPARQLWrapper import SPARQLWrapper, RDFXML, XML
import urllib
import requests

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
text_to_annotate = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in ' \
                   'the bowel and the eye.'

annotations = load_json(resources['links']['annotator'] + f'?text={urllib.parse.quote(text_to_annotate)}')


def describe(uri):
    params = urllib.parse.urlencode({'query': f"""DESCRIBE <{uri}>""", 'apikey': API_KEY})

    return json.loads(
        requests.get(f"http://sparql.bioontology.org/sparql?{params}",
                     headers={'Accept': 'application/json'}).content)


unique_ids = list({annotation['annotatedClass']['@id'] for annotation in annotations})

sparql_descriptions = {id: describe(id) for id in unique_ids}

description_lengths = {key: len(value) for key, value in sparql_descriptions.items()}
description_lengths_df = pd.DataFrame(description_lengths.items(), columns=['uri', 'found'])
description_lengths_df['found'] = description_lengths_df['found'].astype(bool)
found_uris = description_lengths_df[description_lengths_df['found']]['uri']
uris_not_found = description_lengths_df[~description_lengths_df['found']]['uri']

description_lengths = {next(iter(value.keys())): len(next(iter(value.values()))) for value in
                       sparql_descriptions.values() if len(value) > 0}
a = sparql_descriptions['http://purl.obolibrary.org/obo/UBERON_0000019']

parents = load_json(annotations[0]['annotatedClass']['links']['parents'])

instances = load_json(annotations[0]['annotatedClass']['links']['ancestors'])
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

pprint(annotations[1])

ontologies = load_json(resources['links']['ontologies'])
pprint(ontologies[1])
