import base64
import json
import pathlib
from textwrap import wrap
from typing import List

import numpy as np
from ParsedDocument import ParsedDocument

#%%

# Path to folder
test_data_path = pathlib.Path(__file__).parent.parent.joinpath('basf_test_data')


#%%
# Where in the documents are the meaningful texts?
def get_concepts(document_directory: pathlib.Path):
    """
    Gets positions of all concepts in a document, indexed by the section the concept appears in
    :param document_directory: directory containing the docnorm and semantic annotation files
    :return: dictionary {section: [[start, end]]}
    """
    with open(next(f for f in document_directory.iterdir() if f.name.startswith('oc_semantic_annotation'))) as f:
        data_2 = json.load(f)

    concept_dict = {}
    for mention in data_2['doc']['mparts'][1]['semantic_enrichment']['mentions']:
        if mention['section'] in concept_dict.keys():
            concept_dict[mention['section']].append([mention['start'], mention['end']])
        else:
            concept_dict[mention['section']] = [[mention['start'], mention['end']]]

    return concept_dict


concepts = []
for document_directory in test_data_path.iterdir():
    if document_directory.is_dir():
        concepts.append(get_concepts(document_directory))

# compute tensor [document x section x concept x start/end] with concept padding from dictionaries
max_concepts_per_document_and_section = max(
    len(mentions) for concept_dict in concepts for mentions in concept_dict.values())
sections_with_concepts = list({section for concept_dict in concepts for section in concept_dict.keys()})
concept_tensor = []
for concept_dict in concepts:
    doc_row = []
    for section in sections_with_concepts:
        if section in concept_dict.keys():
            doc_row.append(
                concept_dict[section] +
                ((max_concepts_per_document_and_section - len(concept_dict[section])) * [[np.NaN, np.NaN]])
            )
        else:
            doc_row.append(max_concepts_per_document_and_section * [[np.NaN, np.NaN]])
    concept_tensor.append(doc_row)
concept_tensor = np.array(concept_tensor)
concept_counts = {s: np.count_nonzero(~np.isnan(concept_tensor[:, i, :, 0])) for i, s in
                  enumerate(sections_with_concepts)}

print(f'There are {", ".join([f"{v} concepts in section {k}" for k, v in concept_counts.items()])}.')


#%%
# What tag names are there in the tags?
def get_tag_names(document_directory: pathlib.Path):
    """
    Gets names of all tags appearing in the documents
    :param document_directory: directory containing the docnorm and semantic annotation files
    :return: set of all tag names
    """
    with open(next(f for f in document_directory.iterdir() if f.name.startswith('oc_docnorm'))) as f:
        data_1 = json.load(f)
    tags = {tag['tag'] for tag in data_1['mparts'][0]['tags']}
    return tags


all_tag_names = set()
for document_directory in test_data_path.iterdir():
    if document_directory.is_dir():
        all_tag_names.update(get_tag_names(document_directory))

tag_names_array = np.array(list(all_tag_names))
print('\n'.join(
    [f'Documents contain following tag names:'] +
    wrap(', '.join(np.sort(tag_names_array)), 100))
)

#%%
# How do we find the correct texts?
text_tag_candidate_dict = {section: tag_names_array[np.char.startswith(tag_names_array, section[:-1])] for section in
                           sections_with_concepts}
text_tag_candidates = [tag for tag_list in text_tag_candidate_dict.values() for tag in tag_list]


def get_tag_positions(document_directory: pathlib.Path, tag_names: List):
    """
    gets earliest first and latest last position of each specified tag in specified document
    :param document_directory: directory containing the docnorm and semantic annotation files
    :param tag_names: names of tags for which to get the information
    :return: dictionary {tag: [start, end, n_appearances]}
    """
    with open(next(f for f in document_directory.iterdir() if f.name.startswith('oc_docnorm'))) as f:
        data_1 = json.load(f)

    tag_dict = {}
    for tag in data_1['mparts'][0]['tags']:
        if tag['tag'] in tag_dict.keys():
            tag_dict[tag['tag']].append([tag['start'], tag['end']])
        else:
            tag_dict[tag['tag']] = [[tag['start'], tag['end']]]
    out = []
    for tag_name in tag_names:
        if tag_name in tag_dict.keys():
            if len(tag_dict[tag_name]) > 1:
                tag_positions = np.array(tag_dict[tag_name])
                out.append([tag_positions[:, 0].min(), tag_positions[:, 1].max(), tag_positions.shape[0]])
            else:
                out.append(tag_dict[tag_name][0] + [1])
        else:
            out.append([np.nan, np.nan, 0])
    return out


tag_positions = []
for document_directory in test_data_path.iterdir():
    if document_directory.is_dir():
        tag_positions.append(get_tag_positions(document_directory, text_tag_candidates))
tag_positions = np.array(tag_positions)

# Which tags are the concepts located in?
text_tags = []
for section_i, section in enumerate(sections_with_concepts):
    concept_start = concept_tensor[:, section_i, :, 0]
    concept_end = concept_tensor[:, section_i, :, 1]
    for candidate_i, candidate in enumerate(text_tag_candidates):
        if candidate in text_tag_candidate_dict[section]:
            tag_start = tag_positions[:, candidate_i, 0, np.newaxis]
            tag_end = tag_positions[:, candidate_i, 1, np.newaxis]
            concepts_in_tag = np.count_nonzero((concept_start >= tag_start) & (concept_end <= tag_end))
            mean_tag_length = (tag_end - tag_start).mean()
            if not np.isnan(mean_tag_length):
                text_tags.append(candidate)
            print(f'{concepts_in_tag} out of {concept_counts[section]} concepts reported to be '
                  f'located in section {section} are located in corresponding tags of type {candidate}')
            print(f'The average size of {candidate} sections is {mean_tag_length} characters.')


#%%
# What are the texts in tags that contain concepts?

def get_text_from_positions(document_directory: pathlib.Path, tag_positions):
    """
    extracts the texts at the provided positions from the provided document
    :param document_directory: directory containing the docnorm and semantic annotation files
    :param tag_positions: list of tuples containing the start- end end positions of each text section to extract
    :return: list of texts at the provided positions
    """
    with open(next(f for f in document_directory.iterdir() if f.name.startswith('oc_docnorm'))) as f:
        data_1 = json.load(f)
    encoded_text = data_1['text']
    decoded_text = base64.b64decode(encoded_text).decode("UTF-16BE")
    return [decoded_text[int(tag_start):int(tag_end)] for tag_start, tag_end in tag_positions]


# Get the texts from the tags containing concepts
tag_texts = []
text_tag_ids = [i for i, candidate in enumerate(text_tag_candidates) if candidate in text_tags]
i = 0
for document_directory in test_data_path.iterdir():
    if document_directory.is_dir():
        tag_texts.append(get_text_from_positions(document_directory, tag_positions[i, text_tag_ids, :2]))
        i += 1


#%%
# Which sentences contain concepts?

def get_sentence_positions(document_directory: pathlib.Path):
    """
    gets positions of all sentences in the provided document
    :param document_directory: directory containing the docnorm and semantic annotation files
    :return: list of lists with start and end position of each sentence
    """
    with open(next(f for f in document_directory.iterdir() if f.name.startswith('oc_semantic_annotation'))) as f:
        data_2 = json.load(f)

    sentences = []
    for sentence in data_2['doc']['mparts'][0]['semantic_enrichment']['text_structure']['sentences']:
        sentences.append([sentence['start'], sentence['end']])
    return sentences


# Get positions of all sentences for each document
sentence_positions = []
for document_directory in test_data_path.iterdir():
    if document_directory.is_dir():
        sentence_positions.append(get_sentence_positions(document_directory))
max_sentences_per_document = max(len(sentences) for sentences in sentence_positions)
for i, document in enumerate(sentence_positions):
    sentence_positions[i] = document + (max_sentences_per_document - len(document)) * [[np.nan, np.nan]]
sentence_positions = np.array(sentence_positions)

# Get positions of all concepts for each document
concept_positions = []
for document in concepts:
    document_concepts = []
    for concept_list in document.values():
        document_concepts.extend(concept_list)
    concept_positions.append(document_concepts)
max_concepts_per_document = max(len(concept_list) for concept_list in concept_positions)
concept_positions = np.array(
    [document + ((max_concepts_per_document - len(document)) * [[np.nan, np.nan]]) for document in concept_positions])

# For each sentence in each document, figure out whether it contains a concept
sentences_containing_concepts = []
for i in range(sentence_positions.shape[0]):
    sentences_containing_concepts.append(np.any(
        (sentence_positions[i, :, np.newaxis, 0] <= concept_positions[i, np.newaxis, :, 0]) &
        (sentence_positions[i, :, np.newaxis, 1] >= concept_positions[i, np.newaxis, :, 1]), 1))

# Get the texts of all sentences that contain concepts
parsed_sentences = []
i = 0
for document_directory in test_data_path.iterdir():
    if document_directory.is_dir():
        parsed_sentences.append(get_text_from_positions(
            document_directory, sentence_positions[i, sentences_containing_concepts[i]]))
        i += 1
#%%
# Example usage of ParsedDocument class:
documents = [ParsedDocument(directory) for directory in test_data_path.iterdir() if directory.is_dir()]
