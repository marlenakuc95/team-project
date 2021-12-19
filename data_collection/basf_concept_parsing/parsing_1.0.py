import pathlib
import json
import base64

ENCODING = 'UTF-16BE'

# Path to folder
path = pathlib.Path(__file__).parent.parent.joinpath('basf_test_data', 'AR106193A1')

# List json files in a folder
json_files = []
for file in path.iterdir():
    json_files.append(file)

# Load first json
with open(str(json_files[0])) as f:
    data_1 = json.load(f)

# Load second json
with open(str(json_files[1])) as f:
    data_2 = json.load(f)

# Encode textual data from first json
encoded_text = data_1['text']
decoded_text = base64.b64decode(encoded_text).decode("UTF-16BE")


# Get sentence ID and corresponding text
def get_sentence_with_id():
    """
    :return: Dictionary {sentence_id : text}
    """
    annot = data_2['doc']['mparts'][0]['semantic_enrichment']['text_structure']['sentences']
    dct = {}
    for ann in annot:
        sentence_id = ann['sentence_id']
        sentence_text = decoded_text[ann['start']:ann['end']]
        dct[sentence_id] = sentence_text

    return dct


## Test function
sentences = get_sentence_with_id()
print(sentences['s_26'])  # Get text of the sentence with id = s_26


# Get tags with sentences
def get_tags_with_text():
    """
    :return: List of dictionaries [{tag_name: text}]
    """
    doc_tags = []
    tags = data_1['mparts'][0]['tags']
    for tag in tags:
        tag_name = tag['tag']
        tag_text = decoded_text[tag['start']:tag['end']]
        dct = {tag_name: tag_text}
        doc_tags.append(dct)
    return doc_tags


## Test function
tags = get_tags_with_text()
print(tags[2])
