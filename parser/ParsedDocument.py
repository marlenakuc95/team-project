import base64
import json
import pathlib
from typing import Dict, List, Tuple

from ParsedSentence import ParsedElement, ParsedSentence
from utils import cached_property


class ParsedDocument:
    def __init__(self, directory: pathlib.Path):
        self.dir = directory

    @cached_property
    def semantic_annotation(self):
        with open(next(f for f in self.dir.iterdir() if f.name.startswith('oc_semantic_annotation'))) as f:
            return json.load(f)

    @cached_property
    def docnorm(self):
        with open(next(f for f in self.dir.iterdir() if f.name.startswith('oc_docnorm'))) as f:
            return json.load(f)

    @cached_property
    def mention_positions(self) -> Dict[str, List[Tuple[int, int]]]:
        concept_dict = {}
        for mention in self.mentions:
            if mention.section in concept_dict.keys():
                concept_dict[mention.section].append((mention.start, mention.end))
            else:
                concept_dict[mention.section] = [(mention.start, mention.end)]
        return concept_dict

    @cached_property
    def mentions(self):
        return [ParsedElement(self.text, **mention) for mention in
                self.semantic_annotation['doc']['mparts'][1]['semantic_enrichment']['mentions']]

    @cached_property
    def sentence_positions(self) -> List[Tuple[int, int]]:
        """
        gets positions of all sentences in the provided document
        :param document_directory: directory containing the docnorm and semantic annotation files
        :return: list of tuples with start and end position of each sentence
        """
        sentences = []
        for sentence in self.sentences:
            sentences.append((sentence['start'], sentence['end']))
        return sentences

    @cached_property
    def sentences(self):
        """
        List of document's sentences
        """
        sentence_dicts = self.semantic_annotation['doc']['mparts'][0]['semantic_enrichment']['text_structure'][
            'sentences']
        sentences = []
        sentence_idx = 0
        mention_idx = 0
        while sentence_idx <= len(sentence_dicts):
            contains_mention = False
            if mention_idx >= len(self.mentions):
                sentences.extend([ParsedSentence(contains_mention, self.text, **sentence_dict) for sentence_dict in
                                  sentence_dicts[sentence_idx:]])
                sentence_idx = len(sentence_dicts) + 1
            else:
                if self.mentions[mention_idx].end > sentence_dicts[sentence_idx]['start']:
                    if self.mentions[mention_idx].start < sentence_dicts[sentence_idx]['end']:
                        contains_mention = True
                    sentences.append(ParsedSentence(contains_mention, self.text, **sentence_dicts[sentence_idx]))
                    sentence_idx += 1
                else:
                    mention_idx += 1
        return sentences

    @cached_property
    def tags(self):
        """
        gets positions of all tags in the provided document
        :param document_directory: directory containing the docnorm and semantic annotation files
        :return: list of tuples with start and end position of each tag
        """
        return [ParsedElement(self.text, **tag) for tag in self.docnorm['mparts'][0]['tags']]

    @cached_property
    def text(self):
        """
        The document's decoded text
        """
        encoded_text = self.docnorm['text']
        decoded_text = base64.b64decode(encoded_text).decode("UTF-16BE")
        return decoded_text

    def __len__(self):
        return len(self.text)

        pass

    # @cached_property
    # def sentences(self):
