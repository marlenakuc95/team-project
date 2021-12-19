from ParsedElement import ParsedElement


class ParsedSentence(ParsedElement):
    def __init__(self, contains_mention, document_text, **kwargs):
        super().__init__(document_text, **kwargs)
        self.contains_mention = contains_mention
