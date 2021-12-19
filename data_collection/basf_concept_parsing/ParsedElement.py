from utils import cached_property


class ParsedElement:
    def __init__(self, document_text, start=0, end=0, **kwargs):
        self.start = start
        self.end = end
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._document_text = document_text

    def __len__(self):
        return self.end - self.start

    @cached_property
    def parsed_text(self):
        # noinspection PyUnresolvedReferences
        return self._document_text[self.start:self.end]



