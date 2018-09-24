"""A collection of helper/utility functions for processing text.
"""
import re

from spacy.tokenizer import Tokenizer

# NERsuite-like tokenization: alnum sequences preserved as single
# tokens, rest are single-character tokens.
# https://github.com/spyysalo/standoff2conll/blob/master/common.py
INFIX_RE = re.compile(r'''([0-9a-zA-Z]+|[^0-9a-zA-Z])''')

def biomedical_tokenizer(nlp):
    return Tokenizer(nlp.vocab, infix_finditer=INFIX_RE.finditer)
