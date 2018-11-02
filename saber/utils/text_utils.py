"""A collection of helper/utility functions for processing text.
"""
import re

from spacy.tokenizer import Tokenizer

# NERsuite-like tokenization: alnum sequences preserved as single
# tokens, rest are single-character tokens.
# https://github.com/spyysalo/standoff2conll/blob/master/common.py
INFIX_RE = re.compile(r'''([0-9a-zA-Z]+|[^0-9a-zA-Z])''')

# https://spacy.io/usage/linguistic-features#native-tokenizers
def biomedical_tokenizer(nlp):
    """
    Customizes spaCy's tokenizer class for better handling of biomedical text.
    """
    return Tokenizer(nlp.vocab, infix_finditer=INFIX_RE.finditer)
