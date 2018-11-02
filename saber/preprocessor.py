"""Contains the Preprocessor class, which handles low level NLP tasks such as tokenization and
sentence segmentation.
"""
import copy
import logging
import re
from collections import Counter

import numpy as np
from keras.preprocessing.sequence import pad_sequences

import en_coref_md

from . import constants
from .utils import generic_utils, text_utils

LOGGER = logging.getLogger(__name__)

class Preprocessor(object):
    """A class for processing text data."""
    def __init__(self):
        # load the NeuralCoref model, which is built on top of the Spacy english model
        self.nlp = en_coref_md.load()
        # Load our modified tokenizer, better tokenization of biomedical text
        self.nlp.tokenizer = text_utils.biomedical_tokenizer(self.nlp)

    def transform(self, text, word2idx, char2idx, coref=False, sterilize=True):
        """Returns a dictionary collected from processing `text`, including sentences, offsets,
        coreferent mentions, and integer sequences.

        For the given raw text (`text`), returns a dictionary containing the following:
            - 'text': raw text, with minimal processing
            - 'sentences': a list of lists, contains the tokens in each sentence
            - 'offsets': A list of list of tuples containing the start and end indices of every
                token in 'text'
            - 'word_idx_seq': 2-D numpy array which maps every token in 'text' to a unique integer
                ID, given by the mapping `word2idx`.
            - 'char_idx_seq': 3-D numpy array which maps every character in 'text' to a unique
                integer ID, given by the mapping `char2idx`.

        Args:
            text (str): Raw text.
            word2idx (dict): Mapping from words (keys) to unique integers (values).
            char2idx (dict): Mapping from chars (keys) to unique integers (values).
            coref (bool): True if coreference resolution should be applied to `text`, defaults to
                False.
            sterilize (bool): True if text should be sterilized, defaults to True.

        Returns:
            A dictionary containing the processed raw text, sentences, token offsets, etc.
        """
        text = self.sterilize(text) if sterilize else text
        doc = self.nlp(text)

        if coref:
            # doc._.coref_resolved returns empty string if no coreference found
            text = doc._.coref_resolved if doc._.coref_resolved else text
            doc = self.nlp(text)

        # get sentences and token offsets
        sentences, offsets = self._process_text(doc)

        # map sequences to integer IDs
        word_idx_sequence = self.get_type_idx_sequence(sentences, word2idx, type_='word')
        char_idx_sequence = self.get_type_idx_sequence(sentences, char2idx, type_='char')

        transformed_text = {
            'text': text,
            'sents': sentences,
            'offsets': offsets,
            'word_idx_seq': word_idx_sequence,
            'char_idx_seq': char_idx_sequence,
        }

        return transformed_text

    def _process_text(self, doc):
        """Returns tuple of sentences and character offsets of tokens in SpaCy `doc` object.

        Example:
            >>> preprocessor = Preprocessor()
            >>> text = "A simple example!"
            >>> preprocessor._process_text(text)
            ([['Simple', 'example', '!']], [[(0, 6), (7, 14), (14, 15)]])
        """
        # accumulators
        sentences = []
        offsets = []

        # collect token sequence
        for sent in doc.sents:
            token_seq = []
            token_offset_seq = []
            for token in sent:
                token_seq.append(token.text)
                token_offset_seq.append((token.idx, token.idx + len(token.text)))
            sentences.append(token_seq)
            offsets.append(token_offset_seq)

        return sentences, offsets

    @staticmethod
    def type_to_idx(types, initial_mapping=None):
        """Returns a dictionary which maps each item in `types` to a unique integer ID.

        Given a list `types`, returns a dictionary of length `len(types)`, where the keys are
        elements of `types` and the values are unique integer ids from 0 to `len(types) - 1`.

        Args:
            types (list): A list of unique types (words, characters, or tags)
            initial_mapping (dict): An initial mapping of types to integers. If not None, the
                returned mapping of types to integers will include all key, value pairs in this
                dictionary.

        Returns:
            A mapping from items in `types` to unique integer ids.

        Raises:
            ValueError if the values of `initial_mapping` are not a consecutive series of ints
            from 0 to `len(initial_mapping)`.
        """
        if initial_mapping is None:
            return {e: i for i, e in enumerate(types)}

        else:
            if not generic_utils.is_consecutive(initial_mapping.values()):
                err_msg = ("`initial_mapping.values()` must be a consecutive list of ints from 0 "
                           "to `len(initial_mapping)`")
                LOGGER.error('ValueError: %s', err_msg)
                raise ValueError(err_msg)
            # start mapping from end of initial_mapping
            mapping = copy.deepcopy(initial_mapping)
            offset = max(initial_mapping.values()) + 1
            for type_ in types:
                if type_ not in initial_mapping:
                    mapping[type_] = offset
                    offset += 1
            return mapping

    @staticmethod
    def get_type_idx_sequence(seq, type_to_idx, type_='word'):
        """Maps `seq` to a corresponding sequence of indices using `type_to_idx` map.

        Maps `seq`, which contains a sequence of elements (words, characters, or tags), for each
        sentence in a corpora, to a corresponding sequence where all elements have been mapped to
        indices based on the provided `type_to_idx` map. Sentences are either truncated or
        right-padded to match a length of `constants.MAX_SENT_LEN`, and words (character sequences)
        are truncated or right-padded to match a length of `constants.MAX_CHAR_LEN`.

        Args:
            seq (list): list of lists where each list represents a sentence and each inner list
            contains either words, characters or tags
            type_to_idx (dict): a mapping from unique elements in `seq` to unique integer ids
            type_ (str): one of 'word', 'char', 'tag', specifying that `seq` is a sequence of
                words, characters or tags respectively

        Returns:
            `seq`, where all elements have been mapped to unique integer ids based on `type_to_idx`

        Raises:
            ValueError, if `type_` is not one of 'word', 'char', 'tag'
        """
        if type_ not in ['word', 'char', 'tag']:
            err_msg = "Argument `type_` must be one 'word', 'char' or 'type'"
            LOGGER.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)

        # word type
        if type_ == 'word':
            type_seq = [[type_to_idx.get(x, type_to_idx[constants.UNK]) for x in s] for s in seq]
        # tag type
        elif type_ == 'tag':
            type_seq = [[type_to_idx.get(x) for x in s] for s in seq]
        # char type
        elif type_ == 'char':
            # get index sequence of chars
            type_seq = [[[type_to_idx.get(c, type_to_idx[constants.UNK]) for c in w] for w in s] for
                        s in seq]

            # create a sequence of padded character vectors
            for i, char_seq in enumerate(type_seq):
                type_seq[i] = pad_sequences(maxlen=constants.MAX_CHAR_LEN,
                                            sequences=char_seq,
                                            padding="post",
                                            truncating='post',
                                            value=constants.PAD_VALUE)
        # pad sequences
        type_seq = pad_sequences(maxlen=constants.MAX_SENT_LEN,
                                 sequences=type_seq,
                                 padding='post',
                                 truncating='post',
                                 value=constants.PAD_VALUE)

        return np.asarray(type_seq)

    @staticmethod
    def chunk_entities(seq):
        """Chunks entities in the BIO or BIOES format.

        For a given sequence of entities in the BIO or BIOES format, returns the chunked entities.
        Note that invalid tag sequences will not be returned as chunks.

        Args:
            seq (list): sequence of labels.

        Returns:
            list: list of [chunk_type, chunk_start (inclusive), chunk_end (exclusive)].

        Example:
            >>> seq = ['B-PRGE', 'I-PRGE', 'O', 'B-PRGE']
            >>> chunk_entities(seq)
            [('PRGE', 0, 2), ('PRGE', 3, 4)]
        """
        i = 0
        chunks = []
        seq = seq + ['O']  # add sentinel
        types = [tag.split('-')[-1] for tag in seq]
        while i < len(seq):
            if seq[i].startswith('B'):
                for j in range(i+1, len(seq)):
                    if seq[j].startswith('I') and types[j] == types[i]:
                        continue
                    break
                chunks.append((types[i], i, j))
                i = j
            else:
                i += 1
        return chunks

    @staticmethod
    def replace_rare_tokens(sentences, count=constants.NUM_RARE):
        """
        Replaces rare tokens in `sentences` with a special unknown token.

        Returns `sentences`, a list of list where each inner list is a sentence represented as a
        list of strings (tokens), where all tokens appearing less than `count` number of times have
        been replaced with a special unknown token.

        Args:
            sentences (list): List of lists where each inner list is a sentence
                represented as a list of strings (tokens).
            count (int): Tokens appearing `count` times or less are replaced with unknown token.

        Returns:
            sentences, where all rare tokens have been replaced by a special unknown token
        """
        token_count = Counter()
        # create a token count across all sentences
        for sent in sentences:
            token_count.update(sent)

        # replace rare words with constants.UNK token
        for i, sent in enumerate(sentences):
            for j, token in enumerate(sent):
                if token_count[token] <= count:
                    sentences[i][j] = constants.UNK

        return sentences

    @staticmethod
    def sterilize(text, lower=False):
        """Sterilize input text.

        For given input `text`, removes proceeding and preceding spaces, and replaces spans of
        multiple spaces with a single space. Optionally, lowercases `text`.

        Args:
            text (str): Text to sterilize
            lower (bool): True if text should be lower cased

        Returns:
            `text`, where proceeding/preceding spaces have been removed, spans of multiple spaces
            have been replaced with a single space, and optionally, the text has been lowercased.
        """
        sterilized_text = re.sub(r'\s+', ' ', text.strip())
        sterilized_text = sterilized_text.lower() if lower else sterilized_text

        return sterilized_text
