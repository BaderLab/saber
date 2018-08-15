"""Contains the Preprocessor class, which handles low level NLP tasks such as tokenization and
sentence segmentation.
"""
import logging
import re
from collections import Counter

import en_core_web_sm
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from . import constants

class Preprocessor(object):
    """A class for processing text data."""
    # define this at the class level because most methods are static
    log = logging.getLogger(__name__)

    def __init__(self):
        # load Spacy english model (core, small), disable NER pipeline
        self.nlp = en_core_web_sm.load(disable=['ner'])

    def transform(self, text, word2idx, char2idx, sterilize=True):
        """Returns a dictionary collected from processeing `text`, including sentences, offsets,
        and integer sequences.

        For the given raw text (`text`), returns a dictionary containing the following:
            - 'text': raw text, with minimal processing
            - 'sentences': a list of lists, contains the tokens in each sentence
            - 'offsets': A list of list of tuples containing the start and end indices of every
                token in 'text'
            - 'word2idx': 2-D numpy array containing the token index of every token in 'text'.
                Index is chosen based on the mapping `word2idx`
            - 'char2idx': 3-D numpy array containing the character index of
                every character in 'text'. Index is chosen based on the mapping `char2idx`

        Args:
            text (str): raw text
            word2idx (dict): mapping from words (keys) to unique integers (values)
            char2idx (dict): mapping from chars (keys) to unique integers (values)
            sterilize (bool): True if text should be sterilized

        Returns:
            a dictionary containing the processed raw text, sentences, token offsets, etc.
        """
        text = self.sterilize(text) if sterilize else text

        # get sentences and token offsets
        sentences, offsets = self._process_text(text)

        word_idx_sequence = self.get_type_idx_sequence(sentences, word2idx, type_='word')
        char_idx_sequence = self.get_type_idx_sequence(sentences, char2idx, type_='char')

        transformed_text = {
            'text': text,
            'sent': sentences,
            'offsets': offsets,
            'word2idx': word_idx_sequence,
            'char2idx': char_idx_sequence
        }

        return transformed_text

    def _process_text(self, text):
        """Returns sentences and character offsets of tokens in text.

        For the given `text`, uses Spacy to return a two-tuple of the sentences and token
        offsets (relative to their position in `text`) of each token in `text`.

        Args:
            text (str): raw text to process.

        Returns:
            two-tuple, containing the sentences in `text` (as a list of lists) and the token
            offsets for every token in text (relative to their position in `text`, as a list of
            lists).

        Example:
            >>> preprocessor = Preprocessor()
            >>> text = "A simple example!"
            >>> preprocessor._process_text(text)
            ([['Simple', 'example', '!']], [[(0, 6), (7, 14), (14, 15)]])
        """
        doc = self.nlp(text) # process text with spacy

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
    def type_to_idx(types, initial_mapping=None, offset=0):
        """Returns a dictionary of element:index pairs for each element in types.

        Given a list `types`, returns a dictionary of length len(types) + offset, where the keys are
        elements of `types` and the values are unique integer ids.

        Args:
            types (list): A list of unique types (words, characters, or tags)
            initial_mapping (dict): An initial mapping of types to integers. If not None, the
                mapping of types to integers will update this dictionary, with the integer count
                begginning at len(initial_mapping).
            offset (int): Used when computing the mapping. An offset a 1 means we begin computing
                the mapping at 1, useful if we want to use 0 as a padding value. Has no effect if
                initial_mapping is not None.
        Returns:
            a mapping from elements in the `types` to unique integer ids

        Preconditions:
            assumes that all elements in sequence are unique
        """
        if initial_mapping is not None:
            # if a type in initial_mapping already exists in types, remove it
            for type_ in initial_mapping:
                if type_ in types:
                    types.remove(type_)
            mapping = {e: i + len(initial_mapping) for i, e in enumerate(types)}
            mapping.update(initial_mapping)
            return mapping
        # offset accounts for sequence pad
        return {e: i + offset for i, e in enumerate(types)}

    @staticmethod
    def get_type_idx_sequence(seq, type_to_idx, type_='word'):
        """Maps `seq` to a correspoding sequence of indices using `type_to_idx` map.

        Maps `seq`, which contains a sequence of elements (words, characters, or tags), for each
        sentence in a corpora, to a corresponding sequence where all elements have been mapped to
        indicies based on the provided `type_to_idx` map. Sentences are either truncated or
        right-padded to match a length of constants.MAX_SENT_LEN, and words (character sequences)
        are truncated or right-padded to match a length of constants.MAX_CHAR_LEN.

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
            Preprocessor.log.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)

        # Word type
        if type_ == 'word':
            type_seq = [[type_to_idx.get(x, type_to_idx[constants.UNK]) for x \
                in s] for s in seq]
        # Tag type
        elif type_ == 'tag':
            type_seq = [[type_to_idx.get(x) for x in s] for s in seq]
        # Char type
        elif type_ == 'char':
            # get index sequence of chars
            type_seq = [[[type_to_idx.get(c, type_to_idx[constants.UNK]) for \
                c in w] for w in s] for s in seq]

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
        """Chunks enities in the BIO or BIOES format.

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
        Replaces rare tokens in sentences with a special unknown token.

        Returns `sentences`, a list of list where each inner list is a sentence represented as a
        list of strings (tokens), where all tokens appearing less than `count` number of times have
        been replaced with a special unknown token.

        Args:
            sentences (list): list of lists where each inner list is a sentence
                represented as a list of strings (tokens).
            count (int): threshold for token to be considered rare, tokens appearing `count` times
                or less are replaced with a special unknown token.

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

        For given input `text`, removes proceeding and preeceding spaces, and replaces spans of
        multiple spaces with a single space. Optionally, lowercases `text`.

        Args:
            text (str): text to sterilize
            lower (bool): True if text should be lower cased

        Returns:
            `text`, where proceeding/preeceding spaces have been removed, spans of multiple spaces
            have been replaced with a single space, and optionally, the text has been lowercased
        """
        sterilized_text = re.sub(r'\s+', ' ', text.strip())
        sterilized_text = sterilized_text.lower() if lower else sterilized_text

        return sterilized_text
