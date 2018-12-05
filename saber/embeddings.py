"""Contains the Embedding class, which provides all code for working with pre-trained embeddings.
"""
import numpy as np
from gensim.models import KeyedVectors

from . import constants
from .preprocessor import Preprocessor


class Embeddings(object):
    """A class for loading and working with pre-trained word embeddings.

    Args:
        filepath (str): Path to file which contains pre-trained word embeddings.
        token_map (dict): A dictionary which maps tokens to unique integer IDs.
    """
    def __init__(self, filepath, token_map, **kwargs):
        self.filepath = filepath
        self.token_map = token_map

        self.matrix = None # matrix containing row vectors for all embedded tokens
        self.num_found = None # number of loaded embeddings
        self.num_embed = None # final count of embedded words
        self.dimension = None # dimension of these embeddings

        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self, binary=True, load_all=False):
        """Coordinates the loading of pre-trained word embeddings.

        Creates an embedding matrix from the pre-trained word embeddings given at `self.filepath`,
        whose ith row corresponds to the word embedding for the word with value i in
        `self.token_map`. Updates the instance attributes `self.matrix, `self.num_found`,
        `self.dimension`, and `self.num_embed`.

        Args:
            binary (bool): True if pre-trained embeddings are in C binary format, False if they are
                in C text format. Defaults to True.
            load_all (bool): True if all embeddings should be loaded. False if only words that
            appear in `self.token_map` should be loaded. Defaults to False.

        Returns:
            The token map used to map the embeddings to an embedding matrix.
        """
        # prepare the embedding indices
        embedding_idx = self._prepare_embedding_index(binary)
        self.num_found, self.dimension = len(embedding_idx), len(list(embedding_idx.values())[0])
        self.matrix, type_to_idx = self._prepare_embedding_matrix(embedding_idx, load_all)
        self.num_embed = self.matrix.shape[0] # num of embedded words

        return type_to_idx

    def _prepare_embedding_index(self, binary=True):
        """Returns an embedding index for pre-trained token embeddings.

        For pre-trained word embeddings given at `self.filepath`, returns a
        dictionary mapping words to their embedding (an 'embedding index'). If `self.debug` is
        True, only the first ten thousand vectors are loaded.

        Args:
            binary (bool): True if pre-trained embeddings are in C binary format, False if they are
                in C text format. Defaults to True.

        Returns:
            Dictionary mapping words to pre-trained word embeddings, known as an 'embedding index'.
        """
        limit = 10000 if self.__dict__.get("debug", False) else None
        vectors = KeyedVectors.load_word2vec_format(self.filepath, binary=binary, limit=limit)
        embedding_idx = {word: vectors[word] for word in vectors.vocab}

        return embedding_idx

    def _prepare_embedding_matrix(self, embedding_idx, load_all=False):
        """Returns an embedding matrix containing all pre-trained embeddings in `embedding_idx`.

        Creates an embedding matrix from `embedding_idx`, where the ith row contains the
        embedding for the word with value i in `self.token_map`. If no embedding exists for a given
        word in `embedding_idx`, the zero vector is used instead.

        Args:
            embedding_idx (dict): A Dictionary mapping words to their embeddings.
            load_all (bool): True if all embeddings should be loaded. False if only words that
            appear in `self.token_map` should be loaded. Defaults to False.

        Returns:
            A matrix whos ith row corresponds to the word embedding for the word with value i in
            `self.token_map`.
        """
        type_to_idx = None
        if load_all:
            # overwrite provided token map
            type_to_idx = self._generate_type_to_idx(embedding_idx)
            self.token_map = type_to_idx['word']

        # initialize the embeddings matrix
        embedding_matrix = np.zeros((len(self.token_map), self.dimension))

        # lookup embeddings for every word in the dataset
        for word, i in self.token_map.items():
            token_embedding = embedding_idx.get(word)
            if token_embedding is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = token_embedding

        return embedding_matrix, type_to_idx

    @classmethod
    def _generate_type_to_idx(self, embedding_idx):
        """Returns a dictionary mapping tokens in  `embedding_idx` to unique integer IDs.
        """
        word_types, char_types = list(embedding_idx.keys()), []
        for word in word_types:
            char_types.extend(list(word))
        char_types = list(set(char_types))

        type_to_idx = {
            'word': Preprocessor.type_to_idx(word_types, constants.INITIAL_MAPPING['word']),
            'char': Preprocessor.type_to_idx(char_types, constants.INITIAL_MAPPING['word'])
        }

        return type_to_idx
