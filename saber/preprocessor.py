import spacy
import en_core_web_sm

from keras.preprocessing.sequence import pad_sequences

class Preprocessor(object):
    """A class for processing text data."""
    def __init__(self):

        # TODO (johngiorgi): Read about spacys models, choose the most
        # sensible for our purposes. Update the requirements.txt file if you
        # change this, and the import statement above!

        # Load Spacy english model
        self.nlp = en_core_web_sm.load()

    def transform(self, text, w2i, c2i):
        """
        """
        # get tokens, sentences, and word_types
        word_types, char_types, sentences = self._process_text(text)
        word_idx_sequence = self.get_type_idx_sequence(sentences,
                                                       word_type_to_idx=w2i)

        char_idx_sequence = self.get_type_idx_sequence(sentences,
                                                       char_type_to_idx=c2i)

        return word_idx_sequence, char_idx_sequence

    def _process_text(self, text):
        """Process raw text."""
        doc = self.nlp(text) # process text with spacy

        # accumulators
        word_types = set()
        char_types = set()
        sentences = []

        # TODO (johngiorgi): I am almost sure that there is a more elegant
        # way to do this with the Spacy API. Figure it out!

        # get tokens
        for token in doc:
            word_types.add(token.text)

        # get characters
        for word in word_types:
            char_types.update(list(word))

        # get sentences
        for sent in doc.sents:
            token_sequence = [(token.text, 'O') for token in self.nlp(sent.text)]
            sentences.append(token_sequence)

        return list(word_types), list(char_types), sentences

    @staticmethod
    def sequence_to_idx(sequence, offset=0):
        """Returns a dictionary of element:idx pairs for each element in sequence.

        Given a list, returns a dictionary of length len(sequence) + pad where
        the keys are elements of sequence and the values are unique integers.

        Args:
            sequence (list): sequence data.
            offset (int): used when computing the mapping. An offset a 1 means
                    we begin computing the mapping at 1, useful if we want to
                    use 0 as a padding value.
        Returns:
            a mapping from elements in the sequence to numbered indices
        """
        # offset accounts for sequence pad
        return {e: i + offset for i, e in enumerate(list(set(sequence)))}

    @staticmethod
    def get_type_idx_sequence(sentences,
                              word_type_to_idx=None,
                              char_type_to_idx=None,
                              tag_type_to_idx=None):
        """Returns sequence of indices corresponding to data set sentences.

        Returns the sequence of idices corresponding to the type order in
        sentences, where type_ can be "word", "char", or "tag" correpsonding to
        word, char and tag types of the Dataset instance.

        Args:
            sentences: a list of lists, where each list represents a sentence
                for the Dataset instance and each sublist contains ordered
                (word, tag) pairs.

        Returns:
            a list, containing a sequence of idx's corresponding to the type
            order in sentences.

        Preconditions:
            assumes that the first column of the data set contains the word
            types, and the last column contains the tag types.
        """
        assert ((word_type_to_idx is not None) or (char_type_to_idx is not None)
            or (tag_type_to_idx is not None)), '''One of of word_type_to_idx,
            char_type_to_idx, or tag_type_to_idx must be provided.'''

        col_idx = -1 if tag_type_to_idx is not None else 0
        # pad allows use of mask_zero parameter to ignore inputs with value zero
        pad = tag_type_to_idx['O'] if tag_type_to_idx is not None else 0

        # char type
        if char_type_to_idx is not None:
            # get sequence of chars
            type_seq = [[[char_type_to_idx[ch] for ch in ty[col_idx]] for ty in s] for s in sentences]

            # TODO (johngiorgi): this can't be the most efficient sol'n
            # get the length of the longest character sequence
            max_len = max([len(x) for x in max(type_seq, key=(lambda x: len(x)))])

            # create a sequence of padded character vectors
            for i, char_seq in enumerate(type_seq):
                type_seq[i] = pad_sequences(maxlen=max_len,
                                            sequences=char_seq,
                                            padding="post",
                                            truncating='post',
                                            value=pad)
        else:
            # word type
            if word_type_to_idx is not None:
                type_to_idx = word_type_to_idx
            # tag type
            elif tag_type_to_idx is not None:
                type_to_idx = tag_type_to_idx

            # get sequence of types (word, or tag)
            type_seq = [[type_to_idx[ty[col_idx]] for ty in s] for s in sentences]

        # pad sequences
        type_seq = pad_sequences(sequences=type_seq,
                                 padding='post',
                                 truncating='post',
                                 value=pad)

        return type_seq
