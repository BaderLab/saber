import spacy
import en_core_web_sm

from keras.preprocessing.sequence import pad_sequences

UNK = '<UNK>'
PAD_IDX = 0

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

        word_idx_sequence = self.get_type_idx_sequence(sentences, w2i, type='word')
        char_idx_sequence = self.get_type_idx_sequence(sentences, c2i, type='char')

        print(word_idx_sequence)

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
            token_sequence = [token.text for token in self.nlp(sent.text)]
            sentences.append(token_sequence)

        return list(word_types), list(char_types), sentences

    @staticmethod
    def sequence_to_idx(seq, offset=0):
        """Returns a dictionary of element:index pairs for each element in
        sequence.

        Given a list, returns a dictionary of length len(sequence) + offset
        where the keys are elements of sequence and the values are unique
        integers.

        Args:
            sequence (list): sequence data.
            offset (int): used when computing the mapping. An offset a 1 means
                    we begin computing the mapping at 1, useful if we want to
                    use 0 as a padding value.
        Returns:
            a mapping from elements in the sequence to numbered indices

        Preconditions:
            assumes that all elements in sequence are unique
        """
        # offset accounts for sequence pad
        return {e: i + offset for i, e in enumerate(seq)}

    @staticmethod
    def get_type_idx_sequence(seq, type_to_idx, type='word'):
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
        assert type in ['word', 'char', 'tag'], "Argument type must be one 'word', 'char' or 'type'"

        pad = 0 # sequence pad

        # Word type
        if type == 'word':
            type_seq = [[type_to_idx.get(x, type_to_idx[UNK]) for x in s] \
                for s in seq]
        # Tag type
        elif type == 'tag':
            type_seq = [[type_to_idx.get(x) for x in s] for s in seq]
        # Char type
        elif type == 'char':
            # get index sequence of chars
            type_seq = [[[type_to_idx.get(c, type_to_idx[UNK]) for c in w] \
                for w in s] for s in seq]

            # TODO (johngiorgi): this can't be the most efficient sol'n to
            # get the length of the longest character sequence
            max_len = max([len(x) for x in max(type_seq, key=(lambda x: len(x)))])

            # create a sequence of padded character vectors
            for i, char_seq in enumerate(type_seq):
                type_seq[i] = pad_sequences(maxlen=max_len,
                                            sequences=char_seq,
                                            padding="post",
                                            truncating='post',
                                            value=PAD_IDX)

        # pad sequences
        type_seq = pad_sequences(sequences=type_seq,
                                 padding='post',
                                 truncating='post',
                                 value=PAD_IDX)

        return type_seq
