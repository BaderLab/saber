import re
import spacy
import en_core_web_sm

from keras.preprocessing.sequence import pad_sequences

UNK = '<UNK>'
PAD_VALUE = 0

class Preprocessor(object):
    """A class for processing text data."""
    def __init__(self):
        # Load Spacy english model (small), disable NER pipeline
        self.nlp = en_core_web_sm.load(disable=['ner'])

    def transform(self, text, w2i, c2i):
        """
        """
        text_ = self._sterilize(text)
        # get sentences and token offsets
        sentences, offsets = self._process_text(text_)

        word_idx_sequence = self.get_type_idx_sequence(sentences, w2i, type='word')
        char_idx_sequence = self.get_type_idx_sequence(sentences, c2i, type='char')

        transformed_text = {
            'text': text_,
            'sentences': sentences,
            'offsets': offsets,
            'word_idx_sequence': word_idx_sequence,
            'char_idx_sequence': char_idx_sequence
        }

        return transformed_text

    def _process_text(self, text):
        """Process raw text."""
        doc = self.nlp(text) # process text with spacy

        # accumulators
        # word_types = set()
        # char_types = set()
        sentences = []
        offsets = []

        # TODO (johngiorgi): Do I need word types?
        # collect token sequence and word types
        for sent in doc.sents:
            token_seq = []
            token_offset_seq = []
            for token in sent:
                # word_types.add(token.text)
                token_seq.append(token.text)
                token_offset_seq.append((token.idx, token.idx + len(token.text)))
            sentences.append(token_seq)
            offsets.append(token_offset_seq)

        # TODO (johngiorgi): Do I need character types?
        # collect characters types
        # for word in word_types:
        #    char_types.update(list(word))

        # sanity check
        assert [len(s) for s in sentences] == [len(o) for o in offsets], 'sentences of offsets differ in size.'

        return sentences, offsets

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
                                            value=PAD_VALUE)

        # pad sequences
        type_seq = pad_sequences(sequences=type_seq,
                                 padding='post',
                                 truncating='post',
                                 value=PAD_VALUE)

        return type_seq

    @staticmethod
    def chunk_entities(seq):
        """Chunks enities in the BIO or BIOES format.

        For a given sequence of entities in the BIO or BIOES format, returns
        the chunked entities.

        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            >>> seq = ['B-PRGE', 'I-PRGE', 'O', 'B-PRGE']
            >>> print(get_entities(seq))
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

    def _sterilize(self, text):
        """Sterilize input text.

        For given input text, remove proceeding and preeceding spaces, and replace
        spans of multiple spaces with a single space.

        Args:
            text (str): text to sterilize

        Returns:
            sterilized message
        """
        return re.sub('\s+', ' ', text.strip())
