import os
import codecs

import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/
# https://keras.io/metrics/

# TODO (johngiorgi): there is some hard coded ugliness going on in print_table, fix this.
# TODO (johngiorgi): this is likely copying big lists, find a way to get around this

class Metrics(Callback):
    """ A class for handling performance metrics, inherits from Callback. """
    def __init__(self,
                 X_train,
                 X_valid,
                 y_train,
                 y_valid,
                 tag_type_to_idx):
        # training data
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

        self.tag_type_to_idx = tag_type_to_idx

        # epoch counter for model tied to this object
        self.current_epoch = 1

    def on_train_begin(self, logs={}):
        """ Series of steps to perform when training begins. """
        ## TRAIN
        self.train_precision_per_epoch = {}
        self.train_recall_per_epoch = {}
        self.train_f1_per_epoch = {}
        ## VALID
        self.valid_precision_per_epoch = {}
        self.valid_recall_per_epoch = {}
        self.valid_f1_per_epoch = {}

    def on_epoch_end(self, epoch, logs={}):
        """ Series of steps to perform when epoch ends. """
        # get train/valid performance metrics
        train_scores = self._eval(self.X_train, self.y_train)
        valid_scores = self._eval(self.X_valid, self.y_valid)

        self._pretty_print_performance_scores(train_scores, title='train')
        self._pretty_print_performance_scores(valid_scores, title='valid')

        self.current_epoch += 1 # update the current epoch counter

    def _eval(self, X, y):
        """ Performs all evaluation steps for given X (input) and y (labels).

        For a given input (X) and labels (y) performs all the steps in the
        evaluation pipeline, namely: performs prediction on X, chunks the
        annotations by type, and computes performance scores by type.

        Args:
            X: input matrix, of shape (num examples X sequence length)
            y: lables, of shape (num examples X sequence length X num classes)
        """
        # get predictions and gold labels
        y_true, y_pred = self._get_y_true_and_pred(X, y)
        # convert idx sequence to tag sequence
        y_true_tag = self._index_to_tag(y_true, self.tag_type_to_idx)
        y_pred_tag = self._index_to_tag(y_pred, self.tag_type_to_idx)
        # chunk the entities
        y_true_chunks = self._chunk_entities(y_true_tag)
        y_pred_chunks = self._chunk_entities(y_pred_tag)

        # get performance scores per label
        performance_scores = self._get_precision_recall_f1_support(y_true_chunks,
                                                                   y_pred_chunks)

        return performance_scores

    def _get_y_true_and_pred(self, X, y):
        """ Get y_true and y_pred for given input data (X) and labels (y)

        Performs prediction for the current model (self.model), and returns
        a 2-tuple contain 1D array-like objects containing the true (gold)
        labels and the predicted labels, where labels are integers corresponding
        to the sequence tags as per self.tag_type_to_idx.

        Args:
            X: input matrix, of shape (num examples X sequence length)
            y: lables, of shape (num examples X sequence length X num classes)

        Returns:
            y_true: 1D array like object containing the gold label sequence
            y_pred: 1D array like object containing the predicted sequences
        """
        # gold labels
        y_true = y.argmax(axis=-1) # get class label
        y_true = np.asarray(y_true).ravel() # flatten to 1D array
        # predicted labels
        y_pred = self.model.predict(X).argmax(axis=-1)
        y_pred = np.asarray(y_pred).ravel()

        # sanity check
        assert y_true.shape == y_pred.shape, """y_true and y_pred have different
        shapes"""

        return y_true, y_pred

    def _index_to_tag(self, y, tag_type_to_idx):
        """ Converts a sequence of indices to their corresponding tags.

        For a given sequence of indices, returns the corresponding sequence of
        entity tags based on the giving mapping tag_type_to_idx.

        Args:
            y: 1D array like object containing the index sequence
            tag_type_to_idx: a mapping entity tags to numbered indices

        """
        tag_key_list = list(tag_type_to_idx.keys())
        tag_idx_list = list(tag_type_to_idx.values())

        idx_to_tag = []

        for i, idx in enumerate(y):
            idx_to_tag.append(tag_key_list[tag_idx_list.index(idx)])

        return idx_to_tag

    def _chunk_entities(self, seq):
        """ Chunks enities in the BIO or BIOES format.

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

    def _get_precision_recall_f1_support(self, y_true, y_pred):
        """ Returns precision, recall, f1 and support.

        For given gold (y_true) and predicited (y_pred) labels, returns the
        precision, recall, f1 and support. Expected y_true and y_pred to be
        a sequence of entity chunks.

        Args:
            y_true: list of (chunk_type, chunk_start, chunk_end)
            y_pred: list of (chunk_type, chunk_start, chunk_end)
        Returns:
            dictionary containing (precision, recall, f1, support) for each
            chunk type.
        """
        performance_scores = {} # dict acc of metrics
        labels = list(set([chunk[0] for chunk in y_true])) # unique labels

        # get performance scores per label
        for lab in labels:
            # get chunks for current lab
            y_true_lab = [chunk for chunk in y_true if chunk[0] == lab]
            y_pred_lab = [chunk for chunk in y_pred if chunk[0] == lab]

            # accumulators
            FN = 0
            FP = 0
            TP = 0

            # FN
            for gold in y_true_lab:
                if gold not in y_pred_lab:
                    FN += 1

            for pred in y_pred_lab:
                # FP
                if pred not in y_true_lab:
                    FP += 1
                # TP
                elif pred in y_true_lab:
                    TP += 1

            # get performance metrics
            p = TP / (TP + FP) if TP > 0 else 0
            r = TP / (TP + FN) if TP > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            s = len(y_true_lab)

            performance_scores[lab] = (p, r, f1, s)

        return performance_scores

    def _pretty_print_performance_scores(self, performance_scores, title='train'):
        """ Prints a table of performance scores.

        Args:
            performance_scores: a dictionary of label, score pairs where label
                                is a sequence tag and scores is a 4-tuple
                                containing precision, recall, f1 and support
        """
        # collect table dimensions
        col_width = 20
        col_space = ' ' * col_width
        col_width_1 = len('Label') + col_width
        col_width_2 = col_width_1 + len('Precision') + col_width
        col_width_3 = col_width_2 + len('Recall') + col_width
        col_width_4 = col_width_3 + len('F1') + col_width

        tab_width = 120
        light_line = '-' * tab_width
        heavy_line = '=' * tab_width

        ## HEADER
        print()
        print()
        title = '{col}{t}{col}'.format(t=title.upper(),
                                       col=' '*((tab_width-len(title))//2))
        header = 'Label{col}Precision{col}Recall{col}F1{col}Support'.format(col=col_space)
        print(heavy_line)
        print(title)
        print(light_line)
        print(header)

        print(heavy_line)
        ## BODY
        for label, score in performance_scores.items():
            # specify an entire row
            row = '{lab}{col1}{p:.1%}{col2}{r:.1%}{col3}{f1:.1%}{col4}{s}'.format(
                p=score[0],
                r=score[1],
                f1=score[2],
                s=score[3],
                lab=label,
                col1=' ' * (col_width_1 - len(label) + len('Precision')//3 - 2),
                col2=' ' * (col_width_2 - col_width_1 - len('Precision')//3 - 2),
                col3=' ' * (col_width_3 - col_width_2 - len('Precision')//3 - 2),
                col4=' ' * (col_width_4 - col_width_3 - len('Precision')//3))
            print(row)
        print(light_line)
