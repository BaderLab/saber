import json
import os
import time
from statistics import mean
from operator import itemgetter

import numpy as np
from prettytable import PrettyTable
from keras.callbacks import Callback

import utils_models
from utils_generic import make_dir
from preprocessor import Preprocessor

# TODO (johngiorgi): there is some hard coded ugliness going on in print_table, fix this.
# TODO (johngiorgi): this is likely copying big lists, find a way to get around this

class Metrics(Callback):
    """A class for handling performance metrics, inherits from Callback."""
    def __init__(self, X_train, X_valid, y_train, y_valid, idx_to_tag_type,
                 output_dir, fold=0):
        # training data
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

        # inversed mapping from idx: tag
        self.idx_to_tag_type = idx_to_tag_type

        self.output_dir = output_dir

        # epoch counter for model tied to this object
        self.current_epoch = 0

        # current k-fold counter for model tied to this object
        self.current_fold = fold

        # Model performance metrics accumulators
        self.train_performance_metrics_per_epoch = []
        self.valid_performance_metrics_per_epoch = []

    def on_train_begin(self, logs={}):
        """Series of steps to perform when training begins."""
        pass

    def on_epoch_end(self, epoch, logs={}):
        """Series of steps to perform when epoch ends."""
        # get train/valid performance metrics
        train_scores = self._eval(self.X_train, self.y_train)
        valid_scores = self._eval(self.X_valid, self.y_valid)

        self.print_performance_scores(train_scores, title='train')
        self.print_performance_scores(valid_scores, title='valid')

        # accumulate peformance metrics
        self.train_performance_metrics_per_epoch.append(train_scores)
        self.valid_performance_metrics_per_epoch.append(valid_scores)

        # write the performance metrics for the current epoch to disk
        self._write_metrics_to_disk()

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
        y_true_tag = [self.idx_to_tag_type[idx] for idx in y_true]
        y_pred_tag = [self.idx_to_tag_type[idx] for idx in y_pred]
        # chunk the entities
        y_true_chunks = Preprocessor.chunk_entities(y_true_tag)
        y_pred_chunks = Preprocessor.chunk_entities(y_pred_tag)

        # get performance scores per label
        performance_scores = self.get_precision_recall_f1_support(y_true_chunks,
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
        y_pred = self.model.predict(X, batch_size=256).argmax(axis=-1)
        y_pred = np.asarray(y_pred).ravel()

        # sanity check
        if not y_true.shape == y_pred.shape:
            raise AssertionError("y_true and y_pred have different shapes")

        return y_true, y_pred

    @staticmethod
    def get_precision_recall_f1_support(y_true, y_pred):
        """Returns precision, recall, f1 and support.

        For given gold (y_true) and predicited (y_pred) labels, returns the
        precision, recall, f1 and support per label and the average across
        labels. Expected y_true and y_pred to be a sequence of entity chunks.

        Args:
            y_true: list of (chunk_type, chunk_start, chunk_end)
            y_pred: list of (chunk_type, chunk_start, chunk_end)
        Returns:
            dict: dictionary containing (precision, recall, f1, support) for
            each chunk type and the average across chunk types.
        """
        performance_scores = {} # dict accumulator of per label of scores
        # micro performance accumulators
        FN_total = 0
        FP_total = 0
        TP_total = 0

        labels = list(set([chunk[0] for chunk in y_true])) # unique labels

        # get performance scores per label
        for lab in labels:
            # get chunks for current lab
            y_true_lab = [chunk for chunk in y_true if chunk[0] == lab]
            y_pred_lab = [chunk for chunk in y_pred if chunk[0] == lab]

            # per label performance accumulators
            FN = 0
            FP = 0
            TP = 0

            # FN
            for gold in y_true_lab:
                if gold not in y_pred_lab:
                    FN += 1

            for pred in y_pred_lab:
                # FP / TP
                if pred not in y_true_lab:
                    FP += 1
                # TP
                elif pred in y_true_lab:
                    TP += 1

            # get performance metrics
            performance_scores[lab] = \
                utils_models.precision_recall_f1_support(TP, FP, FN)

            # accumulate FNs, FPs, TPs
            FN_total += FN
            FP_total += FP
            TP_total += TP

        # get macro and micro peformance metrics averages
        macro_p = mean([v[0] for v in performance_scores.values()])
        macro_r = mean([v[1] for v in performance_scores.values()])
        macro_f1 = mean([v[2] for v in performance_scores.values()])
        total_support = TP_total + FN_total

        performance_scores['MACRO_AVG'] = \
            (macro_p, macro_r, macro_f1, total_support)
        performance_scores['MICRO_AVG'] = \
            utils_models.precision_recall_f1_support(TP_total, FP_total, \
                FN_total)

        return performance_scores

    @staticmethod
    def print_performance_scores(performance_scores, title=None):
        """Prints an ASCII table of performance scores.

        Args:
            performance_scores: a dictionary of label, score pairs where label
                                is a class tag and scores is a 4-tuple
                                containing precision, recall, f1 and support
            title (str): the title of the table (uppercased).

        Preconditions:
            assumes the values of performance_scores are 4-tuples, where the
            first three items are float representaions of a percentage and the
            last item is an count integer.
        """
        # create table, give it a title a column names
        table = PrettyTable()
        if title is not None: table.title = title.upper()
        table.field_names = ['Label', 'Precision', 'Recall', 'F1', 'Support']

        # column alignment
        table.align['Label'] = 'l'
        table.align['Precision'] = 'r'
        table.align['Recall'] = 'r'
        table.align['F1'] = 'r'
        table.align['Support'] = 'r'

        # create and add the rows
        for label, scores in performance_scores.items():
            row = [label]
            # convert scores to formatted percentage strings
            support = scores[-1]
            per_metrics = ['{:.2%}'.format(x) for x in scores[:-1]]
            row_scores = per_metrics + [support]

            row.extend(row_scores)
            table.add_row(row)

        print(table)

    def _write_metrics_to_disk(self):
        """Write performance metrics to disk as json-formatted .txt file.

        At the end of each epoch, writes a json-formatted .txt file to disk
        (name epoch_<epoch_number>.txt). File contains performance scores
        per label as well as the best-achieved macro and micro averages thus
        far for the current epoch.
        """
        # create evaluation output directory
        fold = 'fold_{}'.format(self.current_fold + 1)
        eval_dirname = os.path.join(self.output_dir, fold)
        make_dir(eval_dirname)

        # create filepath to evaluation file
        eval_filename = 'epoch_{0:03d}.txt'.format(self.current_epoch + 1)
        eval_filepath = os.path.join(eval_dirname, eval_filename)

        # get best epoch based on macro / micro averages
        macro_avg_per_epoch = [x['MACRO_AVG'] for x in self.valid_performance_metrics_per_epoch]
        micro_avg_per_epoch = [x['MICRO_AVG'] for x in self.valid_performance_metrics_per_epoch]

        best_macro_avg_val_score = max(macro_avg_per_epoch, key=itemgetter(2))
        best_micro_avg_val_score = max(micro_avg_per_epoch, key=itemgetter(2))
        best_micro_epoch = micro_avg_per_epoch.index(best_micro_avg_val_score)
        best_macro_epoch = macro_avg_per_epoch.index(best_macro_avg_val_score)

        performance_metrics = {}
        performance_metrics['performance_scores'] = \
            self.valid_performance_metrics_per_epoch[self.current_epoch]
        performance_metrics['best_epoch_macro_avg'] = \
            {'epoch': best_macro_epoch, 'scores': best_macro_avg_val_score}
        performance_metrics['best_epoch_micro_avg'] = \
            {'epoch': best_micro_epoch, 'scores': best_micro_avg_val_score}

        # write performance metrics for current epoch to file
        with open(eval_filepath, 'a') as f:
            f.write(json.dumps(performance_metrics))
