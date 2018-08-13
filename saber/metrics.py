"""Contains the Metrics class, which computes, stores, prints and saves performance metrics
for a Keras model.
"""
import json
import logging
from operator import itemgetter
import os
from statistics import mean

import numpy as np
from prettytable import PrettyTable
from keras.callbacks import Callback

from . import constants
from .utils import model_utils
from .utils.generic_utils import make_dir
from .preprocessor import Preprocessor

class Metrics(Callback):
    """A class for handling performance metrics, inherits from Callback.

    Args:
        training_data (dict): a dictionary with keys 'X_<partition>', 'y_<partition>' which point
            to the data and labels for all partitions 'train', and optionally 'valid', 'test'
            respectively
        idx_to_tag (dict): maps each unique integer index to its label, or tag
        output_dir (str): base directory to save all output to
        criteria (str): criteria which determines which predictions are true-positives.
            One of: 'left' for left-boundary matching, 'right' for right-boundary matching and
            'exact' for exact-boundary matching (default)
        fold (int): current fold, if using k-fold cross validation, defaults to 0.
    """
    # define this at the class level because some methods are static
    log = logging.getLogger(__name__)

    def __init__(self, training_data, idx_to_tag, output_dir, criteria='exact', fold=None):

        self.training_data = training_data

        # inversed mapping from idx: tag
        self.idx_to_tag = idx_to_tag

        self.output_dir = output_dir

        # matching criteria to use when evaluating true_positives
        self.criteria = criteria

        # epoch counter for model tied to this object
        self.current_epoch = 0
        # current k-fold counter for model tied to this object
        self.current_fold = fold

        # Model performance metrics accumulators
        self.performance_metrics_per_epoch = {p: [] for p in constants.PARTITIONS}

    def on_train_begin(self, logs={}):
        """Series of steps to perform when training begins."""
        pass

    def on_epoch_end(self, epoch, logs={}):
        """Series of steps to perform when epoch ends."""
        # get train/valid/test scores, accumulate them, and print them

        # Train
        train_scores = self._eval(self.training_data['X_train'], self.training_data['y_train'])
        self.print_performance_scores(train_scores, title='train')
        self.performance_metrics_per_epoch['train'].append(train_scores)

        # Valid
        valid_scores = self._eval(self.training_data['X_valid'], self.training_data['y_valid'])
        self.print_performance_scores(valid_scores, title='valid')
        self.performance_metrics_per_epoch['valid'].append(valid_scores)

        # Test (optional)
        if self.training_data['X_test'] is not None:
            test_scores = self._eval(self.training_data['X_test'], self.training_data['y_test'])
            self.print_performance_scores(test_scores, title='test')
            self.performance_metrics_per_epoch['test'].append(test_scores)

        # write the performance metrics for the current epoch to disk
        self._write_metrics_to_disk()

        self.current_epoch += 1 # update the current epoch counter

    def _eval(self, X, y):
        """Performs all evaluation steps for given X (input) and y (labels).

        For a given input (X) and labels (y) performs all the steps in the evaluation pipeline,
        namely: performs prediction on X, chunks the annotations by type, and computes performance
        scores by type.

        Args:
            X: input matrix, of shape (num examples X sequence length)
            y: lables, of shape (num examples X sequence length X num classes)

        Returns:
            a dictionary of label, score key, value pairs where label is a class tag and scores is
            a 4-tuple containing precision, recall, f1 and support
        """
        # get predictions and gold labels
        y_true, y_pred = self._get_y_true_and_pred(X, y)
        # convert idx sequence to tag sequence
        y_true_tag = [self.idx_to_tag[idx] for idx in y_true]
        y_pred_tag = [self.idx_to_tag[idx] for idx in y_pred]
        # chunk the entities
        y_true_chunks = Preprocessor.chunk_entities(y_true_tag)
        y_pred_chunks = Preprocessor.chunk_entities(y_pred_tag)

        # get performance scores per label
        performance_scores = \
            self.get_precision_recall_f1_support(y_true_chunks, y_pred_chunks,
                                                 criteria=self.criteria)

        # TEMP: Check CoNLLEval script
        # self.conll_eval(y_true_tag, y_pred_tag)

        return performance_scores

    # TEMP: Check CoNLLEval script
    def conll_eval(self, y_true, y_pred):
        """Calls conlleval script on ConLL-formated file containing predictions and gold labels.

        Args:
            y_true: list of gold label tags for a sequence, e.g. ['O', 'O', 'B-DISO', 'I-DISO']
            y_pred: list of predicted tags for a sequence
        """
        conll_file = os.path.join(self.output_dir, 'conll.tsv')
        with open(conll_file, 'w') as f:
            for gold, pred in zip(y_true, y_pred):
                if gold != constants.PAD and pred != constants.PAD:
                    f.write('{}\t{}\t{}\n'.format('DUMMY', gold, pred))
        os.system('python2 ./conlleval.py {}'.format(conll_file))

    def _get_y_true_and_pred(self, X, y):
        """ Get y_true and y_pred for given input data (X) and labels (y)

        Performs prediction for the current model (self.model), and returns a 2-tuple contain 1D
        array-like objects containing the true (gold) labels and the predicted labels, where labels
        are integers corresponding to the sequence tags as per self.tag_type_to_idx.

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
        y_pred = self.model.predict(X, batch_size=constants.PRED_BATCH_SIZE)
        y_pred = np.asarray(y_pred.argmax(axis=-1)).ravel()

        # sanity check
        if not y_true.shape == y_pred.shape:
            Metrics.log.error(("AssertionError raised because 'y_pred' and 'y_true' in "
                               "'Metrics._get_y_true_and_pred()' have different shapes"))
            raise AssertionError("y_true and y_pred have different shapes")

        return y_true, y_pred

    @staticmethod
    def get_precision_recall_f1_support(y_true, y_pred, criteria='exact'):
        """Returns precision, recall, f1 and support.

        For given gold (y_true) and predicited (y_pred) labels, returns the precision, recall, f1
        and support per label and the average across labels. Expected y_true and y_pred to be a
        sequence of entity chunks.

        Args:
            y_true: list of (chunk_type, chunk_start, chunk_end)
            y_pred: list of (chunk_type, chunk_start, chunk_end)
            criteria (str): criteria to use for evaluation, 'exact' matches
                boundaries directly, 'left' requires only a left boundary match
                and 'right requires only a right boundary match'.
        Returns:
            a dictionary of label, score key, value pairs where label is a class tag and scores is
            a 4-tuple containing precision, recall, f1 and support

        Raises:
            ValueError, if 'criteria' is not one of 'exact', 'left', or 'right'
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
            y_pred_lab = [], []
            # either retain or discard left or right boundaries depending on
            # matching criteria
            if criteria not in ['exact', 'left', 'right']:
                Metrics.log.error(("ValueError raised because 'criteria' in 'Metrics.get_precision_"
                                   "recall_f1_support()' is not one of 'exact', 'left', 'right'"))
                raise ValueError(("Expected criteria to be one of 'exact', 'left', or 'right'."
                                  "Got: {}".format(criteria)))
            if criteria == 'exact':
                y_true_lab = [chunk for chunk in y_true if chunk[0] == lab]
                y_pred_lab = [chunk for chunk in y_pred if chunk[0] == lab]
            elif criteria == 'left':
                y_true_lab = [chunk[:2] for chunk in y_true if chunk[0] == lab]
                y_pred_lab = [chunk[:2] for chunk in y_pred if chunk[0] == lab]
            elif criteria == 'right':
                y_true_lab = [chunk[::2] for chunk in y_true if chunk[0] == lab]
                y_pred_lab = [chunk[::2] for chunk in y_pred if chunk[0] == lab]

            # per label performance accumulators
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
            performance_scores[lab] = \
                model_utils.precision_recall_f1_support(TP, FP, FN)

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
            model_utils.precision_recall_f1_support(TP_total, FP_total, \
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
        if title is not None:
            table.title = title.upper()
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
            performance_metrics = ['{:.2%}'.format(x) for x in scores[:-1]]
            row_scores = performance_metrics + [support]

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
        eval_dirname = self.output_dir
        if self.current_fold is not None:
            fold = 'fold_{}'.format(self.current_fold + 1)
            eval_dirname = os.path.join(self.output_dir, fold)
        make_dir(eval_dirname)

        # create filepath to evaluation file
        eval_filename = 'epoch_{0:03d}.txt'.format(self.current_epoch + 1)
        eval_filepath = os.path.join(eval_dirname, eval_filename)

        # per partition performance metrics accumulator
        performance_metrics = {p: {} for p in constants.PARTITIONS}

        for partition in self.performance_metrics_per_epoch:
            # test partition may be empty
            if self.performance_metrics_per_epoch[partition]:
                # get best epoch based on macro / micro averages
                macro_avg_per_epoch = [x['MACRO_AVG'] for x in
                                       self.performance_metrics_per_epoch[partition]]
                micro_avg_per_epoch = [x['MICRO_AVG'] for x in
                                       self.performance_metrics_per_epoch[partition]]

                best_macro_avg = max(macro_avg_per_epoch, key=itemgetter(2))
                best_micro_avg = max(micro_avg_per_epoch, key=itemgetter(2))

                best_micro_epoch = micro_avg_per_epoch.index(best_micro_avg)
                best_macro_epoch = macro_avg_per_epoch.index(best_macro_avg)

                performance_metrics[partition]['scores'] = \
                    self.performance_metrics_per_epoch[partition][self.current_epoch]
                performance_metrics[partition]['best_epoch_macro_avg'] = \
                    {'epoch': best_macro_epoch, 'scores': best_macro_avg}
                performance_metrics[partition]['best_epoch_micro_avg'] = \
                    {'epoch': best_micro_epoch, 'scores': best_micro_avg}

        # write performance metrics for current epoch to file
        with open(eval_filepath, 'a') as eval_file:
            eval_file.write(json.dumps(performance_metrics, indent=4))
