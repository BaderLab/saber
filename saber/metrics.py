"""Contains the Metrics class, which computes, stores, prints and saves performance metrics
for a Keras model.
"""
import json
import logging
import os
from operator import itemgetter
from statistics import mean

import numpy as np
from keras.callbacks import Callback
from prettytable import PrettyTable

from . import constants
from .preprocessor import Preprocessor
from .utils import model_utils
from .utils.generic_utils import make_dir

# define this at the class level because some methods are static
LOGGER = logging.getLogger(__name__)

class Metrics(Callback):
    """A class for handling performance metrics, inherits from Callback.

    Args:
        config (Config): Contains a set of harmonized arguments provided in a *.ini file and,
            optionally, from the command line.
        data (dict): Contains the data and targets for each partition: 'train', 'valid' and 'test'.
        index_map (dict): A dictionary mapping unique integers to each target (a label).
        output_dir (str): Base directory to save all output to.
    """
    def __init__(self, config, training_data, index_map, output_dir, **kwargs):
        self.config = config # hyperparameters and model details
        self.training_data = training_data # inputs and targets for each partition
        self.index_map = index_map # maps unique IDs to targets

        self.output_dir = output_dir
        self.current_epoch = 0

        # Model performance metrics accumulators
        self.performance_metrics = {p: [] for p in constants.PARTITIONS}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def on_epoch_end(self, epoch, logs={}):
        """Computes, accumulates and prints train/valid/test scores at the end of each epoch."""
        # train
        train_scores = self._evaluate(self.training_data['x_train'], self.training_data['y_train'])
        self.print_performance_scores(train_scores, title='train')
        self.performance_metrics['train'].append(train_scores)
        # valid
        valid_scores = self._evaluate(self.training_data['x_valid'], self.training_data['y_valid'])
        self.print_performance_scores(valid_scores, title='valid')
        self.performance_metrics['valid'].append(valid_scores)
        # test (optional)
        if self.training_data['x_test'] is not None:
            test_scores = self._evaluate(self.training_data['x_test'], self.training_data['y_test'])
            self.print_performance_scores(test_scores, title='test')
            self.performance_metrics['test'].append(test_scores)

        self._write_metrics_to_disk()
        self.current_epoch += 1

    def _evaluate(self, X, y):
        """Performs all evaluation steps for given set of inputs (`X`) and targets (`y`).

        For a given input (`X`) and targets (`y`) performs all the steps in the evaluation pipeline,
        namely: performs prediction on `X`, chunks the annotations by type, and computes performance
        scores by type.

        Args:
            X (numpy.ndarrary): Input matrix; shape (num examples, sequence length).
            y (numpy.ndarrary): Target matrix; shape (num examples, sequence length, num classes).

        Returns:
            A dictionary of label, score pairs; where label is a class tag and scores is a 4-tuple
            containing precision, recall, f1 and support for that class.
        """
        # get predictions and gold labels
        y_true, y_pred = self._get_y_true_and_pred(X, y)
        # convert index sequence to tag sequence
        y_true_tag = [self.index_map[idx] for idx in y_true]
        y_pred_tag = [self.index_map[idx] for idx in y_pred]
        # chunk the entities
        y_true_chunks = Preprocessor.chunk_entities(y_true_tag)
        y_pred_chunks = Preprocessor.chunk_entities(y_pred_tag)

        # get performance scores per label
        return self.get_precision_recall_f1_support(y_true=y_true_chunks,
                                                    y_pred=y_pred_chunks,
                                                    criteria=self.config.criteria)

    def _get_y_true_and_pred(self, X, y):
        """Get `y_true` and `y_pred` for given inputs (`X`) and targets (`y`).

        Performs prediction for the current model (`self.model`), and returns a 2-tuple containing
        the true (gold) labels and the predicted labels, where labels are integers corresponding to
        mapping at `self.index_map`.

        Args:
            X (numpy.ndarrary): Input matrix; shape (num examples, sequence length).
            y (numpy.ndarrary): Target matrix; shape (num examples, sequence length, num classes).

        Returns:
            A two-tuple containing the gold label integer sequences and the predicted integer label
            sequences.
        """
        # gold labels
        y_true = y.argmax(axis=-1) # get class label
        y_true = np.asarray(y_true).ravel() # flatten to 1D array
        # predicted labels
        y_pred = self.model.predict(X, batch_size=constants.PRED_BATCH_SIZE)
        y_pred = np.asarray(y_pred.argmax(axis=-1)).ravel()

        # sanity check
        if not y_true.shape == y_pred.shape:
            err_msg = "'y_true' and 'y_pred' have different shapes"
            LOGGER.error('AssertionError: %s', err_msg)
            raise AssertionError(err_msg)

        return y_true, y_pred

    @staticmethod
    def get_precision_recall_f1_support(y_true, y_pred, criteria):
        """Returns precision, recall, f1 and support.

        For given gold (`y_true`) and predicted (`y_pred`) labels, returns the precision, recall,
        f1 and support per label, and the average of these scores across labels. Expects `y_true`
        and `y_pred` to be a sequence of entity chunks.

        Args:
            y_true: List of (chunk_type, chunk_start, chunk_end).
            y_pred: List of (chunk_type, chunk_start, chunk_end).
            criteria (str): Criteria to use for evaluation, 'exact' matches boundaries directly,
            'left' requires only a left boundary match and 'right' requires only a right boundary
            match.
        Returns:
            A dictionary of label, score key, value pairs where label is a class tag and scores is
            a 4-tuple containing precision, recall, f1 and support

        Raises:
            ValueError, if `criteria` is not one of 'exact', 'left', or 'right'
        """
        performance_scores = {}
        FN_total, FP_total, TP_total = 0, 0, 0 # micro performance accumulators
        labels = list(set([chunk[0] for chunk in y_true])) # unique labels

        # accumulate performance scores per label
        for lab in labels:
            y_pred_lab = [], []
            # either retain or discard left or right boundaries depending on matching criteria
            if criteria not in ['exact', 'left', 'right']:
                err_msg = ("Expected criteria to be one of 'exact', 'left', or 'right'. "
                           "Got: {}").format(criteria)
                LOGGER.error("ValueError %s", err_msg)
                raise ValueError(err_msg)
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
            FN, FP, TP = 0, 0, 0
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
            performance_scores[lab] = model_utils.precision_recall_f1_support(TP, FP, FN)

            # accumulate FNs, FPs, TPs
            FN_total += FN
            FP_total += FP
            TP_total += TP

        # get macro and micro performance metrics averages
        macro_p = mean([v[0] for v in performance_scores.values()])
        macro_r = mean([v[1] for v in performance_scores.values()])
        macro_f1 = mean([v[2] for v in performance_scores.values()])
        total_support = TP_total + FN_total

        performance_scores['MACRO_AVG'] = (macro_p, macro_r, macro_f1, total_support)
        performance_scores['MICRO_AVG'] = model_utils.precision_recall_f1_support(TP_total,
                                                                                  FP_total,
                                                                                  FN_total)

        return performance_scores

    @staticmethod
    def print_performance_scores(performance_scores, title=None):
        """Prints an ASCII table of performance scores.

        Args:
            performance_scores: A dictionary of label, score pairs where label is a class tag and
                scores is a 4-tuple containing precision, recall, f1 and support
            title (str): The title of the table (uppercased).

        Preconditions:
            Assumes the values of performance_scores are 4-tuples, where the first three items are
            float representaions of a percentage and the last item is an count integer.
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
        """Write performance metrics to disk as json-formatted *.txt file.

        At the end of each epoch, writes a json-formatted *.txt file to disk
        (name 'epoch_<epoch_number>.txt'). File contains performance scores per label as well as the
        best-achieved macro and micro averages thus far for the current epoch.
        """
        # create evaluation output directory
        current_fold = self.__dict__.get("fold")
        eval_dirname = self.output_dir
        if current_fold is not None:
            fold = 'fold_{}'.format(current_fold + 1)
            eval_dirname = os.path.join(self.output_dir, fold)
        make_dir(eval_dirname)

        # create filepath to evaluation file
        eval_filename = 'epoch_{0:03d}.txt'.format(self.current_epoch + 1)
        eval_filepath = os.path.join(eval_dirname, eval_filename)

        # per partition performance metrics accumulator
        performance_metrics = {p: {} for p in constants.PARTITIONS}

        for partition in self.performance_metrics:
            # test partition may be empty
            if self.performance_metrics[partition]:
                # get best epoch based on macro / micro averages
                macro_avg_per_epoch = [x['MACRO_AVG'] for x in self.performance_metrics[partition]]
                micro_avg_per_epoch = [x['MICRO_AVG'] for x in self.performance_metrics[partition]]

                best_macro_avg = max(macro_avg_per_epoch, key=itemgetter(2))
                best_micro_avg = max(micro_avg_per_epoch, key=itemgetter(2))

                best_micro_epoch = micro_avg_per_epoch.index(best_micro_avg) + 1
                best_macro_epoch = macro_avg_per_epoch.index(best_macro_avg) + 1

                performance_metrics[partition]['scores'] = \
                    self.performance_metrics[partition][self.current_epoch]
                performance_metrics[partition]['best_epoch_macro_avg'] = {'epoch': best_macro_epoch,
                                                                          'scores': best_macro_avg}
                performance_metrics[partition]['best_epoch_micro_avg'] = {'epoch': best_micro_epoch,
                                                                          'scores': best_micro_avg}

        # write performance metrics for current epoch to file
        with open(eval_filepath, 'a') as eval_file:
            eval_file.write(json.dumps(performance_metrics, indent=4))
