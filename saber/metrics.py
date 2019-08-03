"""Contains the Metrics class, which computes, stores, prints and saves performance metrics
for a Keras model.
"""
import copy
import itertools
import json
import logging
import pathlib
from statistics import mean

from prettytable import PrettyTable
from sklearn.metrics import precision_recall_fscore_support

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics.sequence_labeling import get_entities

from .constants import NEG
from .constants import OUTSIDE
from .constants import PARTITIONS

LOGGER = logging.getLogger(__name__)


class Metrics(object):
    """A class for handling performance metrics, inherits from Callback.

    Args:
        config (Config): Contains a set of harmonized arguments provided in a *.ini file and,
            optionally, from the command line.
        model_ (BaseModel): Model to evaluate, subclass of BaseModel.
        training_data (dict): Contains the data (at key `x_partition`) and targets
            (at key `y_partition`) for each partition: 'train', 'valid' and 'test'.
        idx_to_tag (dict): A dictionary mapping unique integers to labels.
        output_dir (str): Base directory to save all output to.
    """
    def __init__(self, config, model_, training_data, idx_to_tag, output_dir, **kwargs):
        self.config = config  # hyperparameters and model details
        self.model_ = model_   # _ prevents naming collision
        self.training_data = training_data  # inputs and targets for each partition
        self.idx_to_tag = idx_to_tag  # maps unique IDs to targets

        self.output_dir = output_dir

        # Current epoch and fold counters
        self.epoch = 0
        self.fold = 0

        # Model performance metrics accumulators
        self.evaluations = {
            p: {'scores': [], 'best_macro_f1': {}, 'best_micro_f1': {}} for p in PARTITIONS
        }

        # If we are preforming cross-validation, add a "folds" field to the JSON.
        if len(training_data) > 1:
            self.evaluations = {
                'folds': [{'fold': fold + 1, **copy.deepcopy(self.evaluations)}
                          for fold in range(len(training_data))]
            }

        self.evaluation_filepath = pathlib.Path().joinpath(output_dir, 'evaluation.json')

        for key, value in kwargs.items():
            setattr(self, key, value)

    def on_epoch_end(self):
        """Computes, accumulates and prints train/valid/test scores at the end of each epoch.
        """
        training_data = self.training_data[self.fold]

        for partition in PARTITIONS:
            if training_data[partition] is not None:
                evaluation = self._evaluate(training_data, partition=partition)
                evaluation_json = {'epoch': self.epoch + 1}

                for task, scores in evaluation.items():
                    # TODO (John): Should be controlled by `self.config.verbose`
                    _ = self.print_evaluation(scores, title=f'{task.upper()} ({partition.title()})')

                    # Get the evaluation into a json like format
                    evaluation_json[task] = {
                        label: {
                            'precision': float(score[0]),
                            'recall': float(score[1]),
                            'f1': float(score[2]),
                            'support': int(score[3])
                        } for label, score in scores.items()
                    }

                self._update_best_epoch(evaluation_json, partition)

                if 'folds' in self.evaluations:
                    self.evaluations['folds'][self.fold][partition]['scores'].append(evaluation_json)
                else:
                    self.evaluations[partition]['scores'].append(evaluation_json)

        self._write_evaluations_to_disk()
        self.epoch += 1

    def on_fold_end(self):
        """Bumps k-fold cross-validation counter and resets epoch counter at the end of a fold.
        """
        self.epoch = 0
        self.fold += 1

    @staticmethod
    def precision_recall_f1_support_sequence_labelling(y_true, y_pred, criteria='exact'):
        """Compute precision, recall, f1 and support for sequence labelling tasks.

        For given gold (`y_true`) and predicted (`y_pred`) sequence labels, returns the precision,
        recall, f1 and support per label, and the macro and micro average of these scores across
        labels. Expects `y_true` and `y_pred` to be a sequence of IOB1/2, IOE1/2, or IOBES formatted
        labels.

        Args:
            y_true (list): List of IOB1/2, IOE1/2, or IOBES formatted sequence labels.
            y_pred (list): List of IOB1/2, IOE1/2, or IOBES formatted sequence labels.
            criteria (str): Optional, criteria which will be used for evaluation. 'exact' matches
                boundaries exactly, 'left' requires only a left boundary match and 'right' requires
                only a right boundary match. Defaults to 'exact'.

        Returns:
            A dictionary of scores keyed by the labels in `y_true` where each score is a 4-tuple
            containing precision, recall, f1 and support. Additionally includes the keys
            'Macro avg' and 'Micro avg' containing the macro and micro averages across scores.

        Raises:
            ValueError, if `criteria` is not one of 'exact', 'left', or 'right'.
        """
        if criteria not in ['exact', 'left', 'right']:
            err_msg = ("Expected criteria to be one of 'exact', 'left', or 'right'."
                       " Got: {}").format(criteria)
            LOGGER.error("ValueError %s", err_msg)
            raise ValueError(err_msg)

        scores = {}
        # Unique labels, not including NEG
        labels = list({tag.split('-')[-1] for tag in set(y_true) if tag != OUTSIDE})
        labels.sort()  # ensures labels displayed in same order across runs / partitions

        for label in labels:
            y_true_lab = [tag if tag.endswith(label) else OUTSIDE for tag in y_true]
            y_pred_lab = [tag if tag.endswith(label) else OUTSIDE for tag in y_pred]

            # TODO (John): Open a pull request to seqeval with a new function that returns all these
            # scores in one call. There is a lot of repeated computation here.
            precision = precision_score(y_true_lab, y_pred_lab, criteria=criteria)
            recall = recall_score(y_true_lab, y_pred_lab, criteria=criteria)
            f1 = f1_score(y_true_lab, y_pred_lab, criteria=criteria)
            support = len(set(get_entities(y_true_lab)))

            scores[label] = precision, recall, f1, support

        # Get macro and micro performance metrics averages
        macro_precision = mean([v[0] for v in scores.values()])
        macro_recall = mean([v[1] for v in scores.values()])
        macro_f1 = mean([v[2] for v in scores.values()])
        total_support = sum([v[3] for v in scores.values()])

        micro_precision = precision_score(y_true, y_pred, criteria=criteria)
        micro_recall = recall_score(y_true, y_pred, criteria=criteria)
        micro_f1 = f1_score(y_true, y_pred, criteria=criteria)

        scores['Macro avg'] = macro_precision, macro_recall, macro_f1, total_support
        scores['Micro avg'] = micro_precision, micro_recall, micro_f1, total_support

        return scores

    @staticmethod
    def precision_recall_f1_support_multi_class(y_true, y_pred):
        """Compute precision, recall, f1 and support for simple multi-class tasks.

        For given gold (`y_true`) and predicted (`y_pred`) labels, returns the precision,
        recall, f1 and support per label, and the macro and micro average of these scores across
        labels. Expects `y_true` and `y_pred` to be a sequence multi-class targets and predictions.

        Args:
            y_true (list): List of multi-class targets.
            y_pred (list): List of multi-class predictions.

        Returns:
            A dictionary of scores keyed by the labels in `y_true` where each score is a 4-tuple
            containing precision, recall, f1 and support. Additionally includes the keys
            'Macro avg' and 'Micro avg' containing the macro and micro averages across scores.

        Raises:
            ValueError, if `criteria` is not one of 'exact', 'left', or 'right'.
        """
        scores = {}
        # Unique labels, not including NEG
        labels = [label for label in set(y_true) if label != NEG]
        labels.sort()  # ensures labels displayed in same order across runs / partitions

        precision, recall, f1, support = \
            precision_recall_fscore_support(y_true, y_pred, labels=labels)

        # TODO (John): Do I really need to call this function 3 times?
        # Get macro and micro performance metrics averages
        macro_precision, macro_recall, macro_f1, _ = \
            precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro')
        micro_precision, micro_recall, micro_f1, _ = \
            precision_recall_fscore_support(y_true, y_pred, labels=labels, average='micro')

        total_support = 0
        for i, label in enumerate(labels):
            scores[label] = precision[i], recall[i], f1[i], support[i]
            total_support += support[i]

        scores['Macro avg'] = macro_precision, macro_recall, macro_f1, total_support
        scores['Micro avg'] = micro_precision, micro_recall, micro_f1, total_support

        return scores

    @staticmethod
    def print_evaluation(evaluation, title=None):
        """Prints an ASCII table of evaluation scores.

        Args:
            evaluation: A dictionary of label, score pairs where label is a class tag and
                scores is a 4-tuple containing precision, recall, f1 and support.
            title (str): Optional, the title of the table.

        Preconditions:
            Assumes the values of `evaluation` are 4-tuples, where the first three items are
            float representaions of a percentage and the last item is an count integer.
        """
        # Create table, give it a title a column names
        table = PrettyTable()

        if title is not None:
            table.title = title

        table.field_names = ['Label', 'Precision', 'Recall', 'F1', 'Support']

        # Column alignment
        table.align['Label'] = 'l'
        table.align['Precision'] = 'r'
        table.align['Recall'] = 'r'
        table.align['F1'] = 'r'
        table.align['Support'] = 'r'

        # Create and add the rows
        for label, scores in evaluation.items():
            row = [label]
            # convert scores to formatted percentage strings
            support = scores[-1]
            performance_metrics = [f'{x:.2%}' for x in scores[:-1]]
            row_scores = performance_metrics + [support]

            row.extend(row_scores)
            table.add_row(row)

        print(table)

        return table

    def _evaluate(self, training_data, partition='train'):
        """Performs a prediction step on `training_data` with `self.model_`.

        Performs a prediction step for the given inputs (`self.training_data[partition]['x']`) and
        targets (`self.training_data[partition]['y']`) using `self.model_`. Returns a dictionary
        keyed by the task (e.g. 'ner', 're').

        Args:
            partition (str): Which partition to perform a prediction step on, must be one of
                'train', 'valid', or 'test'.

        Returns:
            A dictionary of label, score pairs; where label is a class tag and scores is a 4-tuple
            containing precision, recall, f1 and support for that class.
        """
        eval_scores = {}

        # Get predictions and gold labels
        eval_results = self.model_.evaluate(training_data, partition=partition)

        # TODO (John): This is brittle.
        if len(eval_results) > 2:
            y_true_ner, y_pred_ner, y_true_rc, y_pred_rc = eval_results
        else:
            y_true_ner, y_pred_ner, y_true_rc, y_pred_rc = *eval_results, None, None

        # Flatten the lists
        y_true_ner = list(itertools.chain(*y_true_ner))
        y_pred_ner = list(itertools.chain(*y_pred_ner))

        # Get performance scores for NER
        eval_scores['ner'] = self.precision_recall_f1_support_sequence_labelling(
            y_true=y_true_ner,
            y_pred=y_pred_ner,
            criteria=self.config.criteria
        )

        # Get performance scores for RC
        if y_true_rc is not None:
            # Flatten the lists
            y_true_rc = list(itertools.chain(*y_true_rc))
            y_pred_rc = list(itertools.chain(*y_pred_rc))

            eval_scores['re'] = self.precision_recall_f1_support_multi_class(y_true=y_true_rc,
                                                                             y_pred=y_pred_rc)

        return eval_scores

    def _update_best_epoch(self, current_evaluation, partition='train'):
        """Updates the 'best_macro_f1' and 'best_micro_f1' fields of `self.evaluations`.

        Args:
            current_evaluation (dict): A JSON formatted dict containing the most recent evaluation.
            partition (str): Which partition to perform a prediction step on, must be one of
                'train', 'valid', or 'test'.
        """
        if 'folds' in self.evaluations:
            current_best = self.evaluations['folds'][self.fold][partition]
        else:
            current_best = self.evaluations[partition]

        for task, evaluation in current_evaluation.items():
            if task != 'epoch':
                # If this is the first epoch, add "best_macro/micro_f1" fields for each task
                if self.epoch == 0:
                    current_best['best_macro_f1'][task] = {'epoch': self.epoch + 1,
                                                           'scores': evaluation['Macro avg']}
                    current_best['best_micro_f1'][task] = {'epoch': self.epoch + 1,
                                                           'scores': evaluation['Micro avg']}
                else:
                    if (evaluation['Macro avg']['f1'] >
                            current_best['best_macro_f1'][task]['scores']['f1']):
                        current_best['best_macro_f1'][task]['epoch'] = self.epoch + 1
                        current_best['best_macro_f1'][task]['scores'] = evaluation['Macro avg']

                    if (evaluation['Micro avg']['f1'] >
                            current_best['best_micro_f1'][task]['scores']['f1']):
                        current_best['best_micro_f1'][task]['epoch'] = self.epoch + 1
                        current_best['best_micro_f1'][task]['scores'] = evaluation['Micro avg']

    def _write_evaluations_to_disk(self):
        """Write accumulated, json-formatted `evaluations` to disk.

        Writes a json file to disk with the accumulated evaluations at `self.evaluations`. If
        `self.config.k_folds is None`, this is a single file at filepath
        `self.output_dir/evaluations.json`, otherwise one file per `k` fold is saved at filepath
        `self.output_dir/evaluation_fold_k.json`.
        """
        with open(self.evaluation_filepath, 'w') as f:
            json.dump(self.evaluations, f, indent=2)
