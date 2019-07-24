"""Any and all unit tests for the Metrics class (saber/metrics.py).
"""
import os

import pytest

from ..constants import NEG
from ..constants import PARTITIONS
from ..metrics import Metrics

evaluations = {p: {'scores': [], 'best_macro_f1': {}, 'best_micro_f1': {}} for p in PARTITIONS}


class TestMetrics(object):
    """Collects all unit tests for `saber.metrics.Metrics`.
    """
    def test_attributes_after_initilization(self,
                                            dummy_config,
                                            bilstm_crf_model_specify,
                                            conll2003datasetreader_load,
                                            dummy_training_data,
                                            dummy_output_dir,
                                            dummy_metrics):
        """Asserts instance attributes are as expected."""
        # attributes that are passed to __init__
        assert dummy_metrics.config is dummy_config
        assert dummy_metrics.model_ is bilstm_crf_model_specify
        assert dummy_metrics.training_data is dummy_training_data
        assert dummy_metrics.idx_to_tag is conll2003datasetreader_load.idx_to_tag
        assert dummy_metrics.output_dir == dummy_output_dir[0]
        # other instance attributes
        assert dummy_metrics.epoch == 0
        assert dummy_metrics.fold == 0
        assert dummy_metrics.evaluations == evaluations
        assert dummy_metrics.model_idx == 0
        # test that we can pass arbitrary keyword arguments
        assert dummy_metrics.totally_arbitrary == 'arbitrary'

    def test_on_fold_end(self, dummy_metrics):
        """Asserts that the attributes `epoch` and `fold` are updated as expected when
        `on_epoch_fold()` is called.
        """
        assert dummy_metrics.epoch == 0
        assert dummy_metrics.fold == 0

        dummy_metrics.on_fold_end()

        assert dummy_metrics.epoch == 0
        assert dummy_metrics.fold == 1

    def test_precision_recall_f1_support_sequence_labelling_value_error(self):
        """Asserts that call to `Metrics.get_precision_recall_f1_support` raises a `ValueError`
        error when an invalid value for parameter `criteria` is passed."""
        # These are totally arbitrary
        y_true = ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'I-CHED', 'O', 'O']
        y_pred = ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'I-CHED', 'O', 'O']

        # Anything but 'exact', 'left', or 'right' should throw an error
        invalid_args = ['right ', 'LEFT', 'eXact', 0, []]

        for arg in invalid_args:
            with pytest.raises(ValueError):
                Metrics.precision_recall_f1_support_sequence_labelling(y_true=y_true,
                                                                       y_pred=y_pred,
                                                                       criteria=arg)

    def test_precision_recall_f1_support_sequence_labelling_exact_boundary_matching(self):
        """Asserts that call to `Metrics.precision_recall_f1_support_sequence_labelling` returns the
        expected object when `criteria='exact'`.
        """
        y_true = \
            ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'I-CHED', 'O', 'O', 'B-LIVB', 'I-LIVB']
        y_pred = \
            ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'O', 'O', 'B-LIVB', 'I-LIVB', 'I-LIVB']

        expected = {
            'PRGE': (1.0, 1.0, 1.0, 1),
            'CHED': (0.0, 0.0, 0.0, 1),
            'LIVB': (0.0, 0.0, 0.0, 1),
            'Macro avg': (1/3, 1/3, 1/3, 3),
            'Micro avg': (1/3, 1/3, 1/3, 3),
        }
        actual = Metrics.precision_recall_f1_support_sequence_labelling(y_true=y_true,
                                                                        y_pred=y_pred,
                                                                        criteria='exact')
        assert expected == actual

    def test_precision_recall_f1_support_sequence_labelling_left_boundary_matching(self):
        """Asserts that call to `Metrics.precision_recall_f1_support_sequence_labelling` returns the
        expected object when `criteria='left'`.
        """
        y_true = \
            ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'I-CHED', 'O', 'O', 'B-LIVB', 'I-LIVB']
        y_pred = \
            ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'O', 'O', 'B-LIVB', 'I-LIVB', 'I-LIVB']

        expected = {
            'PRGE': (1.0, 1.0, 1.0, 1),
            'CHED': (1.0, 1.0, 1.0, 1),
            'LIVB': (0.0, 0.0, 0.0, 1),
            'Macro avg': (2/3, 2/3, 2/3, 3),
            'Micro avg': (2/3, 2/3, 2/3, 3),
        }
        actual = Metrics.precision_recall_f1_support_sequence_labelling(y_true=y_true,
                                                                        y_pred=y_pred,
                                                                        criteria='left')
        assert expected == actual

    def test_precision_recall_f1_support_sequence_labelling_right_boundary_matching(self):
        """Asserts that call to `Metrics.precision_recall_f1_support_sequence_labelling` returns the
        expected object when `criteria='right'`.
        """
        y_true = \
            ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'I-CHED', 'O', 'O', 'B-LIVB', 'I-LIVB']
        y_pred = \
            ['B-PRGE', 'I-PRGE', 'I-PRGE', 'O', 'B-CHED', 'O', 'O', 'B-LIVB', 'I-LIVB', 'I-LIVB']

        expected = {
            'PRGE': (1.0, 1.0, 1.0, 1),
            'CHED': (0.0, 0.0, 0.0, 1),
            'LIVB': (1.0, 1.0, 1.0, 1),
            'Macro avg': (2/3, 2/3, 2/3, 3),
            'Micro avg': (2/3, 2/3, 2/3, 3),
        }
        actual = Metrics.precision_recall_f1_support_sequence_labelling(y_true=y_true,
                                                                        y_pred=y_pred,
                                                                        criteria='right')
        assert expected == actual

    def test_precision_recall_f1_support_multi_class(self):
        """Asserts that call to `Metrics.precision_recall_f1_support_multi_class` returns the
        expected object.
        """
        y_true = ['Live_In', NEG, NEG, NEG, 'OrgBased_In', NEG, 'Located_In']
        y_pred = ['Live_In', NEG, NEG, NEG, NEG, NEG, 'Located_In']

        expected = {
            'Live_In': (1.0, 1.0, 1.0, 1),
            'OrgBased_In': (0.0, 0.0, 0.0, 1),
            'Located_In': (1.0, 1.0, 1.0, 1),
            'Macro avg': (2/3, 2/3, 2/3, 3),
            'Micro avg': (1.0, 2/3, 0.8, 3),
        }
        actual = Metrics.precision_recall_f1_support_multi_class(y_true=y_true, y_pred=y_pred)

        assert expected == actual

    def test_evaluate(self, dummy_training_data, dummy_metrics):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if evaluation was unsuccesful and therefore alert us when a code
        change has broke the evaluation loop.
        """
        # _evaluate() exects one fold of one dataset
        training_data = dummy_training_data[0][0]

        _ = dummy_metrics._evaluate(training_data)

    def test_write_evaluations_to_disk(self, dummy_metrics):
        """Asserts that `_write_evaluations_to_disk()` rights a file to disk with the expected
        filename.
        """
        output_filepath = os.path.join(dummy_metrics.output_dir, 'evaluation.json')

        dummy_metrics._write_evaluations_to_disk()

        assert os.path.isfile(output_filepath)
