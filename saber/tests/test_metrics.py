"""Any and all unit tests for the Metrics class (saber/metrics.py).
"""
import pytest

from .. import constants
from ..metrics import Metrics


class TestMetrics(object):
    """Collects all unit tests for `saber.metrics.Metrics`.
    """
    def test_attributes_after_initilization(self,
                                            dummy_config,
                                            conll2003datasetreader_load,
                                            dummy_output_dir,
                                            dummy_training_data,
                                            dummy_metrics):
        """Asserts instance attributes are as expected."""
        # attributes that are passed to __init__
        assert dummy_metrics.config is dummy_config
        assert dummy_metrics.training_data is dummy_training_data
        assert dummy_metrics.idx_to_tag is conll2003datasetreader_load.idx_to_tag
        assert dummy_metrics.output_dir == dummy_output_dir
        # other instance attributes
        assert dummy_metrics.current_epoch == 0
        assert dummy_metrics.performance_metrics == {p: [] for p in constants.PARTITIONS}
        # test that we can pass arbitrary keyword arguments
        assert dummy_metrics.totally_arbitrary == 'arbitrary'

    def test_precision_recall_f1_support_value_error(self):
        """Asserts that call to `Metrics.get_precision_recall_f1_support` raises a `ValueError`
        error when an invalid value for parameter `criteria` is passed."""
        # these are totally arbitrary
        y_true = [('test', 0, 3), ('test', 4, 7), ('test', 8, 11)]
        y_pred = [('test', 0, 3), ('test', 4, 7), ('test', 8, 11)]

        # anything but 'exact', 'left', or 'right' should throw an error
        invalid_args = ['right ', 'LEFT', 'eXact', 0, []]

        for arg in invalid_args:
            with pytest.raises(ValueError):
                Metrics.get_precision_recall_f1_support(y_true, y_pred, criteria=arg)
