import pytest

from ..utils.model_utils import precision_recall_f1_support

def test_precision_recall_f1_support():
    """Asserts that precision_recall_f1_support returns the expected values."""
    TP_dummy = 100
    FP_dummy = 10
    FN_dummy = 20

    prec_dummy = TP_dummy / (TP_dummy + FP_dummy)
    rec_dummy = TP_dummy / (TP_dummy + FN_dummy)
    f1_dummy = 2 * prec_dummy * rec_dummy / (prec_dummy + rec_dummy)
    support_dummy = TP_dummy + FN_dummy

    test_scores_no_null = precision_recall_f1_support(TP_dummy, FP_dummy, FN_dummy)
    test_scores_TP_null = precision_recall_f1_support(0, FP_dummy, FN_dummy)
    test_scores_FP_null = precision_recall_f1_support(TP_dummy, 0, FN_dummy)
    f1_FP_null = 2 * 1. * rec_dummy / (1. + rec_dummy)
    test_scores_FN_null = precision_recall_f1_support(TP_dummy, FP_dummy, 0)
    f1_FN_null = 2 * prec_dummy * 1. / (prec_dummy + 1.)
    test_scores_all_null = precision_recall_f1_support(0, 0, 0)

    assert test_scores_no_null == (prec_dummy, rec_dummy, f1_dummy, support_dummy)
    assert test_scores_TP_null == (0., 0., 0., FN_dummy)
    assert test_scores_FP_null == (1., rec_dummy, f1_FP_null, support_dummy)
    assert test_scores_FN_null == (prec_dummy, 1., f1_FN_null, TP_dummy)
    assert test_scores_all_null == (0., 0., 0., 0)
